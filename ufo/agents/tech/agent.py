# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from ufo.agents.common.buffers.storage import ZBuffer
from ufo.agents.common.checkpoint import load_agent, save_agent
from ufo.agents.common.execution import AgentExecutionMixin, gradient_penalty_wgan, make_adam
from ufo.agents.common.networks import eval_mode, weight_init

from .config import TeCHAgentConfig


def _roll_obs(obs: torch.Tensor | dict[str, torch.Tensor], shifts: int = 1, dim: int = 0):
    if isinstance(obs, torch.Tensor):
        return torch.roll(obs, shifts=shifts, dims=dim)
    return tree_map(lambda x: torch.roll(x, shifts=shifts, dims=dim), obs)


class TeCHAgent(AgentExecutionMixin):
    """Direct TeCH update pipeline, including temporal embedding pretraining."""

    config_class = TeCHAgentConfig

    def __init__(self, obs_space, action_dim, cfg: TeCHAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        seq_length = cfg.model.seq_length
        batch_size = cfg.train.batch_size
        assert (batch_size / seq_length) == (batch_size // seq_length), "Batch size should be divisible by seq_length"
        self._model = self.cfg.model.build(obs_space, action_dim)
        self.setup_training()
        self.env_idx_with_expert_rollout = None

        self._base_goal_encoder_lr = float(cfg.train.lr_goal_encoder)

    @property
    def device(self):
        return self._model.device

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()

        t = self.cfg.train
        self.goal_encoder_optimizer = make_adam(
            self._model._goal_encoder.parameters(),
            self.device,
            lr=t.lr_goal_encoder,
            weight_decay=t.weight_decay,
        )
        self.actor_optimizer = make_adam(
            self._model._actor.parameters(),
            self.device,
            lr=t.lr_actor,
            weight_decay=t.weight_decay,
        )
        self.critic_optimizer = make_adam(
            self._model._critic.parameters(),
            self.device,
            lr=t.lr_critic,
            weight_decay=t.weight_decay,
        )

        self._critic_paramlist = tuple(x for x in self._model._critic.parameters())
        self._target_critic_paramlist = tuple(x for x in self._model._target_critic.parameters())
        self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self._model.device)
        self._goal_encoder_frozen = False

        self.discriminator_optimizer = make_adam(
            self._model._discriminator.parameters(),
            self.device,
            lr=self.cfg.train.lr_discriminator,
            weight_decay=self.cfg.train.weight_decay_discriminator,
        )

        self._aux_critic_map_paramlist = tuple(x for x in self._model._aux_critic.parameters())
        self._aux_target_critic_map_paramlist = tuple(x for x in self._model._target_aux_critic.parameters())

        self.aux_critic_optimizer = make_adam(
            self._model._aux_critic.parameters(),
            self.device,
            lr=self.cfg.train.lr_aux_critic,
            weight_decay=self.cfg.train.weight_decay,
        )

        t = self.cfg.train
        dual_lam_log = torch.tensor(math.log(max(t.dual_lam_init, 1e-12)), dtype=torch.float32, device=self.device)
        self.dual_lam_log_param = torch.nn.Parameter(dual_lam_log)

        self.dual_lam_optimizer = make_adam(
            [self.dual_lam_log_param],
            self.device,
            lr=t.lr_dual_lam,
            weight_decay=0.0,
        )


    @property
    def optimizers(self):
        return {
            "actor_optimizer": self.actor_optimizer,
            "goal_encoder_optimizer": self.goal_encoder_optimizer,
            "dual_lam_optimizer": self.dual_lam_optimizer,
            "critic_optimizer": self.critic_optimizer,
            "discriminator_optimizer": self.discriminator_optimizer,
            "aux_critic_optimizer": self.aux_critic_optimizer,
        }

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        self._apply_goal_encoder_lr_schedule(step)
        pre = self._maybe_tech_pretrain_only(replay_buffer, step)
        if pre is not None:
            pre.update(
                {
                    "disc_loss": torch.tensor(0.0, device=self.device),
                    "disc_expert_loss": torch.tensor(0.0, device=self.device),
                    "disc_train_loss": torch.tensor(0.0, device=self.device),
                    "disc_expert_logit": torch.tensor(0.0, device=self.device),
                    "disc_train_logit": torch.tensor(0.0, device=self.device),
                    "mean_disc_reward": torch.tensor(0.0, device=self.device),
                    "aux_critic_loss": torch.tensor(0.0, device=self.device),
                    "mean_aux_reward": torch.tensor(0.0, device=self.device),
                    "mean_total_reward": torch.tensor(0.0, device=self.device),
                }
            )
            return pre

        self._maybe_freeze_goal_encoder(step)

        expert_batch = replay_buffer["expert"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs_raw = tree_map(lambda x: x.to(self.device), train_batch["observation"])
        train_action = train_batch["action"].to(self.device)
        train_next_obs_raw = tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"])
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)

        expert_obs_raw = tree_map(lambda x: x.to(self.device), expert_batch["observation"])
        expert_next_obs_raw = tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"])

        self._model._obs_normalizer(train_obs_raw)
        self._model._obs_normalizer(train_next_obs_raw)
        self._model._obs_normalizer(expert_obs_raw)
        self._model._obs_normalizer(expert_next_obs_raw)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs = self._model._obs_normalizer(train_obs_raw)
            train_next_obs = self._model._obs_normalizer(train_next_obs_raw)
            expert_obs = self._model._obs_normalizer(expert_obs_raw)

        tech_te_metrics = None
        if self.cfg.train.tech_te_during_rl and (not self._goal_encoder_frozen) and step >= self.cfg.train.tech_pretrain_env_steps:
            tech_te_metrics = self.update_tech_te(train_obs_raw=train_obs_raw, train_next_obs_raw=train_next_obs_raw)


        expert_z = self.encode_expert(next_obs=expert_next_obs_raw)
        train_z = train_batch["z"].to(self.device)

        z_mixed = self.sample_mixed_z(train_goal=train_next_obs_raw, expert_encodings=expert_z).clone()
        self.z_buffer.add(z_mixed)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z_mixed, train_z)

        disc_reward = self._model._discriminator.compute_reward(obs=train_obs, z=train_z)

        with torch.no_grad():
            cur_z = self._model.project_z(self._model._goal_encoder(train_obs)).float()
            next_z = self._model.project_z(self._model._goal_encoder(train_next_obs)).float()
            goal_z = train_z.float()
            tech_reward = torch.norm(goal_z - cur_z, dim=1) - torch.norm(goal_z - next_z, dim=1)
            if tech_reward.dim() == 1:
                tech_reward = tech_reward.unsqueeze(-1)
            tech_reward = tech_reward * self.cfg.train.tech_reward_scale

        total_reward = tech_reward + self.cfg.train.disc_reward_coef * disc_reward

        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None,
        )
        if tech_te_metrics is not None:
            metrics.update(tech_te_metrics)

        aux_reward = torch.zeros((self.cfg.train.batch_size, 1), device=self.device, dtype=torch.float32)
        for aux_reward_name in self.cfg.aux_rewards:
            metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
            aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name].to(self.device)
        aux_reward = self._model._aux_reward_normalizer(aux_reward)

        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        # The actor stays unchanged across both critic updates; sample separately below.
        with torch.no_grad():
            target_dist = self._model._actor(train_next_obs, train_z, self._model.cfg.actor_std)
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
                reward=total_reward,
                target_dist=target_dist,
            )
        )
        metrics.update(
            self.update_aux_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                aux_reward=aux_reward,
                next_obs=train_next_obs,
                z=train_z,
                target_dist=target_dist,
            )
        )
        metrics.update(self._run_actor_update(obs=train_obs, z=train_z, clip_grad_norm=clip_grad_norm))

        with torch.no_grad():
            from ufo.agents.common.networks import _soft_update_params

            _soft_update_params(
                self._critic_paramlist,
                self._target_critic_paramlist,
                self.cfg.train.critic_target_tau,
            )

        metrics["mean_disc_reward"] = disc_reward.mean().detach()
        metrics["mean_total_reward"] = total_reward.mean().detach()
        return metrics

    def _maybe_tech_pretrain_only(self, replay_buffer: dict, step: int) -> Dict[str, torch.Tensor] | None:
        if not self.cfg.train.use_tech_pretrain:
            return None
        if step >= self.cfg.train.tech_pretrain_env_steps:
            return None
        if len(replay_buffer["train"]) == 0:
            return {"tech_te_loss": torch.tensor(0.0, device=self.device)}

        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)
        train_obs_raw = tree_map(lambda x: x.to(self.device), train_batch["observation"])
        train_next_obs_raw = tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"])

        with torch.no_grad():
            _ = self._model._obs_normalizer(train_obs_raw)
            _ = self._model._obs_normalizer(train_next_obs_raw)

        return self.update_tech_te(train_obs_raw=train_obs_raw, train_next_obs_raw=train_next_obs_raw)

    def update_tech_te(self, train_obs_raw: torch.Tensor | dict[str, torch.Tensor], train_next_obs_raw) -> Dict[str, torch.Tensor]:
        t = self.cfg.train
        train_goals_raw = _roll_obs(train_next_obs_raw, shifts=1, dim=0)

        phi_x = self._model.encode_goal(train_obs_raw)
        phi_y = self._model.encode_goal(train_next_obs_raw)
        phi_g = self._model.encode_goal(train_goals_raw)

        phi_x = phi_x.float()
        phi_y = phi_y.float()
        phi_g = phi_g.float()

        squared_dist = ((phi_x - phi_g) ** 2).sum(dim=-1)
        dist = torch.sqrt(torch.clamp(squared_dist, min=1e-6))
        cst_penalty = 1.0 - torch.square(phi_y - phi_x).mean(dim=1)
        cst_penalty = torch.clamp(cst_penalty, max=t.dual_slack)

        dual_lam = self.dual_lam_log_param.exp()
        soft_term = -F.softplus(t.tech_softplus_scale - dist, beta=t.tech_softplus_beta).mean()
        dual_term = (dual_lam.detach() * cst_penalty).mean()
        te_obj = soft_term + dual_term
        loss_te = -te_obj

        self.goal_encoder_optimizer.zero_grad(set_to_none=True)
        loss_te.backward()
        self._sync_gradients(self._model._goal_encoder.parameters())
        self.goal_encoder_optimizer.step()

        loss_dual = torch.tensor(0.0, device=self.device)
        if t.dual_reg:
            loss_dual = self.dual_lam_log_param * cst_penalty.detach().mean()
            self.dual_lam_optimizer.zero_grad(set_to_none=True)
            loss_dual.backward()
            self._sync_gradients([self.dual_lam_log_param])
            self.dual_lam_optimizer.step()

        return {
            "tech_te_loss": loss_te.detach(),
            "tech_te_obj_mean": te_obj.detach(),
            "tech_dist_mean": dist.mean().detach(),
            "tech_cst_penalty_mean": cst_penalty.mean().detach(),
            "tech_dual_lam": dual_lam.detach(),
            "tech_dual_loss": loss_dual.detach(),
            "tech_soft_term": soft_term.detach(),
        }

    def update_discriminator(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        train_obs: torch.Tensor | dict[str, torch.Tensor],
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
        unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)
        expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
        unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
        loss = torch.mean(expert_loss + unlabeled_loss)
        if grad_penalty is not None:
            wgan_gp = gradient_penalty_wgan(self._model._discriminator, expert_obs, expert_z, train_obs, train_z)
            loss += grad_penalty * wgan_gp

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._sync_gradients(self._model._discriminator.parameters())
        self.discriminator_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean().detach(),
                "disc_train_loss": unlabeled_loss.detach().mean().detach(),
                "disc_expert_logit": expert_logits.mean().detach(),
                "disc_train_logit": unlabeled_logits.mean().detach(),
            }
            if grad_penalty is not None:
                output_metrics["disc_wgan_gp_loss"] = wgan_gp.detach()
        return output_metrics

    def update_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        reward: torch.Tensor,
        target_dist=None,
    ) -> Dict[str, torch.Tensor]:
        num_parallel = self.cfg.model.archi.critic.num_parallel
        with torch.no_grad():
            dist = target_dist if target_dist is not None else self._model._actor(next_obs, z, self._model.cfg.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)
            next_qs = self._model._target_critic(next_obs, z, next_action)
            q_mean, q_unc, next_v = self.get_targets_uncertainty(next_qs, self.cfg.train.critic_pessimism_penalty)
            target_q = reward + discount * next_v
            expanded_targets = target_q.expand(num_parallel, -1, -1)

        qs = self._model._critic(obs, z, action)
        critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded_targets)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._sync_gradients(self._model._critic.parameters())
        self.critic_optimizer.step()

        with torch.no_grad():
            return {
                "target_Q": target_q.mean().detach(),
                "Q1": qs.mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_next_Q": q_mean.mean().detach(),
                "unc_Q": q_unc.mean().detach(),
                "mean_reward_train": reward.mean().detach(),
            }

    def update_aux_critic(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        aux_reward: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        target_dist=None,
    ) -> Dict[str, torch.Tensor]:
        num_parallel = self.cfg.model.archi.critic.num_parallel
        with torch.no_grad():
            dist = target_dist if target_dist is not None else self._model._actor(next_obs, z, self._model.cfg.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)
            next_qs = self._model._target_aux_critic(next_obs, z, next_action)
            q_mean, q_unc, next_v = self.get_targets_uncertainty(next_qs, self.cfg.train.aux_critic_pessimism_penalty)
            target_q = aux_reward + discount * next_v
            expanded_targets = target_q.expand(num_parallel, -1, -1)

        qs = self._model._aux_critic(obs, z, action)
        aux_critic_loss = 0.5 * num_parallel * F.mse_loss(qs, expanded_targets)

        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        self._sync_gradients(self._model._aux_critic.parameters())
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_auxQ": target_q.mean().detach(),
                "auxQ1": qs.mean().detach(),
                "mean_next_auxQ": q_mean.mean().detach(),
                "unc_auxQ": q_unc.mean().detach(),
                "aux_critic_loss": aux_critic_loss.mean().detach(),
                "mean_aux_reward": aux_reward.mean().detach(),
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        dist = self._model._actor(obs, z, self._model.cfg.actor_std)
        action = dist.sample(clip=self.cfg.train.stddev_clip)

        qs = self._model._critic(obs, z, action)
        _, _, q = self.get_targets_uncertainty(qs, self.cfg.train.actor_pessimism_penalty)

        qs_aux = self._model._aux_critic(obs, z, action)
        _, _, q_aux = self.get_targets_uncertainty(qs_aux, self.cfg.train.actor_pessimism_penalty)

        disc_reward = self._model._discriminator.compute_reward(obs=obs, z=z)
        weight = q.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
        disc_actor_weight = self.cfg.train.reg_coeff_disc * weight
        aux_actor_weight = self.cfg.train.reg_coeff_aux * weight
        actor_loss = -q.mean() - disc_actor_weight * disc_reward.mean() - aux_actor_weight * q_aux.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._sync_gradients(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        return {
            "actor_loss": actor_loss.detach(),
            "Q_actor": q.mean().detach(),
            "Q_aux": q_aux.mean().detach(),
            "disc_reward_actor": disc_reward.mean().detach(),
            "weight/disc_actor": disc_actor_weight.detach(),
            "weight/aux_actor": aux_actor_weight.detach(),
            "weight/scale_q": weight.detach() if isinstance(weight, torch.Tensor) else torch.tensor(weight, device=self.device),
            "cfg/reg_coeff_disc": torch.tensor(self.cfg.train.reg_coeff_disc, device=self.device),
            "cfg/reg_coeff_aux": torch.tensor(self.cfg.train.reg_coeff_aux, device=self.device),
            "cfg/disc_reward_coef": torch.tensor(self.cfg.train.disc_reward_coef, device=self.device),
        }

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        preds_uns = preds.unsqueeze(dim=dim)
        preds_uns2 = preds.unsqueeze(dim=dim + 1)
        preds_diffs = torch.abs(preds_uns - preds_uns2)
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = preds_diffs.sum(dim=(dim, dim + 1)) / num_parallel_scaling
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    def _apply_goal_encoder_lr_schedule(self, step: int) -> None:
        t = self.cfg.train
        if t.goal_encoder_lr_schedule == "none":
            return
        total_steps = int(t.goal_encoder_lr_schedule_steps)
        if total_steps <= 0:
            return

        progress = min(max(float(step) / float(total_steps), 0.0), 1.0)
        min_lr = float(t.goal_encoder_lr_min)
        base_lr = self._base_goal_encoder_lr
        if t.goal_encoder_lr_schedule == "linear":
            new_lr = min_lr + (base_lr - min_lr) * (1.0 - progress)
        elif t.goal_encoder_lr_schedule == "cosine":
            new_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown goal_encoder_lr_schedule={t.goal_encoder_lr_schedule}")

        for group in self.goal_encoder_optimizer.param_groups:
            group["lr"] = float(new_lr)

    def _maybe_freeze_goal_encoder(self, step: int) -> None:
        if not self.cfg.train.freeze_goal_encoder_after_pretrain:
            return
        if not self.cfg.train.use_tech_pretrain:
            return
        if self._goal_encoder_frozen:
            return
        if step < self.cfg.train.tech_pretrain_env_steps:
            return
        for p in self._model._goal_encoder.parameters():
            p.requires_grad_(False)
        self._goal_encoder_frozen = True

    @torch.no_grad()
    def sample_mixed_z(
        self,
        train_goal: torch.Tensor | dict[str, torch.Tensor],
        expert_encodings: torch.Tensor,
    ) -> torch.Tensor:
        t = self.cfg.train
        b = self.cfg.train.batch_size

        z = self._model.sample_z(b, device=self._model.device)
        p_goal = float(t.train_goal_ratio)
        p_expert = float(t.expert_asm_ratio)
        p_random = 1.0 - p_goal - p_expert
        if p_goal < 0 or p_expert < 0 or p_random < 0:
            raise ValueError(
                "Invalid z-mix probabilities: "
                f"train_goal_ratio={p_goal}, expert_asm_ratio={p_expert}, random_ratio={p_random}. "
                "Require all >= 0 and train_goal_ratio + expert_asm_ratio <= 1."
            )
        prob = torch.tensor([p_goal, p_expert, p_random], dtype=torch.float32, device=self.device)
        mix_idxs = torch.multinomial(prob, num_samples=b, replacement=True).reshape(-1, 1)

        rolled_goal = _roll_obs(train_goal, shifts=1, dim=0)
        goals = self._model.encode_goal(rolled_goal)
        z = torch.where(mix_idxs == 0, goals, z)

        if p_expert > 0:
            expert_roll = torch.roll(expert_encodings, shifts=1, dims=0)
            z = torch.where(mix_idxs == 1, expert_roll, z)

        return z

    @torch.no_grad()
    def encode_expert(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        b_expert = self._model._goal_encoder(
            self._model._normalize(next_obs)  # type: ignore[arg-type]
        ).detach()
        b_expert = b_expert.view(
            self.cfg.train.batch_size // self.cfg.model.seq_length,
            self.cfg.model.seq_length,
            b_expert.shape[-1],
        )
        z_expert = b_expert.mean(dim=1)
        z_expert = self._model.project_z(z_expert)
        z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)
        return z_expert

    @torch.no_grad()
    def _sample_tracking_z(self, replay_buffer, batch_dim: int, traj_length: int) -> torch.Tensor:
        batch = replay_buffer["expert"].sample(batch_dim * traj_length, seq_length=traj_length)
        z = self._model.goal_inference(batch["next"]["observation"])
        z = z.view(batch_dim, traj_length, z.shape[-1])
        for step in range(traj_length):
            end_idx = min(step + self.cfg.model.seq_length, traj_length)
            z[:, step] = z[:, step:end_idx].mean(dim=1)
        return self._model.project_z(z)

    def maybe_update_rollout_context(
        self,
        z: torch.Tensor | None,
        step_count: torch.Tensor,
        replay_buffer: dict | None = None,
        global_env_step: int | None = None,
    ) -> torch.Tensor:
        del global_env_step
        if z is not None:
            mask_reset_z = step_count % self.cfg.train.update_z_every_step == 0
            if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0], device=self._model.device)
            else:
                new_z = self._model.sample_z(z.shape[0], device=self._model.device)
            z = torch.where(mask_reset_z, new_z, z.to(self._model.device))
            if self.cfg.train.rollout_expert_trajectories and replay_buffer is not None:
                idxs = step_count % self.cfg.train.rollout_expert_trajectories_length
                if torch.any(idxs == 0):
                    n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage * step_count.shape[0])
                    self.env_idx_with_expert_rollout = torch.randint(0, step_count.shape[0], size=(n_elem,), device=self._model.device)
                    self.tracking_z = self._sample_tracking_z(replay_buffer, n_elem, self.cfg.train.rollout_expert_trajectories_length)
                if self.env_idx_with_expert_rollout is not None:
                    mod_time = idxs[self.env_idx_with_expert_rollout].ravel()
                    z[self.env_idx_with_expert_rollout] = self.tracking_z[
                        torch.arange(len(self.env_idx_with_expert_rollout), device=self._model.device), mod_time
                    ]
        else:
            z = self._model.sample_z(step_count.shape[0], device=self._model.device)
            if self.cfg.train.rollout_expert_trajectories and replay_buffer is not None:
                n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage * step_count.shape[0])
                self.env_idx_with_expert_rollout = torch.randint(0, step_count.shape[0], size=(n_elem,), device=self._model.device)
                self.tracking_z = self._sample_tracking_z(replay_buffer, n_elem, self.cfg.train.rollout_expert_trajectories_length)
                z[self.env_idx_with_expert_rollout] = self.tracking_z[:, 0]
        return z

    def save(self, output_folder: str) -> None:
        save_agent(self, output_folder)
        torch.save(
            {"dual_lam_log_param": self.dual_lam_log_param.detach(), "goal_encoder_frozen": self._goal_encoder_frozen},
            Path(output_folder) / "training_state.pth",
        )

    @classmethod
    def load(cls, path: str, device: str | None = None):
        path = Path(path)
        agent = load_agent(cls.config_class, path, device)
        extra = path / "training_state.pth"
        state = torch.load(extra, weights_only=True, map_location=device)
        with torch.no_grad():
            agent.dual_lam_log_param.copy_(state["dual_lam_log_param"])
        agent._goal_encoder_frozen = state["goal_encoder_frozen"]
        if agent._goal_encoder_frozen:
            agent._model._goal_encoder.requires_grad_(False)
        return agent

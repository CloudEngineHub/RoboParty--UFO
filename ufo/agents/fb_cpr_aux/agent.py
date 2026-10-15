# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_map

from ufo.agents.common.buffers.storage import ZBuffer
from ufo.agents.common.checkpoint import load_agent, save_agent
from ufo.agents.common.execution import AgentExecutionMixin, gradient_penalty_wgan, make_adam
from ufo.agents.common.networks import _soft_update_params, eval_mode, weight_init

from .config import FBcprAuxAgentConfig


class FBcprAuxAgent(AgentExecutionMixin):
    """Direct FB-CPR-Aux training pipeline; no algorithm inheritance."""

    config_class = FBcprAuxAgentConfig

    def __init__(self, obs_space, action_dim, cfg: FBcprAuxAgentConfig):
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = cfg
        # make sure batch size is a multiple of seq_length
        seq_length = cfg.model.seq_length
        batch_size = cfg.train.batch_size
        assert (batch_size / seq_length) == (batch_size // seq_length), "Batch size should be divisable by seq_length"
        del seq_length, batch_size

        self._model = self.cfg.model.build(obs_space, action_dim)
        self.setup_training()
        self.env_idx_with_expert_rollout = None

    @property
    def device(self):
        return self._model.device

    def _autocast_context(self):
        # Only network forwards use BF16; losses and master weights stay FP32.
        return torch.autocast(
            torch.device(self.device).type,
            dtype=torch.bfloat16,
            enabled=self.cfg.train.use_bf16_amp,
        )

    def _discriminator_reward(self, obs, z):
        with self._autocast_context():
            logits = self._model._discriminator.compute_logits(obs=obs, z=z)
        probability = torch.sigmoid(logits.float()).clamp(1e-7, 1 - 1e-7)
        return probability.log() - (1 - probability).log()

    def setup_training(self) -> None:
        self._model.train(True)
        self._model.requires_grad_(True)
        self._model.apply(weight_init)
        self._model._prepare_for_train()  # ensure that target nets are initialized after applying the weights

        self.backward_optimizer = make_adam(
            self._model._backward_map.parameters(),
            self.device,
            lr=self.cfg.train.lr_b,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.forward_optimizer = make_adam(
            self._model._forward_map.parameters(),
            self.device,
            lr=self.cfg.train.lr_f,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.actor_optimizer = make_adam(
            self._model._actor.parameters(),
            self.device,
            lr=self.cfg.train.lr_actor,
            weight_decay=self.cfg.train.weight_decay,
        )

        # prepare parameter list
        self._forward_map_paramlist = tuple(x for x in self._model._forward_map.parameters())
        self._target_forward_map_paramlist = tuple(x for x in self._model._target_forward_map.parameters())
        self._backward_map_paramlist = tuple(x for x in self._model._backward_map.parameters())
        self._target_backward_map_paramlist = tuple(x for x in self._model._target_backward_map.parameters())

        # precompute some useful variables
        self.off_diag = 1 - torch.eye(self.cfg.train.batch_size, self.cfg.train.batch_size, device=self.device)
        self.off_diag_sum = self.off_diag.sum()

        self.z_buffer = ZBuffer(self.cfg.train.z_buffer_size, self.cfg.model.archi.z_dim, self._model.device)

        # prepare parameter list
        self._critic_map_paramlist = tuple(x for x in self._model._critic.parameters())
        self._target_critic_map_paramlist = tuple(x for x in self._model._target_critic.parameters())

        self.critic_optimizer = make_adam(
            self._model._critic.parameters(),
            self.device,
            lr=self.cfg.train.lr_critic,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.discriminator_optimizer = make_adam(
            self._model._discriminator.parameters(),
            self.device,
            lr=self.cfg.train.lr_discriminator,
            weight_decay=self.cfg.train.weight_decay_discriminator,
        )

        if self.cfg.model.archi.shared_critic:
            if self.cfg.train.lr_critic != self.cfg.train.lr_aux_critic:
                raise ValueError("Shared critic requires equal critic learning rates")
            return

        # prepare parameter list
        self._aux_critic_map_paramlist = tuple(x for x in self._model._aux_critic.parameters())
        self._aux_target_critic_map_paramlist = tuple(x for x in self._model._target_aux_critic.parameters())

        self.aux_critic_optimizer = make_adam(
            self._model._aux_critic.parameters(),
            self.device,
            lr=self.cfg.train.lr_aux_critic,
            weight_decay=self.cfg.train.weight_decay,
        )


    @property
    def optimizers(self):
        optimizers = {
            "actor_optimizer": self.actor_optimizer,
            "backward_optimizer": self.backward_optimizer,
            "forward_optimizer": self.forward_optimizer,
            "critic_optimizer": self.critic_optimizer,
            "discriminator_optimizer": self.discriminator_optimizer,
        }
        if not self.cfg.model.archi.shared_critic:
            optimizers["aux_critic_optimizer"] = self.aux_critic_optimizer
        return optimizers

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        return self._model.act(obs, z, mean)

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs, train_action, train_next_obs = (
            tree_map(lambda x: x.to(self.device), train_batch["observation"]),
            train_batch["action"].to(self.device),
            tree_map(lambda x: x.to(self.device), train_batch["next"]["observation"]),
        )
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)
        expert_obs, expert_next_obs = (
            tree_map(lambda x: x.to(self.device), expert_batch["observation"]),
            tree_map(lambda x: x.to(self.device), expert_batch["next"]["observation"]),
        )

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs, train_next_obs = (
                self._model._obs_normalizer(train_obs),
                self._model._obs_normalizer(train_next_obs),
            )
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device)

        # train the discriminator
        grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=grad_penalty,
        )

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        # The actor is unchanged until update_actor. Reuse its distribution,
        # but draw independent actions at the original positions in the update.
        with torch.no_grad(), self._autocast_context():
            target_dist = self._model._actor(train_next_obs, train_z, self._model.cfg.actor_std)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                goal=train_next_obs,
                z=train_z,
                target_dist=target_dist,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        if not self.cfg.model.archi.shared_critic:
            metrics.update(
                self.update_critic(
                    obs=train_obs,
                    action=train_action,
                    discount=discount,
                    next_obs=train_next_obs,
                    z=train_z,
                    target_dist=target_dist,
                )
            )
        # compute scalar auxiliary reward as a weighted sum of the auxiliary rewards
        aux_reward = torch.zeros(
            (self.cfg.train.batch_size, 1),
            device=self.device,
            dtype=torch.float32,
        )
        for aux_reward_name in self.cfg.aux_rewards:
            # let's log even this information
            metrics[f"aux_rew/{aux_reward_name}"] = train_batch["aux_rewards"][aux_reward_name].mean()
            aux_reward += self.cfg.aux_rewards_scaling[aux_reward_name] * train_batch["aux_rewards"][aux_reward_name].to(self.device)

        aux_reward = self._model._aux_reward_normalizer(aux_reward)

        update_aux = self.update_shared_critic if self.cfg.model.archi.shared_critic else self.update_aux_critic
        metrics.update(
            update_aux(
                obs=train_obs,
                action=train_action,
                discount=discount,
                aux_reward=aux_reward,
                target_dist=target_dist,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        metrics.update(
            self._run_actor_update(
                obs=train_obs,
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._critic_map_paramlist,
                self._target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )
            if not self.cfg.model.archi.shared_critic:
                _soft_update_params(
                    self._aux_critic_map_paramlist,
                    self._aux_target_critic_map_paramlist,
                    self.cfg.train.critic_target_tau,
                )

        return metrics

    def update_fb(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
        goal: torch.Tensor,
        z: torch.Tensor,
        q_loss_coef: float | None,
        clip_grad_norm: float | None,
        target_dist=None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            # dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
            # next_action = dist.sample(clip=self.cfg.train.stddev_clip)
            next_action = self.sample_action_from_norm_obs(next_obs, z, target_dist)
            with self._autocast_context():
                target_Fs = self._model._target_forward_map(next_obs, z, next_action)  # num_parallel x batch x z_dim
                target_B = self._model._target_backward_map(goal)  # batch x z_dim
            target_Fs, target_B = target_Fs.float(), target_B.float()
            target_Ms = torch.matmul(target_Fs, target_B.T)  # num_parallel x batch x batch
            _, _, target_M = self.get_targets_uncertainty(target_Ms, self.cfg.train.fb_pessimism_penalty)  # batch x batch

        # compute FB loss
        with self._autocast_context():
            Fs = self._model._forward_map(obs, z, action)
            B = self._model._backward_map(goal)
        Fs, B = Fs.float(), B.float()
        Ms = torch.matmul(Fs, B.T)  # num_parallel x batch x batch

        diff = Ms - discount * target_M  # num_parallel x batch x batch
        fb_offdiag = 0.5 * (diff * self.off_diag).pow(2).sum() / self.off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=1, dim2=2).mean() * Ms.shape[0]
        fb_loss = fb_offdiag + fb_diag

        # compute orthonormality loss for backward embedding
        Cov = torch.matmul(B, B.T)
        orth_loss_diag = -Cov.diag().mean()
        orth_loss_offdiag = 0.5 * (Cov * self.off_diag).pow(2).sum() / self.off_diag_sum
        orth_loss = orth_loss_offdiag + orth_loss_diag
        fb_loss += self.cfg.train.ortho_coef * orth_loss

        q_loss = torch.zeros(1, device=z.device, dtype=z.dtype)
        if q_loss_coef is not None:
            with torch.no_grad():
                next_Qs = (target_Fs * z).sum(dim=-1)  # num_parallel x batch
                _, _, next_Q = self.get_targets_uncertainty(next_Qs, self.cfg.train.fb_pessimism_penalty)  # batch
                cov = torch.matmul(B.T, B) / B.shape[0]  # z_dim x z_dim
                # inv_cov = torch.inverse(cov)  # z_dim x z_dim
                B_inv_conv = torch.linalg.solve(cov, B, left=False)
                implicit_reward = (B_inv_conv * z).sum(dim=-1)  # batch
                target_Q = implicit_reward.detach() + discount.squeeze() * next_Q  # batch
                expanded_targets = target_Q.expand(Fs.shape[0], -1)
            Qs = (Fs * z).sum(dim=-1)  # num_parallel x batch
            q_loss = 0.5 * Fs.shape[0] * F.mse_loss(Qs, expanded_targets)
            fb_loss += q_loss_coef * q_loss

        # optimize FB
        self.forward_optimizer.zero_grad(set_to_none=True)
        self.backward_optimizer.zero_grad(set_to_none=True)
        fb_loss.backward()
        self._sync_gradients((*self._model._forward_map.parameters(), *self._model._backward_map.parameters()))
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._forward_map.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(self._model._backward_map.parameters(), clip_grad_norm)
        self.forward_optimizer.step()
        self.backward_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_M": target_M.mean(),
                "M1": Ms[0].mean(),
                "F1": Fs[0].mean(),
                "B": B.mean(),
                "B_norm": torch.norm(B, dim=-1).mean(),
                "z_norm": torch.norm(z, dim=-1).mean(),
                "fb_loss": fb_loss,
                "fb_diag": fb_diag,
                "fb_offdiag": fb_offdiag,
                "orth_loss": orth_loss,
                "orth_loss_diag": orth_loss_diag,
                "orth_loss_offdiag": orth_loss_offdiag,
                "q_loss": q_loss,
            }
        return output_metrics

    def update_discriminator(
        self,
        expert_obs: torch.Tensor | dict[str, torch.Tensor],
        expert_z: torch.Tensor,
        train_obs: torch.Tensor | dict[str, torch.Tensor],
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        with self._autocast_context():
            expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
            unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)
        expert_logits, unlabeled_logits = expert_logits.float(), unlabeled_logits.float()
        # these are equivalent to binary cross entropy
        expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
        unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
        loss = torch.mean(expert_loss + unlabeled_loss)

        if grad_penalty is not None:
            wgan_gp = gradient_penalty_wgan(
                self._model._discriminator, expert_obs, expert_z, train_obs, train_z,
                autocast_context=self._autocast_context(),
            )
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
        target_dist=None,
    ) -> Dict[str, torch.Tensor]:
        num_parallel = self.cfg.model.archi.critic.num_parallel
        # compute target critic
        with torch.no_grad():
            reward = self._discriminator_reward(obs, z)
            next_action = self.sample_action_from_norm_obs(next_obs, z, target_dist)
            with self._autocast_context():
                next_Qs = self._model._target_critic(next_obs, z, next_action)  # num_parallel x batch x 1
            next_Qs = next_Qs.float()
            Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.critic_pessimism_penalty)
            target_Q = reward + discount * next_V
            expanded_targets = target_Q.expand(num_parallel, -1, -1)

        # compute critic loss
        with self._autocast_context():
            Qs = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
        Qs = Qs.float()
        critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._sync_gradients(self._model._critic.parameters())
        self.critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_Q": target_Q.mean().detach(),
                "Q1": Qs.mean().detach(),
                "mean_next_Q": Q_mean.mean().detach(),
                "unc_Q": Q_unc.mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_disc_reward": reward.mean().detach(),
            }
        return output_metrics

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
        # compute target critic
        with torch.no_grad():
            next_action = self.sample_action_from_norm_obs(next_obs, z, target_dist)
            with self._autocast_context():
                next_Qs = self._model._target_aux_critic(next_obs, z, next_action)  # num_parallel x batch x 1
            next_Qs = next_Qs.float()
            # TODO AL: should we have aux_critic parameters here?
            Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.aux_critic_pessimism_penalty)
            target_Q = aux_reward + discount * next_V
            expanded_targets = target_Q.expand(num_parallel, -1, -1)

        # compute critic loss
        with self._autocast_context():
            Qs = self._model._aux_critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
        Qs = Qs.float()
        aux_critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.aux_critic_optimizer.zero_grad(set_to_none=True)
        aux_critic_loss.backward()
        self._sync_gradients(self._model._aux_critic.parameters())
        self.aux_critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_auxQ": target_Q.mean().detach(),
                "auxQ1": Qs.mean().detach(),
                "mean_next_auxQ": Q_mean.mean().detach(),
                "unc_auxQ": Q_unc.mean().detach(),
                "aux_critic_loss": aux_critic_loss.mean().detach(),
                "mean_aux_reward": aux_reward.mean().detach(),
            }
        return output_metrics

    def update_shared_critic(self, obs, action, discount, aux_reward, next_obs, z, target_dist=None):
        n = self.cfg.model.archi.critic.num_parallel
        with torch.no_grad():
            reward = self._discriminator_reward(obs, z)
            with self._autocast_context():
                dist = target_dist if target_dist is not None else self._model._actor(next_obs, z, self._model.cfg.actor_std)
            # Preserve independent target-action draws and two target forwards.
            next_action = self.sample_action_from_norm_obs(next_obs, z, dist)
            with self._autocast_context():
                next_disc = self._model._target_critic(next_obs, z, next_action)[0]
            next_disc = next_disc.float()
            next_aux_action = self.sample_action_from_norm_obs(next_obs, z, dist)
            with self._autocast_context():
                next_aux = self._model._target_critic(next_obs, z, next_aux_action)[1]
            next_aux = next_aux.float()
            mean_disc, unc_disc, value_disc = self.get_targets_uncertainty(next_disc, self.cfg.train.critic_pessimism_penalty)
            mean_aux, unc_aux, value_aux = self.get_targets_uncertainty(next_aux, self.cfg.train.aux_critic_pessimism_penalty)
            target_disc = reward + discount * value_disc
            target_aux = aux_reward + discount * value_aux
        with self._autocast_context():
            q_disc, q_aux = self._model._critic(obs, z, action)
        q_disc, q_aux = q_disc.float(), q_aux.float()
        disc_loss = 0.5 * n * F.mse_loss(q_disc, target_disc.expand(n, -1, -1))
        aux_loss = 0.5 * n * F.mse_loss(q_aux, target_aux.expand(n, -1, -1))
        loss = disc_loss + aux_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._sync_gradients(self._model._critic.parameters())
        self.critic_optimizer.step()
        return {
            k: v.detach()
            for k, v in dict(
                target_Q=target_disc.mean(),
                Q1=q_disc.mean(),
                mean_next_Q=mean_disc.mean(),
                unc_Q=unc_disc.mean(),
                critic_loss=disc_loss,
                mean_disc_reward=reward.mean(),
                target_auxQ=target_aux.mean(),
                auxQ1=q_aux.mean(),
                mean_next_auxQ=mean_aux.mean(),
                unc_auxQ=unc_aux.mean(),
                aux_critic_loss=aux_loss,
                mean_aux_reward=aux_reward.mean(),
            ).items()
        }

    def update_actor(
        self,
        obs: torch.Tensor | dict[str, torch.Tensor],
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        action = self.sample_action_from_norm_obs(obs, z)

        # compute discriminator reward loss
        with self._autocast_context():
            if self.cfg.model.archi.shared_critic:
                Qs_discriminator, Qs_aux = self._model._critic(obs, z, action)
            else:
                Qs_discriminator = self._model._critic(obs, z, action)
                Qs_aux = self._model._aux_critic(obs, z, action)
        Qs_discriminator, Qs_aux = Qs_discriminator.float(), Qs_aux.float()
        _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.cfg.train.actor_pessimism_penalty)  # batch

        # compute auxiliary reward loss
        _, _, Q_aux = self.get_targets_uncertainty(Qs_aux, self.cfg.train.actor_pessimism_penalty)  # batch

        # compute fb reward loss
        with self._autocast_context():
            Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
        Fs = Fs.float()
        Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

        weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
        actor_loss = (
            -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight
            - Q_aux.mean() * self.cfg.train.reg_coeff_aux * weight
            - Q_fb.mean()
        )

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._sync_gradients(self._model._actor.parameters())
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_aux": Q_aux.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
            }
        return output_metrics

    def get_targets_uncertainty(
        self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dim = 0
        preds_mean = preds.mean(dim=dim)
        if preds.shape[dim] == 2:
            preds_unc = (preds[0] - preds[1]).abs()
            return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc
        preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
        preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
        preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
        num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
        preds_unc = (
            preds_diffs.sum(
                dim=(dim, dim + 1),
            )
            / num_parallel_scaling
        )
        return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc

    def sample_action_from_norm_obs(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, target_dist=None) -> torch.Tensor:
        with self._autocast_context():
            dist = target_dist if target_dist is not None else self._model._actor(obs, z, self._model.cfg.actor_std)
            action = dist.sample(clip=self.cfg.train.stddev_clip)
        return action.float()

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor, expert_encodings: torch.Tensor, *args, **kwargs):
        z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
        p_goal = self.cfg.train.train_goal_ratio
        p_expert_asm = self.cfg.train.expert_asm_ratio
        prob = torch.tensor(
            [p_goal, p_expert_asm, 1 - p_goal - p_expert_asm],
            dtype=torch.float32,
            device=self.device,
        )
        mix_idxs = torch.multinomial(prob, num_samples=self.cfg.train.batch_size, replacement=True).reshape(-1, 1)

        # zs obtained by encoding train goals
        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        train_goal = tree_map(lambda x: x[perm], train_goal)
        with self._autocast_context():
            goals = self._model._backward_map(train_goal)
        goals = self._model.project_z(goals.float())
        z = torch.where(mix_idxs == 0, goals, z)

        # zs obtained by encoding expert trajectories
        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

        return z

    @torch.no_grad()
    def encode_expert(
        self,
        next_obs: torch.Tensor | dict[str, torch.Tensor],
    ):
        # encode expert trajectories through B
        with self._autocast_context():
            B_expert = self._model._backward_map(next_obs).detach()  # batch x d
        B_expert = B_expert.float()
        B_expert = B_expert.view(
            self.cfg.train.batch_size // self.cfg.model.seq_length,
            self.cfg.model.seq_length,
            B_expert.shape[-1],
        )  # N x L x d
        z_expert = B_expert.mean(dim=1)  # N x d
        z_expert = self._model.project_z(z_expert)
        z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)  # batch x d
        return z_expert

    def _sample_tracking_z(self, replay_buffer, batch_dim, traj_length):
        batch = replay_buffer["expert"].sample(batch_dim * traj_length, seq_length=traj_length)  # N*T x obs_dim
        z = self._model.backward_map(batch["next"]["observation"])  # NT x z_dim
        z = z.view(batch_dim, traj_length, z.shape[-1])  # N x T x z_dim
        for step in range(traj_length):
            end_idx = min(step + self.cfg.model.seq_length, traj_length)
            z[:, step] = z[:, step:end_idx].mean(dim=1)
        return self._model.project_z(z)  # N x T x z_dim

    def maybe_update_rollout_context(self, z: torch.Tensor | None, step_count: torch.Tensor, replay_buffer: None = None) -> torch.Tensor:
        # get mask for environmets where we need to change z
        if z is not None:
            mask_reset_z = step_count % self.cfg.train.update_z_every_step == 0
            if self.cfg.train.use_mix_rollout and not self.z_buffer.empty():
                new_z = self.z_buffer.sample(z.shape[0], device=self._model.device)
            else:
                new_z = self._model.sample_z(z.shape[0], device=self._model.device)
            z = torch.where(mask_reset_z, new_z, z.to(self._model.device))
            if self.cfg.train.rollout_expert_trajectories:
                idxs = step_count % self.cfg.train.rollout_expert_trajectories_length
                if torch.any(idxs == 0):
                    n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage * step_count.shape[0])
                    self.env_idx_with_expert_rollout = torch.randint(0, step_count.shape[0], size=(n_elem,), device=self._model.device)
                    self.tracking_z = self._sample_tracking_z(
                        replay_buffer, n_elem, self.cfg.train.rollout_expert_trajectories_length
                    )  # N x T x z_dim
                mod_time = idxs[self.env_idx_with_expert_rollout].ravel()
                z[self.env_idx_with_expert_rollout] = self.tracking_z[
                    torch.arange(len(self.env_idx_with_expert_rollout), device=self._model.device), mod_time
                ]
        else:
            z = self._model.sample_z(step_count.shape[0], device=self._model.device)
            if self.cfg.train.rollout_expert_trajectories:
                n_elem = int(self.cfg.train.rollout_expert_trajectories_percentage * step_count.shape[0])
                self.env_idx_with_expert_rollout = torch.randint(0, step_count.shape[0], size=(n_elem,), device=self._model.device)
                self.tracking_z = self._sample_tracking_z(
                    replay_buffer, n_elem, self.cfg.train.rollout_expert_trajectories_length
                )  # N x T x z_dim
                z[self.env_idx_with_expert_rollout] = self.tracking_z[:, 0]
        return z

    def save(self, output_folder: str) -> None:
        save_agent(self, output_folder)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        return load_agent(cls.config_class, path, device)

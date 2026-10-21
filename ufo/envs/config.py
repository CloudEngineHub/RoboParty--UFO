"""Resolve the UFO task and robot config, then build the mjlab physics scene.

MJLabCore owns task resets and observations; mjlab owns physics and episode counters.
"""

import os
import typing as tp
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pydantic

import ufo

if TYPE_CHECKING:
    from .vector_env import MJLabVectorEnv
import torch
from omegaconf import OmegaConf

from ufo.agents.common.base import BaseConfig

if getattr(ufo, "__file__", None) is not None:
    UFO_DIR = os.path.dirname(ufo.__file__)
else:
    UFO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TASK_DEFAULTS = Path(UFO_DIR).parent / "configs/env.yaml"

G1_MJLAB_MJCF_PATH = "ufo/data/robots/g1/g1_29dof.xml"

def _obs_joint_pos(env):
    return env.scene["robot"].data.joint_pos


def _zero_reward(env):
    return torch.zeros(env.num_envs, device=env.device)


def _to_list(value) -> list:
    if value is None:
        return []
    return list(OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value)


def _to_float_dict(value) -> dict[str, float]:
    value = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return {str(k): float(v) for k, v in value.items()}


def _match_joint_value(joint_name: str, value_by_substring: dict[str, float], default: float = 0.0) -> float:
    for key, value in value_by_substring.items():
        if key in joint_name:
            return float(value)
    return float(default)


def _joint_limits_from_robot_training(robot_training: dict[str, Any], dof_names: tuple[str, ...]) -> tuple[list[float], list[float]]:
    joint_ranges = dict(robot_training.get("robot", {}).get("joint_ranges") or {})
    lower, upper = [], []
    for joint_name in dof_names:
        value = joint_ranges.get(joint_name)
        if value is None:
            lower.append(-3.14159)
            upper.append(3.14159)
        else:
            lower.append(float(value[0]))
            upper.append(float(value[1]))
    return lower, upper


def _patch_ufo_robot_config(config, robot_training: dict[str, Any] | None) -> None:
    if not robot_training:
        return
    robot_info = dict(robot_training["robot"])
    dof_names = [str(name) for name in robot_info["control_joint_names"]]
    body_names = [str(name) for name in robot_info["body_names"]]
    feet = [str(name) for name in robot_info.get("feet") or []]
    lower, upper = _joint_limits_from_robot_training(robot_training, tuple(dof_names))

    config.robot.dof_names = dof_names
    config.robot.actions_dim = len(dof_names)
    config.robot.body_names = body_names
    config.robot.num_bodies = len(body_names)
    config.robot.key_bodies = list(robot_info.get("key_bodies") or [])
    config.robot.contact_bodies = list(robot_training.get("contact_bodies") or feet)
    config.robot.torso_name = str(robot_training.get("torso_name") or robot_info.get("base_body"))
    config.robot.penalize_contacts_on = list(robot_training.get("undesired_contact_bodies") or [])
    config.robot.left_ankle_dof_names = list(robot_training.get("left_ankle_dof_names") or [])
    config.robot.right_ankle_dof_names = list(robot_training.get("right_ankle_dof_names") or [])
    config.robot.dof_pos_lower_limit_list = lower
    config.robot.dof_pos_upper_limit_list = upper
    config.robot.dof_vel_limit_list = list(robot_training["velocity_limits"])
    config.robot.dof_effort_limit_list = list(robot_training["effort_limits"])
    config.robot.dof_effort_limit_scale = float(robot_training.get("effort_limit_scale", 1.0))
    actuator = robot_training.get("actuator", {})
    if actuator.get("model") == "roboparty":
        config.robot.rated_effort = [float(actuator["joints"][name]["rated_effort"]) for name in dof_names]
        config.robot.rated_velocity = [float(actuator["joints"][name]["rated_velocity"]) for name in dof_names]

    if config.robot.get("init_state") is None:
        config.robot.init_state = OmegaConf.create({})
    if config.robot.get("control") is None:
        config.robot.control = OmegaConf.create({})

    config.robot.init_state.pos = list(robot_training["init_state"]["pos"])
    config.robot.init_state.rot = list(robot_training["init_state"]["rot"])
    config.robot.init_state.lin_vel = list(robot_training["init_state"]["lin_vel"])
    config.robot.init_state.ang_vel = list(robot_training["init_state"]["ang_vel"])
    config.robot.init_state.default_joint_angles = dict(robot_training["default_joint_angles"])
    config.robot.control.stiffness = dict(robot_training["stiffness"])
    config.robot.control.damping = dict(robot_training["damping"])
    config.robot.control.action_scale = float(robot_training["action_scale"])
    config.robot.control.action_rescale = bool(robot_training.get("action_rescale", True))
    config.robot.control.action_clip_value = float(robot_training["action_clip_value"])
    config.robot.control.normalize_action_to = float(robot_training["normalize_action_to"])

    xml_path = Path(robot_info["xml_path"]).expanduser().resolve()
    if config.robot.get("asset") is None:
        config.robot.asset = OmegaConf.create({})
    config.robot.asset.asset_root = str(xml_path.parent)
    config.robot.asset.assetFileName = xml_path.name
    config.robot.asset.xml_file = str(xml_path)

    if config.robot.get("motion") is None:
        config.robot.motion = OmegaConf.create({})
    if config.robot.motion.get("asset") is None:
        config.robot.motion.asset = OmegaConf.create({})
    config.robot.motion.asset.assetRoot = str(xml_path.parent)
    config.robot.motion.asset.assetFileName = xml_path.name

    extend_config = []
    for item in _to_list(config.robot.motion.get("extend_config", [])):
        parent_name = str(item.get("parent_name", ""))
        if parent_name in body_names:
            extend_config.append(dict(item))
    config.robot.motion.extend_config = extend_config



def _default_joint_pos(config) -> torch.Tensor:
    values = [float(config.robot.init_state.default_joint_angles[name]) for name in config.robot.dof_names]
    return torch.tensor(values, dtype=torch.float32)


def _action_target_scale_values(config) -> list[float]:
    dof_names = tuple(_to_list(config.robot.dof_names))
    stiffness = _to_float_dict(config.robot.control.stiffness)
    # UFO action_rescale uses the configured effort limits.  The Isaac
    # path does not apply dof_effort_limit_scale to those limits, so MJLab must
    # not do it either.
    effort_limits = [float(x) for x in _to_list(config.robot.dof_effort_limit_list)]
    scales = []
    for i, joint_name in enumerate(dof_names):
        kp = _match_joint_value(joint_name, stiffness)
        scale = float(config.robot.control.action_scale)
        if bool(config.robot.control.action_rescale):
            if kp <= 0.0:
                raise ValueError(f"Cannot action_rescale joint {joint_name}: stiffness={kp}")
            scale *= effort_limits[i] / kp
        scales.append(scale)
    return scales


def _action_target_scale(config) -> torch.Tensor:
    return torch.tensor(_action_target_scale_values(config), dtype=torch.float32)


def _small_random_quaternions(n: int, max_angle: float, device: str) -> torch.Tensor:
    axis = torch.randn((n, 3), device=device)
    axis = axis / torch.clamp(torch.norm(axis, dim=1, keepdim=True), min=1.0e-6)
    angles = max_angle * torch.rand((n, 1), device=device)
    sin_half_angle = torch.sin(angles / 2)
    cos_half_angle = torch.cos(angles / 2)
    return torch.cat([sin_half_angle * axis, cos_half_angle], dim=1)


def _push_by_replacing_velocity(env, env_ids, velocity_range) -> None:
    """The RP1 source overwrites velocity; mjlab's stock event adds a delta."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    velocity = torch.zeros((len(env_ids), 6), device=env.device)
    for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        if key in velocity_range:
            velocity[:, index].uniform_(*velocity_range[key])
    env.scene["robot"].write_root_com_velocity_to_sim(velocity, env_ids=env_ids)


def _randomize_mass_and_inertia(env, env_ids, ranges, asset_cfg, operation) -> None:
    """Match Isaac's recompute_inertia=True without changing the G1 mass event."""
    from mjlab.envs.mdp import dr

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.int)
    body_ids = env.scene[asset_cfg.name].indexing.body_ids[asset_cfg.body_ids]
    rows, columns = torch.meshgrid(env_ids.to(dtype=body_ids.dtype), body_ids, indexing="ij")
    before = env.sim.model.body_mass[rows, columns].clone()
    dr.body_mass(env, env_ids, ranges=ranges, asset_cfg=asset_cfg, operation=operation)
    after = env.sim.model.body_mass[rows, columns]
    if torch.any(before <= 0.0) or torch.any(after <= 0.0):
        raise ValueError("Mass randomization must preserve positive body masses")
    env.sim.model.body_inertia[rows, columns] *= (after / before).unsqueeze(-1)


def make_mjlab_ufo_env_cfg(
    config,
    *,
    num_envs: int,
    seed: int | None,
    mjcf_path: str | None,
    auto_reset: bool,
    robot_training: dict[str, Any] | None = None,
):
    """Create an MJLab ManagerBasedRlEnvCfg with UFO robot metadata."""
    import mujoco
    from mjlab.actuator import DcMotorActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs import mdp as mjlab_mdp
    from mjlab.envs.mdp import dr as mjlab_dr
    from mjlab.envs.mdp import terminations as mjlab_terminations
    from mjlab.envs.mdp.actions import JointPositionActionCfg
    from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.scene import SceneCfg
    from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
    from mjlab.sim import MujocoCfg, SimulationCfg
    from mjlab.terrains import TerrainEntityCfg

    dof_names = tuple(_to_list(config.robot.dof_names))
    body_names = tuple(_to_list(config.robot.body_names))
    xml_path = Path(mjcf_path) if mjcf_path is not None else Path(G1_MJLAB_MJCF_PATH)
    xml_path = xml_path if xml_path.is_absolute() else Path(UFO_DIR).parent / xml_path
    if not xml_path.exists():
        raise FileNotFoundError(f"MJCF asset not found: {xml_path}")
    if "actuatorfrcrange" in xml_path.read_text():
        raise ValueError(f"MJLab robot XML must not contain actuatorfrcrange: {xml_path}")

    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(xml_path))
        # The UFO Isaac path uses implicit position PD targets. XML motor
        # actuators are removed so MJLab adds equivalent position actuators.
        for actuator in list(spec.actuators):
            spec.delete(actuator)
        if config.rewards.set_reward == "roboparty":
            # Floor friction is 1 in the source's multiply rule; preserve robot samples below 1.
            for geom in spec.geoms:
                if geom.contype or geom.conaffinity:
                    geom.priority = 1
        return spec

    stiffness = _to_float_dict(config.robot.control.stiffness)
    damping = _to_float_dict(config.robot.control.damping)
    effort_scale = float(getattr(config.robot, "dof_effort_limit_scale", 1.0))
    bfm_effort_limits = [float(x) for x in _to_list(config.robot.dof_effort_limit_list)]
    from .actuators import _actuator_params_from_training

    actuator_source, actuator_params = _actuator_params_from_training(dof_names, robot_training)
    effort_limits = actuator_params["effort_limit"]
    velocity_limits = actuator_params["velocity_limit"]
    armature = actuator_params["armature"]
    friction = actuator_params["friction"]
    actuator_settings = (robot_training or {}).get("actuator", {})
    actuator_cfg_type = DcMotorActuatorCfg
    actuator_kwargs = {}
    if actuator_settings.get("model") == "roboparty":
        from .actuators import RoboPartyActuatorCfg

        actuator_cfg_type = RoboPartyActuatorCfg
        actuator_kwargs = {
            "delay_min_lag": actuator_settings["min_delay"],
            "delay_max_lag": actuator_settings["max_delay"],
        }

    actuators = []
    for i, joint_name in enumerate(dof_names):
        kp = _match_joint_value(joint_name, stiffness)
        kd = _match_joint_value(joint_name, damping)
        effort_limit = effort_limits[i]
        saturation_effort = effort_limit
        if actuator_settings.get("model") == "roboparty":
            saturation_effort = effort_limit / (1.0 - actuator_params["keep_velocity"][i] / velocity_limits[i])
            actuator_kwargs["viscous_damping"] = actuator_params["viscous_friction"][i]
        actuators.append(
            actuator_cfg_type(
                target_names_expr=(joint_name,),
                stiffness=kp,
                damping=kd,
                effort_limit=effort_limit,
                saturation_effort=saturation_effort,
                velocity_limit=velocity_limits[i],
                armature=armature[i] if i < len(armature) else None,
                frictionloss=friction[i] if i < len(friction) else None,
                **actuator_kwargs,
            )
        )

    action_scale = dict(zip(dof_names, _action_target_scale_values(config)))
    if len(actuators) != len(dof_names):
        raise ValueError(f"Expected one MJLab actuator per UFO dof, got {len(actuators)} for {len(dof_names)} dofs")
    scaled_effort_limits = [float(x) * effort_scale for x in bfm_effort_limits]
    if effort_scale != 1.0 and any(abs(a - b) < 1.0e-6 for a, b in zip(effort_limits, scaled_effort_limits)):
        raise ValueError("MJLab actuator effort limits unexpectedly include dof_effort_limit_scale")
    print(
        "[INFO] MJLab asset: "
        f"xml_path={xml_path}, actuator_source={actuator_source}, "
        f"actuator_count={len(actuators)}, joint_order={list(dof_names)}, "
        f"action_scale={[action_scale[name] for name in dof_names]}, "
        f"kp={[_match_joint_value(name, stiffness) for name in dof_names]}, "
        f"kd={[_match_joint_value(name, damping) for name in dof_names]}, "
        f"effort_limit={effort_limits}, velocity_limit={velocity_limits}, "
        f"armature={armature}, friction={friction}, "
        f"dof_effort_limit_scale={effort_scale} ignored_for_mjlab_actuator_limits",
        flush=True,
    )

    init_rot_xyzw = tuple(float(x) for x in config.robot.init_state.rot)
    init_rot_wxyz = (init_rot_xyzw[3], init_rot_xyzw[0], init_rot_xyzw[1], init_rot_xyzw[2])
    init_state = EntityCfg.InitialStateCfg(
        pos=tuple(float(x) for x in config.robot.init_state.pos),
        rot=init_rot_wxyz,
        lin_vel=tuple(float(x) for x in config.robot.init_state.lin_vel),
        ang_vel=tuple(float(x) for x in config.robot.init_state.ang_vel),
        joint_pos={name: float(config.robot.init_state.default_joint_angles[name]) for name in dof_names},
        joint_vel={".*": 0.0},
    )

    robot_cfg = EntityCfg(
        spec_fn=spec_fn,
        init_state=init_state,
        articulation=EntityArticulationInfoCfg(actuators=tuple(actuators), soft_joint_pos_limit_factor=1.0),
        sort_actuators=True,
    )
    sensors = (
        ContactSensorCfg(
            name="body_contact",
            primary=ContactMatch(mode="body", pattern=body_names, entity="robot"),
            fields=("found", "force"),
            reduce="netforce",
            history_length=int(config.simulator.config.sim.control_decimation),
        ),
    )
    if config.rewards.set_reward == "roboparty":
        from .contact import NormalContactSensorCfg

        sensors += (NormalContactSensorCfg(name="normal_contact", body_names=body_names),)
    observations = {
        "actor": ObservationGroupCfg(
            terms={"joint_pos": ObservationTermCfg(func=_obs_joint_pos)},
            concatenate_terms=True,
            enable_corruption=False,
        )
    }
    actions = {
        "actions": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=dof_names,
            preserve_order=True,
            scale=action_scale,
            use_default_offset=True,
        )
    }
    reward_keys = tuple(config.rewards.reward_scales.keys())
    rewards = {key: RewardTermCfg(func=_zero_reward, weight=0.0) for key in reward_keys}
    terminations = {
        "time_out": TerminationTermCfg(func=mjlab_terminations.time_out, time_out=True),
    }
    domain_rand = config.domain_rand
    events = {}
    mass_randomizer = mjlab_dr.body_mass
    if bool(domain_rand.get("mass_scale_inertia", False)):
        mass_randomizer = requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)(
            _randomize_mass_and_inertia
        )
    if bool(domain_rand.get("push_robots", False)):
        max_push_vel_xy = float(domain_rand.max_push_vel_xy)
        max_push_ang_vel = float(domain_rand.get("max_push_ang_vel", 0.0))
        velocity_range = {
            "x": (-max_push_vel_xy, max_push_vel_xy),
            "y": (-max_push_vel_xy, max_push_vel_xy),
        }
        if max_push_ang_vel > 0.0:
            velocity_range.update(
                {
                    "roll": (-max_push_ang_vel, max_push_ang_vel),
                    "pitch": (-max_push_ang_vel, max_push_ang_vel),
                    "yaw": (-max_push_ang_vel, max_push_ang_vel),
                }
            )
        push_func = mjlab_mdp.push_by_setting_velocity
        if domain_rand.get("push_velocity_range") is not None:
            velocity_range = {name: tuple(float(x) for x in values) for name, values in domain_rand.push_velocity_range.items()}
            push_func = _push_by_replacing_velocity
        events["push_robots"] = EventTermCfg(
            func=push_func,
            mode="interval",
            interval_range_s=tuple(float(x) for x in _to_list(domain_rand.push_interval_s)),
            params={"velocity_range": velocity_range},
        )
    if bool(domain_rand.get("randomize_base_com", False)):
        base_com_range = domain_rand.base_com_range
        events["random_base_com"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(str(config.robot.torso_name),)),
                "operation": "add",
                "ranges": {
                    0: tuple(float(x) for x in _to_list(base_com_range.x)),
                    1: tuple(float(x) for x in _to_list(base_com_range.y)),
                    2: tuple(float(x) for x in _to_list(base_com_range.z)),
                },
            },
        )
    if bool(domain_rand.get("randomize_link_mass", False)):
        mass_bodies = domain_rand.get("link_mass_bodies")
        events["random_link_mass"] = EventTermCfg(
            mode="startup",
            func=mass_randomizer,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=tuple(_to_list(mass_bodies)) if mass_bodies is not None else ".*"),
                "operation": "scale",
                "ranges": tuple(float(x) for x in _to_list(domain_rand.link_mass_range)),
            },
        )
    if bool(domain_rand.get("randomize_base_mass", False)):
        events["random_base_mass"] = EventTermCfg(
            mode="startup",
            func=mass_randomizer,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(str(config.robot.torso_name),)),
                "operation": "add",
                "ranges": tuple(float(x) for x in _to_list(domain_rand.base_mass_range)),
            },
        )
    if bool(domain_rand.get("randomize_pd_gain", False)):
        events["random_pd_gain"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.pd_gains,
            params={
                "asset_cfg": SceneEntityCfg("robot", actuator_names=dof_names),
                "operation": "scale",
                "kp_range": tuple(float(x) for x in _to_list(domain_rand.kp_range)),
                "kd_range": tuple(float(x) for x in _to_list(domain_rand.kd_range)),
            },
        )
    if bool(domain_rand.get("randomize_joint_parameters", False)):
        for name, func, ranges in (
            ("friction", mjlab_dr.joint_friction, domain_rand.joint_friction_range),
            ("armature", mjlab_dr.joint_armature, domain_rand.armature_range),
        ):
            events[f"random_joint_{name}"] = EventTermCfg(
                mode="startup",
                func=func,
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=dof_names), "operation": "scale", "ranges": tuple(ranges)},
            )
    if bool(domain_rand.get("randomize_friction", False)):
        events["random_geom_friction"] = EventTermCfg(
            mode="startup",
            func=mjlab_dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=".*"),
                "operation": "abs",
                "axes": [0],
                "ranges": tuple(float(x) for x in _to_list(domain_rand.friction_range)),
            },
        )

    physics_options = {}
    if config.rewards.set_reward == "roboparty":
        physics_options = {"integrator": "implicitfast", "cone": "pyramidal", "iterations": 100, "ls_iterations": 50, "tolerance": 1e-6}
    return ManagerBasedRlEnvCfg(
        decimation=int(config.simulator.config.sim.control_decimation),
        scene=SceneCfg(
            num_envs=num_envs,
            env_spacing=float(config.env_spacing),
            terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=float(config.env_spacing)),
            entities={"robot": robot_cfg},
            sensors=sensors,
        ),
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
        events=events,
        seed=seed,
        sim=SimulationCfg(
            nconmax=512,
            njmax=4096,
            contact_sensor_maxmatch=256,
            mujoco=MujocoCfg(timestep=1.0 / float(config.simulator.config.sim.fps), **physics_options),
        ),
        episode_length_s=float(config.max_episode_length_s),
        auto_reset=auto_reset,
        scale_rewards_by_dt=False,
    )


def resolve_task_config(env, num_envs):
    if not env.robot_training:
        raise ValueError("Native task configuration requires robot_training")
    defaults = OmegaConf.load(TASK_DEFAULTS)
    OmegaConf.set_struct(defaults, True)
    task = OmegaConf.merge(defaults, env.task_overrides or {})
    OmegaConf.set_struct(task, False)
    task.num_envs = num_envs
    robot = OmegaConf.create(
        {
            "control": {"action_rescale": True, "normalize_action": True, "normalize_action_from": 1.0},
            "asset": {},
            "motion": {"asset": {}, "extend_config": env.robot_training.get("motion_extensions", [])},
        }
    )
    wrapper = OmegaConf.create({"robot": robot})
    _patch_ufo_robot_config(wrapper, env.robot_training)
    task.robot = wrapper.robot
    task.robot.motion.motion_file = env.motion_data_path
    if env.data_mix_weights is not None:
        task.robot.motion.motion_file_weights = env.data_mix_weights
    task.obs.root_height_obs = env.root_height_obs
    count = len(task.robot.dof_names)
    # Preserve the legacy config dimension convention; actual root-height
    # observation sizing is computed from the environment tensors.
    dims = {
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "dof_pos": count,
        "dof_vel": count,
        "actions": count,
        "max_local_self": 15 * (len(task.robot.body_names) + len(task.robot.motion.extend_config)) - 3,
    }
    for name, history in task.obs.obs_auxiliary.items():
        dims[name] = sum(dims[key] * length for key, length in history.items())
    task.obs.obs_dims = dims
    if env.disable_obs_noise:
        for name in task.obs.noise_scales:
            task.obs.noise_scales[name] = 0.0
    if env.disable_domain_randomization:
        for name in task.domain_rand:
            if name.startswith("randomize_") or name == "push_robots":
                task.domain_rand[name] = False
    unsupported = [name for name in ("randomize_torque_rfi", "randomize_rfi_lim", "randomize_ctrl_delay")
                   if task.domain_rand.get(name, False)]
    if unsupported:
        raise ValueError(f"Unsupported mjlab domain randomization: {', '.join(unsupported)}. "
                         "RP1 motor delay is configured separately in training.actuator.")
    if env.max_episode_length_s is not None:
        task.max_episode_length_s = env.max_episode_length_s
    if any(task.termination.values()):
        raise ValueError("The UFO motion task currently supports timeout termination only")
    if task.robot.control.normalize_action_to != task.robot.control.action_clip_value:
        raise ValueError("normalize_action_to must equal action_clip_value")
    return task


class MJLabEnvConfig(BaseConfig):
    name: tp.Literal["mjlab"] = "mjlab"

    device: str = "cuda:0"
    motion_data_path: str | list[str]
    data_mix_weights: list[float] | None = None
    mjcf_path: str | None = None
    robot_config_path: str | None = None
    robot_training: dict[str, Any] | None = None
    task_overrides: dict[str, Any] = pydantic.Field(default_factory=dict)
    max_episode_length_s: float | None = None
    disable_obs_noise: bool = False
    disable_domain_randomization: bool = False
    include_last_action: bool = True
    include_history_actor: bool = False
    root_height_obs: bool = False
    auto_reset: bool = False
    seed: int | None = None


    def build(self, num_envs: int = 1) -> tp.Tuple["MJLabVectorEnv", tp.Any]:
        if self.auto_reset:
            raise ValueError("UFO owns task reset; auto_reset must be False")
        assert num_envs >= 1
        from mjlab.envs import ManagerBasedRlEnv

        from .core import MJLabCore
        from .vector_env import MJLabVectorEnv

        task_config = resolve_task_config(self, num_envs)
        mjlab_cfg = make_mjlab_ufo_env_cfg(
            task_config,
            num_envs=num_envs,
            seed=self.seed,
            mjcf_path=self.mjcf_path,
            auto_reset=self.auto_reset,
            robot_training=self.robot_training,
        )
        mjlab_env = ManagerBasedRlEnv(mjlab_cfg, device=self.device)
        core = MJLabCore(task_config, mjlab_env)
        env = MJLabVectorEnv(
            core,
            include_last_action=self.include_last_action,
            include_history_actor=self.include_history_actor,
        )
        return env, {
            "task_config": OmegaConf.to_container(task_config, resolve=True),
            "mjlab_env_cfg": mjlab_cfg,
        }

"""Per-body normal contact forces for the RP1 auxiliary rewards."""

from dataclasses import dataclass

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mjlab.sensor.sensor import Sensor, SensorCfg


@wp.kernel
def _sum_normal_forces(
    nacon: wp.array[int],
    world: wp.array[int],
    geom: wp.array[wp.vec2i],
    frame: wp.array[wp.mat33],
    force: wp.array[wp.spatial_vector],
    geom_body: wp.array[int],
    body_index: wp.array[int],
    out: wp.array2d[wp.vec3],
):
    contact = wp.tid()
    if contact < nacon[0]:
        pair = geom[contact]
        normal = frame[contact][0] * force[contact][0]
        for side in range(2):
            if pair[side] >= 0:
                body = body_index[geom_body[pair[side]]]
                if body >= 0:
                    sign = float(2 * side - 1)
                    wp.atomic_add(out, world[contact], body, sign * normal)


@dataclass
class NormalContactSensorCfg(SensorCfg):
    body_names: tuple[str, ...]

    def build(self):
        return NormalContactSensor(self)


class NormalContactSensor(Sensor[torch.Tensor]):
    """Sum all contacts, then keep the last three physics samples."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def edit_spec(self, scene_spec, entities):
        pass

    def initialize(self, mj_model, model, data, device):
        self.model, self.physics = model, data
        indices = np.full(mj_model.nbody, -1, dtype=np.int32)
        for index, name in enumerate(self.cfg.body_names):
            body = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{name}")
            if body < 0:
                raise ValueError(f"Normal contact sensor cannot find robot/{name}")
            indices[body] = index
        self.body_index = wp.array(indices, dtype=int, device=device)
        self.ids = wp.array(np.arange(data.naconmax, dtype=np.int32), dtype=int, device=device)
        self.forces = wp.zeros(data.naconmax, dtype=wp.spatial_vector, device=device)
        self.net = wp.zeros((data.nworld, len(indices[indices >= 0])), dtype=wp.vec3, device=device)
        self.net_tensor = wp.to_torch(self.net)
        self.history = torch.zeros((data.nworld, 3, len(self.cfg.body_names), 3), device=device)

    def update(self, dt):
        super().update(dt)
        self.net.zero_()
        mjwarp.contact_force(self.model, self.physics, self.ids, False, self.forces)
        contact = self.physics.contact
        wp.launch(
            _sum_normal_forces,
            dim=self.physics.naconmax,
            inputs=[self.physics.nacon, contact.worldid, contact.geom, contact.frame,
                    self.forces, self.model.geom_bodyid, self.body_index],
            outputs=[self.net],
            device=self.net.device,
        )
        self.history[:, 1:] = self.history[:, :-1].clone()
        self.history[:, 0] = self.net_tensor

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self.history[slice(None) if env_ids is None else env_ids] = 0.0

    def _compute_data(self):
        return self.history

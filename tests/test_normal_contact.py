import numpy as np
import torch
import warp as wp

from ufo.envs.contact import NormalContactSensor, NormalContactSensorCfg, _sum_normal_forces


def test_normal_forces_sum_all_contacts_and_preserve_worlds():
    # Two different normals on one body, including a robot self-contact.
    nacon = wp.array([3], dtype=int, device="cpu")
    world = wp.array([0, 0, 1, 0], dtype=int, device="cpu")
    geom = wp.array([[0, 1], [1, 2], [0, 2], [0, 1]], dtype=wp.vec2i, device="cpu")
    frames = np.tile(np.eye(3, dtype=np.float32), (4, 1, 1))
    frames[1, 0] = [0, 0, 1]
    force = np.zeros((4, 6), np.float32)
    force[:, 0] = [10, 20, 30, 999]
    force[:, 1:] = 1000  # Tangential forces/torques must not enter the normal sum.
    out = wp.zeros((2, 2), dtype=wp.vec3, device="cpu")
    wp.launch(_sum_normal_forces, dim=4, inputs=[
        nacon, world, geom, wp.array(frames, dtype=wp.mat33, device="cpu"),
        wp.array(force, dtype=wp.spatial_vector, device="cpu"),
        wp.array([0, 1, 2], dtype=int, device="cpu"),
        wp.array([-1, 0, 1], dtype=int, device="cpu"),
    ], outputs=[out], device="cpu")
    np.testing.assert_array_equal(out.numpy(), [[[10, 0, -20], [0, 0, 20]], [[0, 0, 0], [30, 0, 0]]])


def test_contact_partial_reset_only_clears_selected_history():
    sensor = NormalContactSensor(NormalContactSensorCfg(name="test", body_names=("a", "b")))
    sensor.history = torch.ones(3, 3, 2, 3)
    sensor.reset(torch.tensor([1]))
    assert not sensor.data[1].any()
    assert sensor.data[[0, 2]].all()

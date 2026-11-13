# Copyright (C) 2026 Aletheia contributors
# SPDX-License-Identifier: GPL-3.0-only
#!/usr/bin/env python3
"""Lightweight public checks for grasp geometry and measured spectral entities."""
from __future__ import annotations

import json
import numpy as np
import torch

from aletheia_absolute_kinematics import absolute_orientation, metric_grasp_targets
from aletheia_grasp_frame import parallel_jaw_rotation
from metric_entities import metric_entity


def rotation(rng: np.random.Generator) -> np.ndarray:
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def angle(matrix: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(matrix)-1.0)/2.0, -1.0, 1.0)))


def main() -> None:
    rng = np.random.default_rng(314159)
    maximum_place_identity = 0.0
    maximum_orientation_identity = 0.0
    maximum_geodesic_identity = 0.0
    for _ in range(20):
        current, transport = rotation(rng), rotation(rng)
        desired = absolute_orientation(current, transport)["rotation"]
        maximum_orientation_identity = max(
            maximum_orientation_identity, float(np.max(np.abs(desired-transport@current))))

        center = rng.normal(size=3)
        plate = rng.normal(size=3)
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        radial = np.cross(normal, rng.normal(size=3))
        radial /= np.linalg.norm(radial)
        radius, length = 0.055, 0.0446
        depth, height = 0.01, 0.03
        result = metric_grasp_targets(
            center, radius, radial, normal, length, depth, plate, height)
        expected = plate+height*normal
        maximum_place_identity = max(maximum_place_identity, float(np.max(np.abs(
            result["place_bowl_center"]-expected))))
        np.testing.assert_allclose(
            result["grasp_finger_midpoint"]-result["grasp_eef"], length*normal,
            rtol=0, atol=1e-12)

        approach = rotation(rng)@np.array([0.2, -0.3, -1.0])
        approach /= np.linalg.norm(approach)
        # The measured jaw closing axis must be orthogonal to its approach;
        # construct a valid physical frame rather than unrelated random axes.
        closing = np.cross(approach, [0.0, 1.0, 0.0])
        closing /= np.linalg.norm(closing)
        finger = current@approach*length
        mapped = parallel_jaw_rotation(current, finger, current@closing, normal, radial)
        source = np.column_stack((approach, closing, np.cross(approach, closing)))
        target_approach = -normal
        target = np.column_stack((target_approach, radial, np.cross(target_approach, radial)))
        primary = target@source.T@current.T
        alternate = target@np.diag([1.0, -1.0, -1.0])@source.T@current.T
        primary_angle, alternate_angle = angle(primary), angle(alternate)
        maximum_geodesic_identity = max(
            maximum_geodesic_identity,
            abs(mapped["primary_geodesic"]-primary_angle),
            abs(mapped["alternate_geodesic"]-alternate_angle),
        )
        np.testing.assert_allclose(mapped["rotation"], alternate if alternate_angle < primary_angle else primary,
                                   rtol=0, atol=1e-12)

    increments = rng.normal(scale=0.003, size=(12, 3))
    trajectory = torch.from_numpy(np.cumsum(increments, axis=0)).to(torch.float64)
    entity = metric_entity(trajectory, None)
    checks = {
        "randomized_kinematic_cases": 20,
        "absolute_orientation_max_abs": maximum_orientation_identity,
        "rigid_place_identity_max_abs": maximum_place_identity,
        "world_transport_geodesic_max_abs": maximum_geodesic_identity,
        "gram_dual_route_max_abs": float(entity["gram_dual_route_max_abs"]),
        "spectral_direct_sum_max_abs": float(entity["spectral_direct_sum_max_abs"]),
        "weld_inverse_max_abs": float(entity["weld_inverse_max_abs"]),
        "compass_roundtrip_max_abs": float(entity["compass_roundtrip_max_abs"]),
    }
    assert maximum_orientation_identity < 1e-12
    assert maximum_place_identity < 1e-12
    assert maximum_geodesic_identity < 1e-12
    assert checks["gram_dual_route_max_abs"] < 1e-12
    assert checks["spectral_direct_sum_max_abs"] < 1e-12
    assert checks["weld_inverse_max_abs"] < 1e-10
    # mean+(x_last-mean) is an exact algebraic readback, but floating-point
    # summation can round. Bound it by the trajectory's floating-point scale.
    readback_bound = (trajectory.shape[0]*torch.finfo(torch.float64).eps
                      * max(1.0, float(trajectory.abs().max())))
    assert checks["compass_roundtrip_max_abs"] <= readback_bound
    print(json.dumps({"passed": True, "compass_roundtrip_float64_bound": readback_bound, **checks}, indent=2))


if __name__ == "__main__":
    main()

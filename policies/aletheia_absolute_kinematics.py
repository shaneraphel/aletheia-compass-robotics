# Copyright (C) 2026 Aletheia contributors
# SPDX-License-Identifier: GPL-3.0-only
"""Absolute SO(3) composition and rigid metric grasp/place geometry.

Attribution: kimi-supervisor implementation after Aletheia's official-MLX
robotics drafts 1--4 failed their preserved executable receipts. No Aletheia
candidate is overwritten or represented as passing. The world-transport Q is
constructed separately by the already validated parallel_jaw_rotation.

No object truth, answer table, candidate scoring, or Cartesian normalization.
"""
from __future__ import annotations

import numpy as np


def _array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array


def _unit(value, name):
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"Degenerate {name}")
    return value/norm


def _nonnegative(value, name):
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return scalar


def absolute_orientation(current_rotation, world_transport):
    """Compose a world-left transport with the current orientation: D=Q R."""
    current = _array(current_rotation, (3, 3), "current_rotation")
    transport = _array(world_transport, (3, 3), "world_transport")
    for matrix, name in ((current, "current_rotation"), (transport, "world_transport")):
        residual = float(np.max(np.abs(matrix.T@matrix-np.eye(3))))
        determinant = float(np.linalg.det(matrix))
        if residual > 1e-8 or abs(determinant-1.0) > 1e-8:
            raise ValueError(f"{name} must belong to SO(3)")
    desired = transport@current
    return {
        "rotation": desired,
        "determinant": float(np.linalg.det(desired)),
        "orthonormality_max_abs": float(np.max(np.abs(desired.T@desired-np.eye(3)))),
    }


def metric_grasp_targets(rim_center, radius, radial, normal, finger_length,
                         descent, plate_center, bowl_height):
    """Place Panda's grip_site at the rim, then translate that rigid grasp.

    panda_gripper.xml puts grip_site at local z=0.097 and the finger bodies at
    z=0.0524. Thus approach=eef-fingers and grip_site is the contact point,
    not a wrist point behind the fingers. At grasp, grip_site is
    rim_center+radius*grasp_radial-descent*normal. The finger-body midpoint
    lies finger_length behind it, opposite the approach direction.
    """
    center = _array(rim_center, (3,), "rim_center")
    plate = _array(plate_center, (3,), "plate_center")
    radius = _nonnegative(radius, "radius")
    length = _nonnegative(finger_length, "finger_length")
    depth = _nonnegative(descent, "descent")
    height = _nonnegative(bowl_height, "bowl_height")
    if radius <= 1e-12 or length <= 1e-12:
        raise ValueError("radius and finger_length must be positive")
    approach_normal = _unit(_array(normal, (3,), "normal"), "normal")
    free_radial = _array(radial, (3,), "radial")
    projected = free_radial-np.dot(free_radial, approach_normal)*approach_normal
    grasp_radial = _unit(projected, "radial projection orthogonal to normal")

    grasp_eef = center+radius*grasp_radial-depth*approach_normal
    grasp_finger_midpoint = grasp_eef+length*approach_normal
    place_eef = plate+radius*grasp_radial+(height-depth)*approach_normal
    place_bowl_center = place_eef-radius*grasp_radial+depth*approach_normal
    return {
        "grasp_eef": grasp_eef,
        "place_eef": place_eef,
        "grasp_finger_midpoint": grasp_finger_midpoint,
        "place_bowl_center": place_bowl_center,
        "grasp_radial": grasp_radial,
    }

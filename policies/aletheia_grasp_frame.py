# Copyright (C) 2026 Aletheia contributors
# SPDX-License-Identifier: GPL-3.0-only
"""Sensor-only grasp frames; complete kimi-supervisor repair after Aletheia drafts 2--4 failed.

No simulator object truth, action table, candidate-action scoring, or fitted rim SVD is used.
This attribution header must remain with the module and in derived evidence.
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
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Degenerate {name}")
    return value / norm


def _orthonormality(frame):
    return float(np.max(np.abs(frame.T @ frame - np.eye(3))))


def surface_frame(surface_basis, rim_center, eef) -> dict:
    """Construct a metric tangent frame from the visible-surface SVD chart."""
    basis = _array(surface_basis, (3, 3), "surface_basis")
    center = _array(rim_center, (3,), "rim_center")
    position = _array(eef, (3,), "eef")
    if _orthonormality(basis) > 1e-8 or np.linalg.det(basis) <= 0.0:
        raise ValueError("surface_basis must be a right-handed orthonormal SO(3) chart")
    normal = _unit(basis[:, -1], "surface normal")
    if normal[2] < 0.0:
        normal = -normal
    displacement = position - center
    removed = np.dot(displacement, normal) * normal
    planar = displacement - removed
    radial = _unit(planar, "radial direction")
    tangent = np.cross(normal, radial)
    frame = np.column_stack((normal, radial, tangent))
    return {
        "normal": normal,
        "radial": radial,
        "tangent": tangent,
        "orthonormality_max_abs": _orthonormality(frame),
        "radial_residual_metres": float(np.linalg.norm(removed)),
    }


def parallel_jaw_rotation(current_rotation, finger_vector, closing_axis, normal, radial) -> dict:
    """Map measured jaw axes to the task frame in the parallel-finger Z2 quotient."""
    current = _array(current_rotation, (3, 3), "current_rotation")
    if _orthonormality(current) > 1e-8 or np.linalg.det(current) <= 0.0:
        raise ValueError("current_rotation must belong to SO(3)")
    finger = _array(finger_vector, (3,), "finger_vector")
    closing = _array(closing_axis, (3,), "closing_axis")
    task_normal = _unit(_array(normal, (3,), "normal"), "task normal")
    task_radial = _unit(_array(radial, (3,), "radial"), "task radial")

    approach_local = _unit(current.T @ finger, "approach")
    closing_unprojected = current.T @ closing
    closing_planar = closing_unprojected - np.dot(closing_unprojected, approach_local) * approach_local
    closing_local = _unit(closing_planar, "closing direction orthogonal to approach")
    source = np.column_stack((approach_local, closing_local,
                              np.cross(approach_local, closing_local)))

    target_approach = -task_normal
    target_planar = task_radial - np.dot(task_radial, target_approach) * target_approach
    target_closing = _unit(target_planar, "target closing direction")
    target = np.column_stack((target_approach, target_closing,
                              np.cross(target_approach, target_closing)))

    # The measured world frame is current@source; map that entire frame to target.
    transport = target @ source.T @ current.T
    quotient_transport = target @ np.diag([1.0, -1.0, -1.0]) @ source.T @ current.T

    def geodesic(candidate):
        # candidate=Q is already the world-left relative rotation D@current.T.
        # Its SO(3) geodesic is log(Q), not log(current.T@Q).
        cosine = (np.trace(candidate) - 1.0) / 2.0
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    primary_geodesic = geodesic(transport)
    alternate_geodesic = geodesic(quotient_transport)
    quotient_flip = bool(alternate_geodesic < primary_geodesic)
    rotation = quotient_transport if quotient_flip else transport
    return {
        "rotation": rotation,
        "quotient_flip": quotient_flip,
        "determinant": float(np.linalg.det(rotation)),
        "primary_geodesic": primary_geodesic,
        "alternate_geodesic": alternate_geodesic,
        "approach_alignment_error": float(np.linalg.norm(rotation @ current @ approach_local - target_approach)),
        "closing_alignment_abs_error": float(abs(abs(np.dot(rotation @ current @ closing_local, task_radial)) - 1.0)),
        "orthonormality_max_abs": _orthonormality(rotation),
    }

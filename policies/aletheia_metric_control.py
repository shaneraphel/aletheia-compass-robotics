# Copyright (C) 2026 Aletheia contributors
# SPDX-License-Identifier: GPL-3.0-only
#!/usr/bin/env python3
"""Real-time metric geometry and PineKF assimilation for LIBERO.

Only depth, instance segmentation, robot proprioception and task-specified
entity names enter this module. No object-pose observation or simulator object
position is accessed. Every controller update returns mathematical entities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import ConvexHull, QhullError
from scipy.linalg import expm
from robosuite.utils import camera_utils

from metric_entities import metric_entity
CAMERAS = ("agentview", "robot0_eye_in_hand")


def plain(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def semantic_instance_masks(env: Any, segmentation: np.ndarray) -> dict[str, np.ndarray]:
    """Reproduce robot_env._create_segementation_sensor's official mapping.

    SegmentationRenderEnv.instance_to_id is not used: its Panda0 special case
    is incompatible with MountedPanda0. Identity is semantic, never a pose.
    """
    # The instance sensor has already performed geom-to-instance conversion in
    # robot_env._create_segementation_sensor. A second conversion is incorrect.
    ids = np.asarray(segmentation, dtype=np.int64)
    if ids.ndim == 3:
        ids = ids[..., 0]
    return {name: ids == instance_id for name, instance_id in env.instance_to_id.items()}


def camera_cloud(env: Any, obs: dict[str, np.ndarray], camera: str,
                 names: tuple[str, ...], stride: int = 2) -> dict[str, Any]:
    """Invert the pinhole map in the sensor's native (bottom-up) row order."""
    normalized = np.asarray(obs[f"{camera}_depth"])
    if normalized.ndim == 3:
        normalized = normalized[..., 0]
    depth = camera_utils.get_real_depth_map(env.sim, normalized)
    height, width = depth.shape
    intrinsic = camera_utils.get_camera_intrinsic_matrix(env.sim, camera, height, width)
    camera_to_world = camera_utils.get_camera_extrinsic_matrix(env.sim, camera)
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    # mjr_readPixels starts at the bottom row. This pinned environment uses
    # macros.IMAGE_CONVENTION='opengl', so its sensor performs no further flip.
    native_rows = height - 1 - rows
    z = depth[::stride, ::stride].astype(np.float64)
    x = (columns - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (native_rows - intrinsic[1, 2]) / intrinsic[1, 1] * z
    homogeneous = np.stack((x, y, z, np.ones_like(z)), axis=-1)
    world = (homogeneous @ camera_to_world.T)[..., :3]
    native_pixels = np.stack((native_rows, columns), axis=-1)
    transform = camera_utils.get_camera_transform_matrix(env.sim, camera, height, width)
    projected = camera_utils.project_points_from_world_to_camera(world, transform, height, width)
    error = float(np.abs(projected - native_pixels).max())
    masks = semantic_instance_masks(env, obs[f"{camera}_segmentation_instance"])
    targets = {}
    for name in names:
        mask = masks[name][::stride, ::stride]
        targets[name] = {
            "points": world[mask],
            "pixels": native_pixels[mask],
            "depth": z[mask],
            "pixel_count": int(mask.sum()),
        }
    return {
        "camera": camera,
        "targets": targets,
        "intrinsic": intrinsic,
        "camera_to_world": camera_to_world,
        "reprojection_max_abs_pixels": error,
        "image_convention": "opengl",
        "native_row_rule": "height - 1 - sensor_row",
    }


def quadratic_surface(points: np.ndarray) -> dict[str, Any]:
    """Closed-form local graph, fundamental forms, shape spectrum and foci.

    z=p*u+q*v+(a*u^2+2*b*u*v+c*v^2)/2 in the sampled metric tangent frame.
    I=[[1+p^2,pq],[pq,1+q^2]], II=[[a,b],[b,c]]/sqrt(1+p^2+q^2),
    S=I^{-1}II. This is a local surface chart, not an action candidate fit.
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 6:
        return {"valid": False, "reason": "fewer than six visible surface points", "count": len(points)}
    if not np.isfinite(points).all():
        return {"valid": False, "reason": "non-finite depth point", "count": len(points)}
    center = points.mean(axis=0)
    xc = points - center
    _, singular, vh = np.linalg.svd(xc, full_matrices=False)
    basis = vh.T
    local = xc @ basis
    u, v, z = local.T
    # Scale the chart by its own measured RMS radius. This is an exact change
    # of coordinates, not a fitted control parameter or an arbitrary threshold.
    chart_scale = float(np.sqrt(np.mean(u*u+v*v+z*z)))
    if not math.isfinite(chart_scale) or chart_scale <= np.finfo(float).eps:
        return {"valid": False, "reason": "degenerate metric chart scale", "count": len(points),
                "center": center, "singular_values": singular, "chart_scale_metres": chart_scale}
    un, vn, zn = u/chart_scale, v/chart_scale, z/chart_scale
    design = np.column_stack((un, vn, un*un/2, un*vn, vn*vn/2, np.ones(len(u))))
    design_pinv = np.linalg.pinv(design, hermitian=False)
    normalized_parameters = design_pinv @ zn
    if not np.isfinite(normalized_parameters).all():
        raise FloatingPointError(f"Non-finite normalized surface chart: scale={chart_scale}, "
                                 f"range={np.ptp(points, axis=0)}, singular={singular}")
    p, q, an, bn, cn, dn = normalized_parameters
    # z=s*zn and u=s*un imply curvature scales as 1/s, not s.
    a, b, c = an/chart_scale, bn/chart_scale, cn/chart_scale
    normalizer = math.sqrt(1 + p*p + q*q)
    first = np.array([[1+p*p, p*q], [p*q, 1+q*q]])
    second = np.array([[a, b], [b, c]]) / normalizer
    shape = np.linalg.solve(first, second)
    # S is self-adjoint in I, generally not Euclidean-symmetric. Diagonalize
    # L^-1 II L^-T with I=L L^T, then pull eigenvectors back to the chart.
    chol = np.linalg.cholesky(first)
    metric_shape = np.linalg.solve(chol, np.linalg.solve(chol, second).T)
    metric_shape = (metric_shape+metric_shape.T)/2
    principal, metric_directions = np.linalg.eigh(metric_shape)
    directions = np.linalg.solve(chol.T, metric_directions)
    gaussian = float(np.linalg.det(shape))
    mean = float(np.trace(shape)/2)
    spectrum_identity = float(np.abs(shape@directions-directions*principal[None, :]).max())
    residual = (design @ normalized_parameters - zn)*chart_scale
    dof = max(1, len(points) - design.shape[1])
    variance = float(residual @ residual / dof)
    normalized_covariance = variance * (design_pinv @ design_pinv.T)
    units = np.diag([1, 1, 1/chart_scale, 1/chart_scale, 1/chart_scale, chart_scale])
    parameter_covariance = units @ normalized_covariance @ units
    parameters = np.array([p, q, a, b, c, dn*chart_scale])
    focus = [1/float(k) if abs(k) > np.finfo(np.float64).eps * max(1., abs(mean)) else None
             for k in principal]
    gram = xc @ xc.T
    diagonal = np.diag(gram)
    distances = np.maximum(diagonal[:, None] + diagonal[None, :] - 2*gram, 0)
    centering = np.eye(len(points)) - np.ones((len(points), len(points))) / len(points)
    dual_gram = -0.5 * centering @ distances @ centering
    result = {
        "valid": True, "count": len(points), "center": center, "basis": basis,
        "minimum": points.min(axis=0), "maximum": points.max(axis=0),
        "singular_values": singular, "chart_scale_metres": chart_scale,
        "normalized_design_rank": int(np.linalg.matrix_rank(design)),
        "graph_parameters": parameters,
        "residual_rms_metres": math.sqrt(variance),
        "parameter_covariance": parameter_covariance,
        "first_fundamental_form": first, "second_fundamental_form": second,
        "shape_operator": shape, "principal_directions": directions,
        "gaussian_curvature": gaussian, "mean_curvature": mean,
        "principal_curvatures": principal, "parallel_surface_foci_metres": focus,
        "shape_spectrum_identity_max_abs": spectrum_identity,
        "gram_coordinate_route": gram, "gram_distance_route": dual_gram,
        "gram_dual_route_max_abs": float(np.abs(gram-dual_gram).max()),
    }
    scalar_values = [gaussian, mean, variance, spectrum_identity, result["gram_dual_route_max_abs"]]
    array_values = [parameters, parameter_covariance, shape, directions, principal, gram, dual_gram]
    if not all(math.isfinite(float(value)) for value in scalar_values) or not all(
            np.isfinite(value).all() for value in array_values):
        raise FloatingPointError(f"Non-finite differential surface entity: scale={chart_scale}, "
                                 f"range={np.ptp(points, axis=0)}, singular={singular}, "
                                 f"graph={parameters}")
    return result


def rim_entity(points: np.ndarray) -> dict[str, Any]:
    """Algebraic circular rim from the measured projected convex boundary."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 6:
        return {"valid": False, "reason": "insufficient rim points", "count": len(points)}
    xy = points[:, :2]
    try:
        hull = ConvexHull(xy)
    except QhullError as exc:
        return {"valid": False, "reason": str(exc), "count": len(points)}
    rim = points[hull.vertices]
    x, y, z = rim.T
    design = np.column_stack((x, y, np.ones(len(x))))
    rhs = -(x*x + y*y)
    coefficients, *_ = np.linalg.lstsq(design, rhs, rcond=None)
    center = np.array([-coefficients[0]/2, -coefficients[1]/2])
    radius_squared = float(center@center - coefficients[2])
    if not math.isfinite(radius_squared) or radius_squared <= 0:
        return {"valid": False, "reason": "non-positive algebraic rim radius", "count": len(points)}
    radius = math.sqrt(radius_squared)
    fitted = design @ coefficients
    residual = fitted-rhs
    variance = float(residual@residual / max(1, len(x)-3))
    parameter_covariance = variance * np.linalg.pinv(design.T@design, hermitian=True)
    transform = np.array([[-.5, 0, 0], [0, -.5, 0]])
    center_covariance = transform@parameter_covariance@transform.T
    radial_error = np.abs(np.linalg.norm(rim[:, :2]-center, axis=1)-radius)
    rim_band = radial_error <= max(float(np.median(radial_error)), np.finfo(float).eps)
    rim_height = float(np.median(rim[rim_band, 2]))
    position = np.array([center[0], center[1], rim_height])
    return {
        "valid": True, "count": len(points), "rim_count": len(rim),
        "position": position, "radius_metres": radius,
        "center_covariance": center_covariance,
        "height_variance": variance / max(1, int(rim_band.sum())),
        "rim_points": rim, "algebraic_coefficients": coefficients,
        "radial_residual_rms_metres": float(np.sqrt(np.mean(radial_error**2))),
    }


def measure_geometry(env: Any, obs: dict[str, np.ndarray], names: tuple[str, ...]) -> dict[str, Any]:
    clouds = [camera_cloud(env, obs, camera, names) for camera in CAMERAS]
    targets = {}
    for name in names:
        per_camera = []
        for cloud in clouds:
            points = cloud["targets"][name]["points"]
            per_camera.append({
                "camera": cloud["camera"], "count": len(points),
                "surface": quadratic_surface(points), "rim": rim_entity(points),
            })
        combined = np.concatenate([cloud["targets"][name]["points"] for cloud in clouds], axis=0)
        targets[name] = {
            "cameras": per_camera,
            "combined_surface": quadratic_surface(combined),
            "combined_rim": rim_entity(combined),
        }
    return {
        "schema": "aletheia-real-time-metric-observation-v1",
        "ground_truth_object_positions_used": False,
        "cameras": [{key: plain(value) for key, value in cloud.items() if key != "targets"} for cloud in clouds],
        "targets": targets,
    }


@dataclass
class PineKF:
    """Constant-velocity metric state with exact Joseph covariance update.

    x=[p,v], F=[[I,dt*I],[0,I]], H=[I,0]. G=[dt^2/2,dt]^T and
    Q=sigma_a^2*G*G^T. sigma_a is the measured innovation acceleration, not a
    tuned score. R comes from the closed-form rim and surface residuals.
    """
    dt: float

    def __post_init__(self) -> None:
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None
        self.previous_innovation = np.zeros(3)
        self.updates = 0

    @staticmethod
    def from_rim(rim: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        if not rim.get("valid"):
            return None
        covariance = np.zeros((3, 3))
        covariance[:2, :2] = rim["center_covariance"]
        covariance[2, 2] = rim["height_variance"]
        # Preserve PSD and use the measured radial residual as the missing-depth
        # variance when the algebraic center covariance is rank deficient.
        floor = max(float(rim["radial_residual_rms_metres"])**2, np.finfo(float).eps**2)
        covariance += np.eye(3)*floor
        return np.asarray(rim["position"]), covariance

    def update(self, measurement: tuple[np.ndarray, np.ndarray] | None,
               advance_time: bool = True) -> dict[str, Any]:
        if measurement is None:
            return {"valid": False, "updates": self.updates, "advance_time": advance_time}
        z, R = measurement
        H = np.zeros((3, 6)); H[:, :3] = np.eye(3)
        if self.x is None:
            self.x = np.concatenate((z, np.zeros(3)))
            self.P = np.zeros((6, 6))
            self.P[:3, :3] = R
            self.P[3:, 3:] = R/(self.dt*self.dt)
        F = np.eye(6)
        if advance_time:
            F[:3, 3:] = np.eye(3)*self.dt
        G = np.concatenate((np.eye(3)*self.dt**2/2, np.eye(3)*self.dt))
        acceleration = self.previous_innovation/(self.dt*self.dt)
        Q = G@np.diag(acceleration**2)@G.T if advance_time else np.zeros((6, 6))
        predicted = F@self.x
        P_predicted = F@self.P@F.T + Q
        innovation = z-H@predicted
        self.previous_innovation = innovation
        S = H@P_predicted@H.T + R
        K = np.linalg.solve(S, H@P_predicted.T).T
        self.x = predicted + K@innovation
        identity = np.eye(6)
        residual = identity-K@H
        self.P = residual@P_predicted@residual.T + K@R@K.T
        self.P = (self.P+self.P.T)/2
        self.updates += 1
        return {
            "valid": True, "updates": self.updates, "state": self.x,
            "covariance": self.P, "innovation": innovation,
            "measurement_covariance": R, "kalman_gain": K,
            "joseph_identity_residual": float(np.abs(
                self.P-(residual@P_predicted@residual.T+K@R@K.T)).max()),
        }


def ultrametric_phase(residual: np.ndarray, tangent_hint: np.ndarray,
                      uncertainty: float) -> dict[str, Any]:
    """Dyadic circle exit and a norm-preserving skew-exponential tangent."""
    residual = np.asarray(residual, dtype=np.float64)
    norm = float(np.linalg.norm(residual))
    direction = residual/norm if norm > 0 else np.array([0., 0., 1.])
    hint = np.asarray(tangent_hint, dtype=np.float64)
    tangent = hint-direction*(direction@hint)
    if np.linalg.norm(tangent) <= np.finfo(float).eps:
        axis = np.array([0., 0., 1.]) if abs(direction[2]) < 1 else np.array([1., 0., 0.])
        tangent = np.cross(axis, direction)
    tangent /= np.linalg.norm(tangent)
    axis = np.cross(direction, tangent)
    phi = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    sigma = max(float(uncertainty), np.finfo(float).eps)
    integer = max(1, int(round(norm/sigma)))
    scale = integer & -integer
    exponent = scale.bit_length()-1
    unit = integer//scale
    center = scale*unit
    readback = center + (integer-center)
    theta = math.atan2(scale*sigma, norm) if norm else math.pi/2
    exit_direction = math.cos(theta)*direction + math.sin(theta)*tangent
    return {
        "residue": integer, "dyadic_valuation_s": exponent,
        "unit_u": unit, "circle_center": center, "roundtrip_exact": readback == integer,
        "uncertainty_metres": sigma, "phase_matrix_skew": phi,
        "phase_skew_identity_max_abs": float(np.abs(phi+phi.T).max()),
        "phase_angle": theta, "exit_direction": exit_direction,
        "phase_exponential": expm(theta*phi),
        "norm_after_exit": float(np.linalg.norm(exit_direction)),
    }


class MetricPath:
    def __init__(self, limit: int = 16) -> None:
        self.limit = limit
        self.eef: list[np.ndarray] = []
        self._metric_function = None

    def _load_metric(self):
        # The GPL release contains the measured metric operators directly; it
        # never imports the private Aletheia language-model runtime.
        return metric_entity

    def append(self, eef: np.ndarray, uncertainty: float = 0.0) -> dict[str, Any]:
        self.eef.append(np.asarray(eef, dtype=np.float64).copy())
        self.eef = self.eef[-self.limit:]
        if len(self.eef) < 3:
            return {"valid": False, "count": len(self.eef), "stuck": False}
        entity = self._load_metric()(torch.from_numpy(np.asarray(self.eef)), None)
        displacement = np.linalg.norm(self.eef[-1]-self.eef[-2])
        trapped = entity["spectral_trapped_indices"].numpy()
        eta = float(entity["spectral_eta"])
        eigenvalues = entity["laplacian_eigenvalues"].numpy()
        trapped_cutoff = float(eigenvalues[trapped].max()*eta) if len(trapped) else 0.0
        return {
            "valid": True, "count": len(self.eef), "entity": entity,
            "last_displacement_metres": displacement,
            "trapped_band_eta_lambda_max": trapped_cutoff,
            "stuck": bool(len(trapped) > 0 and displacement <= math.sqrt(np.finfo(float).eps)),
        }

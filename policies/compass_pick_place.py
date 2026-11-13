# Copyright (C) 2026 Aletheia contributors
# SPDX-License-Identifier: GPL-3.0-only
#!/usr/bin/env python3
"""Closed-form geometric pick/place controller with in-loop Aletheia fields."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from robosuite.utils import transform_utils as T

from aletheia_metric_control import PineKF, MetricPath, measure_geometry, plain, ultrametric_phase
# Supervisor wiring; the two mathematical functions are separately attributed and unit-tested.
from aletheia_grasp_frame import surface_frame, parallel_jaw_rotation
from aletheia_absolute_kinematics import absolute_orientation, metric_grasp_targets


class CompassPickPlace:
    """Servo a measured rim onto a measured plate, with no object-pose truth.

    Position actions are exact normalized OSC increments. A chunk is the
    closed-form projection of a learned action chunk onto the current metric
    task manifold. The complete difference is retained for later weight welding.
    """

    def __init__(self, env: Any, source: str = "akita_black_bowl_1", target: str = "plate_1") -> None:
        self.env = env
        self.source = source
        self.target = target
        self.names = (source, target)
        self.filters = {name: PineKF(float(env.env.control_timestep)) for name in self.names}
        self.path = MetricPath()
        controller = env.env.robots[0].controller
        self.position_scale = np.asarray(controller.output_max[:3], dtype=np.float64)
        self.orientation_scale = np.asarray(controller.output_max[3:], dtype=np.float64)
        self.dt = float(env.env.control_timestep)
        self.phase = "orient"
        self.latest: dict[str, Any] = {}
        self.initialized = False
        self.grasp_angle = 0.0
        self.grasp_angle_offset = 0.0
        self.desired_orientation = np.eye(3)
        self.orientation_anchor = np.zeros(3)
        self.finger_length = 0.0
        self.initial_aperture = 0.0
        self.descent = 0.0
        self.bowl_height = 0.0
        self.initial_rim = np.zeros(3)
        self.initial_grasp = np.zeros(3)
        self.release_position = np.zeros(3)
        self.radius = 0.0
        self.last_phase_exit: dict[str, Any] = {}

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector/norm if norm > np.finfo(float).eps else np.zeros(3)

    def _robot(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        sim = self.env.sim
        # Robot body poses are proprioception, not task-object ground truth.
        left = np.asarray(sim.data.body_xpos[sim.model.body_name2id("gripper0_leftfinger")])
        right = np.asarray(sim.data.body_xpos[sim.model.body_name2id("gripper0_rightfinger")])
        return {
            "eef": np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            "rotation": np.asarray(self.env.env.robots[0].controller.ee_ori_mat, dtype=np.float64),
            "fingers": (left+right)/2,
            "closing_axis": self._unit(right-left),
            "qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
            "aperture": float(obs["robot0_gripper_qpos"][0]-obs["robot0_gripper_qpos"][1]),
        }

    def _initialize_geometry(self, robot: dict[str, np.ndarray]) -> None:
        bowl = self.filters[self.source]
        target = self.latest["geometry"]["targets"][self.source]
        if bowl.x is None or not target["combined_rim"].get("valid"):
            return
        rim = target["combined_rim"]
        surface = target["combined_surface"]
        if not surface.get("valid"):
            return
        self.radius = float(rim["radius_metres"])
        self.initial_rim = bowl.x[:3].copy()
        eef = robot["eef"]
        principal = np.asarray(surface["principal_curvatures"])
        largest = max(float(np.max(np.abs(principal))), np.finfo(float).eps)
        focus_radius = 1/largest
        self.finger_length = float(np.linalg.norm(robot["fingers"]-eef))
        # Parallel-surface focal bound and actual gripper geometry determine
        # the maximum penetration. There is no coordinate or depth lookup.
        self.descent = min(focus_radius, self.finger_length/2)
        self.bowl_height = float(max(0.0, self.initial_rim[2]-float(surface["minimum"][2])))
        self.initial_aperture = robot["aperture"]
        # The visible-surface chart defines one parallel-jaw SO(3)/Z2 frame.
        # Q is a world transport, never an absolute target orientation.
        measured_frame = surface_frame(np.asarray(surface["basis"]), self.initial_rim, eef)
        self.base_surface_normal = np.asarray(measured_frame["normal"], dtype=np.float64)
        self.grasp_angle = math.atan2(measured_frame["radial"][1], measured_frame["radial"][0])
        self.surface_orthonormality_max_abs = measured_frame["orthonormality_max_abs"]
        self.surface_radial_residual_metres = measured_frame["radial_residual_metres"]
        self.orientation_anchor = robot["eef"].copy()
        self._update_orientation(robot)
        self.initial_grasp = self._grasp_position()
        self.initialized = True

    def _metric_targets(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        angle = self.grasp_angle+self.grasp_angle_offset
        cosine, sine = math.cos(self.grasp_angle_offset), math.sin(self.grasp_angle_offset)
        # The rim was fitted in the world XY chart. A new quotient exit rotates
        # its measured normal with the radial frame, not just the position.
        rotation_z = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
        normal = rotation_z@self.base_surface_normal
        radial = np.array([math.cos(angle), math.sin(angle), 0.0])
        targets = metric_grasp_targets(
            self.filters[self.source].x[:3], self.radius, radial, normal,
            self.finger_length, self.descent, self.filters[self.target].x[:3],
            self.bowl_height,
        )
        return targets, normal

    def _update_orientation(self, robot: dict[str, np.ndarray]) -> None:
        targets, normal = self._metric_targets()
        transport = parallel_jaw_rotation(
            robot["rotation"], robot["eef"]-robot["fingers"], robot["closing_axis"],
            normal, targets["grasp_radial"],
        )
        absolute = absolute_orientation(robot["rotation"], transport["rotation"])
        self.world_transport = np.asarray(transport["rotation"], dtype=np.float64)
        self.desired_orientation = absolute["rotation"]
        self.surface_normal = normal
        self.grasp_radial = targets["grasp_radial"]
        self.grasp_approach_alignment_error = transport["approach_alignment_error"]
        self.grasp_closing_alignment_abs_error = transport["closing_alignment_abs_error"]
        self.quotient_flip = transport["quotient_flip"]
        self.quotient_primary_geodesic = transport["primary_geodesic"]
        self.quotient_alternate_geodesic = transport["alternate_geodesic"]
        self.absolute_orientation_determinant = absolute["determinant"]
        self.absolute_orientation_orthonormality_max_abs = absolute["orthonormality_max_abs"]

    def _grasp_position(self) -> np.ndarray:
        return self._metric_targets()[0]["grasp_eef"]

    def _place_position(self) -> np.ndarray:
        return self._metric_targets()[0]["place_eef"]

    def assimilate(self, obs: dict[str, np.ndarray]) -> dict[str, Any]:
        geometry = measure_geometry(self.env, obs, self.names)
        assimilation = {}
        for name in self.names:
            updates = []
            for index, camera in enumerate(geometry["targets"][name]["cameras"]):
                updates.append(self.filters[name].update(
                    PineKF.from_rim(camera["rim"]), advance_time=index == 0))
            assimilation[name] = updates
        robot = self._robot(obs)
        uncertainty = (math.sqrt(max(0., float(np.trace(self.filters[self.source].P[:3, :3]))))
                       if self.filters[self.source].P is not None else 0.)
        path = self.path.append(robot["eef"], uncertainty)
        # MetricPath passes this measured metric uncertainty to its spectral
        # trapped-band decision; no arbitrary time or position threshold.
        path["measurement_uncertainty_metres"] = uncertainty
        path["stuck"] = bool(path.get("valid") and len(path["entity"]["spectral_trapped_indices"]) > 0
                             and path["last_displacement_metres"] <= uncertainty)
        self.latest = {"geometry": geometry, "assimilation": assimilation,
                       "robot": robot, "path": path}
        if not self.initialized:
            self._initialize_geometry(robot)
        else:
            # Recompose Q with the current measured robot frame, including after
            # restore; D=Q R remains absolute while the finger frame moves.
            self._update_orientation(robot)
        return self.latest

    def state(self) -> dict[str, Any]:
        return {
            "phase": self.phase, "initialized": self.initialized,
            "grasp_angle": self.grasp_angle, "grasp_angle_offset": self.grasp_angle_offset,
            "desired_orientation": self.desired_orientation.tolist(),
            "orientation_anchor": self.orientation_anchor.tolist(),
            "world_transport": self.world_transport.tolist() if hasattr(self, "world_transport") else None,
            "base_surface_normal": self.base_surface_normal.tolist() if hasattr(self, "base_surface_normal") else None,
            "surface_normal": self.surface_normal.tolist() if hasattr(self, "surface_normal") else None,
            "grasp_radial": self.grasp_radial.tolist() if hasattr(self, "grasp_radial") else None,
            "absolute_orientation_determinant": getattr(self, "absolute_orientation_determinant", None),
            "absolute_orientation_orthonormality_max_abs": getattr(self, "absolute_orientation_orthonormality_max_abs", None),
            "finger_length": self.finger_length, "initial_aperture": self.initial_aperture,
            "descent": self.descent, "bowl_height": self.bowl_height,
            "initial_rim": self.initial_rim.tolist(), "initial_grasp": self.initial_grasp.tolist(),
            "release_position": self.release_position.tolist(), "radius": self.radius,
            "surface_orthonormality_max_abs": getattr(self, "surface_orthonormality_max_abs", None),
            "surface_radial_residual_metres": getattr(self, "surface_radial_residual_metres", None),
            "grasp_approach_alignment_error": getattr(self, "grasp_approach_alignment_error", None),
            "grasp_closing_alignment_abs_error": getattr(self, "grasp_closing_alignment_abs_error", None),
            "quotient_flip": getattr(self, "quotient_flip", False),
            "quotient_primary_geodesic": getattr(self, "quotient_primary_geodesic", None),
            "quotient_alternate_geodesic": getattr(self, "quotient_alternate_geodesic", None),
        }

    def restore(self, state: dict[str, Any]) -> None:
        self.phase = state["phase"]
        self.initialized = bool(state["initialized"])
        self.grasp_angle = float(state["grasp_angle"])
        self.grasp_angle_offset = float(state["grasp_angle_offset"])
        self.desired_orientation = np.asarray(state["desired_orientation"])
        self.orientation_anchor = np.asarray(state["orientation_anchor"], dtype=np.float64)
        if state.get("base_surface_normal") is None:
            raise ValueError("Absolute-orientation resume requires the measured base surface normal")
        self.base_surface_normal = np.asarray(state["base_surface_normal"], dtype=np.float64)
        self.finger_length = float(state["finger_length"])
        self.initial_aperture = float(state["initial_aperture"])
        self.descent = float(state["descent"])
        self.bowl_height = float(state["bowl_height"])
        self.initial_rim = np.asarray(state["initial_rim"])
        self.initial_grasp = np.asarray(state["initial_grasp"])
        self.release_position = np.asarray(state["release_position"])
        self.radius = float(state["radius"])
        for name in ("surface_orthonormality_max_abs", "surface_radial_residual_metres",
                     "grasp_approach_alignment_error", "grasp_closing_alignment_abs_error",
                     "quotient_primary_geodesic", "quotient_alternate_geodesic",
                     "absolute_orientation_determinant", "absolute_orientation_orthonormality_max_abs"):
            if state.get(name) is not None:
                setattr(self, name, float(state[name]))
        self.quotient_flip = bool(state.get("quotient_flip", False))

    def restore_episode_evidence(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Restore pre-current-observation entities from the committed trajectory.

        Each row stores the two PineKF updates and metric path immediately before
        its action. Replaying physics reaches the observation after the last row;
        restoring the last row's pre-action entities and assimilating that final
        observation once is therefore identical to rebuilding every historical
        geometry update, without repeating those O(T) surface constructions.
        """
        if not rows:
            raise ValueError("Episode-evidence restore requires at least one committed row")
        restored_from: dict[str, int] = {}
        for name in self.names:
            for row in reversed(rows):
                updates = row["math"]["targets"][name]["pinekf"]
                valid = [update for update in updates if update.get("valid")]
                if valid:
                    self.filters[name].restore(valid[-1])
                    restored_from[name] = int(row["step"])
                    break
            else:
                raise ValueError(f"No committed valid PineKF entity for {name}")
        # state[:3] is only float32 and must not be promoted as a counterfeit
        # float64 path. The committed path entity retains the original float64
        # observations immediately before the last action; restore those exactly.
        path_entity = rows[-1]["math"]["path"].get("entity")
        if not path_entity or "observations" not in path_entity:
            raise ValueError("Fast resume requires the committed float64 MetricPath observations")
        positions = [np.asarray(position, dtype=np.float64)
                     for position in np.asarray(path_entity["observations"], dtype=np.float64)]
        self.path.restore(positions)
        return {
            "mode": "committed_entity_restore_plus_one_current_assimilation",
            "pinekf_source_steps": restored_from,
            "metric_path_source_step": int(rows[-1]["step"]),
            "metric_path_committed_positions_before_current": len(self.path.eef),
            "historical_surface_assimilations_skipped": len(rows),
        }

    def _advance_phase(self) -> None:
        if not self.initialized:
            return
        robot = self.latest["robot"]
        eef = robot["eef"]
        P = self.filters[self.source].P[:3, :3]
        sigma = math.sqrt(max(float(np.trace(P)), np.finfo(float).eps))
        radial_rms = float(self.latest["geometry"]["targets"][self.source][
            "combined_rim"]["radial_residual_rms_metres"])
        tolerance = math.sqrt(sigma*sigma+radial_rms*radial_rms)
        grasp = self._grasp_position()
        horizontal = float(np.linalg.norm(eef[:2]-grasp[:2]))
        relative = self.desired_orientation@robot["rotation"].T
        self.orientation_geodesic = float(np.arccos(np.clip(
            (np.trace(relative)-1.0)/2.0, -1.0, 1.0)))
        # A residual rotation delta_theta moves the fingertip laterally by
        # approximately finger_length*delta_theta. Approach is permitted only
        # when that displacement lies inside the measured metric tolerance.
        self.orientation_tolerance = math.asin(min(1.0, tolerance/self.finger_length))
        if self.phase == "orient" and self.orientation_geodesic <= self.orientation_tolerance:
            self.phase = "approach"
        elif self.phase == "approach" and horizontal <= tolerance and eef[2] >= grasp[2]:
            self.phase = "descend"
        elif self.phase == "descend" and horizontal <= tolerance and abs(eef[2]-grasp[2]) <= tolerance:
            self.phase = "close"
        elif self.phase == "close":
            wall_bound = 2*radial_rms+sigma
            if np.finfo(np.float32).eps < robot["aperture"] <= wall_bound:
                self.phase = "lift"
            elif robot["aperture"] <= np.finfo(np.float32).eps:
                self.phase = "open_recovery"
        elif self.phase == "open_recovery" and robot["aperture"] >= self.initial_aperture-np.finfo(np.float32).eps:
            exit_field = ultrametric_phase(eef-grasp, np.array([-math.sin(self.grasp_angle),
                                               math.cos(self.grasp_angle), 0.]), sigma)
            self.grasp_angle_offset += math.pi*2**(-exit_field["dyadic_valuation_s"])
            self._update_orientation(robot)
            self.phase = "approach"
        elif self.phase == "lift" and eef[2] >= self.initial_grasp[2]+self.radius/2:
            measured_lift = self.filters[self.source].x[2]-self.initial_rim[2]
            if measured_lift >= self.radius/4:
                self.phase = "transport"
            else:
                self.phase = "open_recovery"
        elif self.phase == "transport":
            place = self._place_position()
            if np.linalg.norm(eef[:2]-place[:2]) <= tolerance:
                self.phase = "place"
        elif self.phase == "place":
            place = self._place_position()
            plate_rms = float(self.latest["geometry"]["targets"][self.target][
                "combined_surface"]["residual_rms_metres"])
            if np.linalg.norm(eef[:2]-place[:2]) <= tolerance and abs(eef[2]-place[2]) <= math.hypot(tolerance, plate_rms):
                self.phase = "release"
                self.release_position = eef.copy()
        elif self.phase == "release" and robot["aperture"] >= self.initial_aperture-np.finfo(np.float32).eps:
            self.phase = "retreat"

    def _target(self) -> np.ndarray:
        grasp = self._grasp_position()
        place = self._place_position()
        eef = self.latest["robot"]["eef"]
        if self.phase == "orient":
            return self.orientation_anchor.copy()
        if self.phase == "approach" or self.phase == "open_recovery":
            return grasp+np.array([0., 0., self.radius])
        if self.phase in ("descend", "close"):
            return grasp
        if self.phase == "lift":
            return self.initial_grasp+np.array([0., 0., self.radius])
        if self.phase == "transport":
            return place+np.array([0., 0., self.radius])
        if self.phase in ("place", "release"):
            return place
        return self.release_position+np.array([0., 0., self.bowl_height])

    def project_chunk(self, raw_chunk: np.ndarray) -> dict[str, Any]:
        self._advance_phase()
        if not self.initialized:
            raise RuntimeError("Metric geometry is not observable; refusing blind control")
        raw = np.asarray(raw_chunk, dtype=np.float32)
        if raw.shape != (50, 7):
            raise RuntimeError(f"Expected a [50,7] learned action chunk, got {raw.shape}")
        eef = self.latest["robot"]["eef"]
        residual = self._target()-eef
        sigma = math.sqrt(max(float(np.trace(self.filters[self.source].P[:3, :3])), np.finfo(float).eps))
        tangent = np.asarray(self.latest["geometry"]["targets"][self.source][
            "combined_surface"]["basis"])[:, 0]
        exit_field = ultrametric_phase(residual, tangent, sigma)
        stuck = bool(self.latest["path"].get("stuck"))
        free = self.phase in ("approach", "open_recovery", "lift", "transport", "retreat")
        if stuck and free:
            residual = exit_field["exit_direction"]*np.linalg.norm(residual)
        count = max(1, int(math.ceil(np.linalg.norm(residual)/self.position_scale[0])))
        horizon = min(50, count) if free and not stuck else 1
        position_step = residual/count
        rotation_error = T.quat2axisangle(T.mat2quat(
            self.desired_orientation@self.latest["robot"]["rotation"].T))
        orientation_step = rotation_error/max(1, min(50, int(math.ceil(
            np.linalg.norm(rotation_error)/self.orientation_scale[0]))))
        # Robosuite PandaGripper.format_action: -1 opens, +1 closes.
        gripper = 1.0 if self.phase in ("close", "lift", "transport", "place") else -1.0
        desired = np.zeros((50, 7), dtype=np.float64)
        desired[:, :3] = np.clip(position_step/self.position_scale, -1, 1)
        desired[:, 3:6] = np.clip(orientation_step/self.orientation_scale, -1, 1)
        desired[:, 6] = gripper
        executed = desired.astype(np.float32)
        delta = executed-raw
        self.last_phase_exit = exit_field
        return {
            "corrected_chunk": executed, "delta_chunk": delta, "horizon": horizon,
            "phase": self.phase, "target_position": self._target(),
            "position_residual": residual, "orientation_error": rotation_error,
            "measurement_sigma": sigma, "stuck": stuck,
            "ultrametric_phase": exit_field, "controller_state": self.state(),
        }


def compact_assimilation(latest: dict[str, Any]) -> dict[str, Any]:
    """Keep mathematical evidence compact; no image-sized N-by-N JSON tables."""
    result = {"targets": {}, "path": plain(latest["path"])}
    for name, target in latest["geometry"]["targets"].items():
        surface = target["combined_surface"]
        rim = target["combined_rim"]
        points = np.concatenate([camera["surface"]["center"][None, :] for camera in target["cameras"]
                                 if camera["surface"].get("valid")], axis=0)
        result["targets"][name] = {
            "pinekf": plain(latest["assimilation"][name]),
            "surface": {key: plain(surface[key]) for key in (
                "count", "minimum", "maximum", "singular_values", "graph_parameters",
                "residual_rms_metres", "first_fundamental_form", "second_fundamental_form",
                "shape_operator", "principal_directions", "gaussian_curvature", "mean_curvature",
                "principal_curvatures", "parallel_surface_foci_metres", "gram_dual_route_max_abs")},
            "rim": {key: plain(rim[key]) for key in (
                "count", "rim_count", "position", "radius_metres", "radial_residual_rms_metres",
                "algebraic_coefficients")},
            "camera_surface_centers": points,
        }
    return result

# Copyright (C) 2026 Aletheia contributors
# SPDX-License-Identifier: GPL-3.0-only
#!/usr/bin/env python3
"""Run or independently replay a real in-loop Aletheia LIBERO episode."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import libero_smolvla_episode as base
from compass_pick_place import CompassPickPlace, compact_assimilation
from aletheia_metric_control import plain

SCHEMA = "aletheia-libero-in-loop-compass-pinekf-v1"
REPLAY_SCHEMA = "aletheia-libero-in-loop-independent-replay-v1"
LIMIT = base.OFFICIAL_SPATIAL_LIMIT
SEED = base.SEED


def sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def tensor_tree(value: Any) -> Any:
    """Recursively remove NumPy scalars so weights_only checkpoints stay readable."""
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: tensor_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensor_tree(item) for item in value)
    return value


def observation_hashes(obs: dict[str, np.ndarray], sim_state: np.ndarray) -> dict[str, str]:
    return {
        "sim_state_sha256": sha(sim_state),
        "state_sha256": sha(base.proprio_state(obs)),
        "agentview_raw_sha256": sha(obs["agentview_image"]),
        "wrist_raw_sha256": sha(obs["robot0_eye_in_hand_image"]),
        "agentview_depth_sha256": sha(obs["agentview_depth"]),
        "wrist_depth_sha256": sha(obs["robot0_eye_in_hand_depth"]),
        "agentview_segmentation_sha256": sha(obs["agentview_segmentation_instance"]),
        "wrist_segmentation_sha256": sha(obs["robot0_eye_in_hand_segmentation_instance"]),
    }


def checkpoint(path: Path, init_state: int) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], {}, {}
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("init_state") != init_state or payload.get("seed") != SEED:
        raise RuntimeError("Checkpoint schema, init state, or seed mismatch")
    rows = payload["rows"]
    for index, row in enumerate(rows):
        if row["step"] != index or np.asarray(row["executed_action"]).shape != (7,):
            raise RuntimeError(f"Invalid committed step {index}")
    return rows, payload["policy_chunks"], payload.get("metadata", {})


def history(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else []


def summary(rows: list[dict[str, Any]], init_state: int, task: Any, output: Path,
            policy_details: dict[str, Any], timings: dict[str, float], runs: list[dict[str, Any]],
            replay: dict[str, Any] | None = None) -> dict[str, Any]:
    success = next((row["step"]+1 for row in rows if row["check_success"]), None)
    phases = {}
    for row in rows:
        phase = row["decision"]["phase"]
        phases[phase] = phases.get(phase, 0)+1
    complete_runs = [run for run in runs if run.get("wall_seconds") is not None]
    return {
        "schema": SCHEMA, "init_state": init_state, "task": task.name,
        "language": task.language, "goal_state": [["on", "akita_black_bowl_1", "plate_1"]],
        "status": "success" if success else ("complete_failed" if len(rows) >= LIMIT else "resumable"),
        "check_success": success is not None, "first_check_success_step": success,
        "actions_executed": len(rows), "policy_inference_calls": len({row["policy_chunk_id"] for row in rows}),
        "official_spatial_action_limit": LIMIT, "phases": phases,
        "controller": "Real-time point-cloud Gram + I/II/S + foci + PineKF + secular/trapped bands + skew ultrametric exits",
        "ground_truth_object_positions_used_for_control": False,
        "force_sensor_available": False, "force_closure_claimed": False,
        "mathematical_actions_nonzero": sum(bool(row["delta_action_l2"] > 0) for row in rows),
        "delta_action_l2_sum": float(sum(row["delta_action_l2"] for row in rows)),
        "policy": policy_details,
        "timings_this_invocation_seconds": timings,
        "completed_invocation_wall_seconds_sum": sum(run["wall_seconds"] for run in complete_runs),
        "run_count": len(complete_runs), "trajectory_file": str(output / "trajectory.pt"),
        "baseline_comparison": {
            "original_smolvla_chunk4_init2": {"actions": 220, "check_success": False,
                "evidence": "fieldrt/results/libero_chunk4_v1/smolvla_chunk4_seed7_init2/report.json"},
            "this_controller": {"actions": len(rows), "check_success": success is not None},
        },
        "single_episode_not_success_rate": True, "leaderboard_claim": False,
        "dexterous_hand_or_force_control_claim": False,
        "final_independent_replay": replay,
    }


def main() -> None:
    invocation_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-state", type=int, required=True)
    parser.add_argument("--max-new-actions", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--inspect-geometry", action="store_true",
                        help="Persist initial sensor-only geometry without loading a policy or executing actions")
    parser.add_argument("--inspect-replay-step", type=int, default=-1,
                        help="Stop after this committed action and persist its measured grasp geometry")
    args = parser.parse_args()
    if not 0 <= args.init_state < 50 or not 1 <= args.max_new_actions <= LIMIT:
        raise ValueError("Invalid init-state or action limit")
    output = Path(os.environ.get("LIBERO_OUTPUT", base.ROOT /
        f"results/libero_geometry_control_v1/compass_init{args.init_state}"))
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.pt"
    rows, chunks, metadata = checkpoint(trajectory_path, args.init_state)
    runs = history(output / "run_history.json")
    timings = {"model_load": 0., "inference": 0., "geometry": 0., "physics": 0., "serialization": 0., "replay": 0.}
    new_actions = 0
    policy_details: dict[str, Any] = {}
    error = None

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    states = base.load_initial_states(suite, 0)
    initial = np.asarray(states[args.init_state])
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = SegmentationRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                               camera_depths=True, camera_segmentations="instance")
    try:
        env.seed(SEED)
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(base.WAIT_STEPS):
            obs, _, _, _ = env.step(base.WAIT_ACTION)
        settled = observation_hashes(obs, np.asarray(env.get_sim_state()))
        if metadata.get("settled") and metadata["settled"] != settled:
            raise RuntimeError("Settled sensor state differs from committed episode")
        if args.verify_only:
            if not rows:
                raise RuntimeError("No trajectory to verify")
            # Diagnostic: preserve the original MPS policy context without doing
            # any inference; all executed actions still come from the checkpoint.
            policy_context_loaded = os.environ.get("ALETHEIA_REPLAY_POLICY_CONTEXT") == "1"
            if policy_context_loaded:
                base.load_policy()
            replay_started = time.perf_counter()
            corrected_roundtrip_max_abs = 0.0
            corrected_roundtrip_max_bound = 0.0
            float32_epsilon = float(np.finfo(np.float32).eps)
            for row in rows:
                raw_chunk = np.asarray(chunks[row["policy_chunk_id"]]["postprocessed"])
                expected_raw = raw_chunk[row["chunk_offset"]]
                raw = np.asarray(row["raw_action"])
                corrected_chunk = np.asarray(row["corrected_action_chunk"])
                delta_chunk = np.asarray(row["delta_action_chunk"])
                delta = np.asarray(row["delta_action"])
                executed = np.asarray(row["executed_action"])
                if not np.array_equal(raw, expected_raw):
                    raise RuntimeError(f"Raw-action lineage mismatch at {row['step']}")
                if corrected_chunk.shape != (50, 7) or delta_chunk.shape != (50, 7):
                    raise RuntimeError(f"Corrected-chunk shape mismatch at {row['step']}")
                if not np.array_equal(delta_chunk, corrected_chunk-raw_chunk):
                    raise RuntimeError(f"Whole-chunk delta identity mismatch at {row['step']}")
                # The stored delta is float32(corrected-raw), so raw+delta can
                # differ from nominal corrected by one float32 rounding. The
                # action sent to physics must nevertheless be bit-identical to
                # the stored raw+delta result, with a strict round-trip bound.
                if not np.array_equal(executed, (raw+delta).astype(np.float32)):
                    raise RuntimeError(f"Mathematical action identity mismatch at {row['step']}")
                nominal_corrected = corrected_chunk[row["chunk_offset"]]
                roundtrip_residual = np.abs(
                    executed.astype(np.float64)-nominal_corrected.astype(np.float64))
                roundtrip_bound = float32_epsilon*(
                    np.abs(raw.astype(np.float64))+np.abs(delta.astype(np.float64)))
                if np.any(roundtrip_residual > roundtrip_bound):
                    raise RuntimeError(
                        f"Corrected-action float32 round-trip mismatch at {row['step']}: "
                        f"max_abs={float(np.max(roundtrip_residual))}, "
                        f"bound={float(np.max(roundtrip_bound))}")
                corrected_roundtrip_max_abs = max(
                    corrected_roundtrip_max_abs, float(np.max(roundtrip_residual)))
                corrected_roundtrip_max_bound = max(
                    corrected_roundtrip_max_bound, float(np.max(roundtrip_bound)))
                if float(row["delta_action_l2"]) != float(np.linalg.norm(delta)):
                    raise RuntimeError(f"Delta magnitude mismatch at {row['step']}")
                obs, reward, done, _ = env.step(executed.tolist())
                current = observation_hashes(obs, np.asarray(env.get_sim_state()))
                expected = {key: row[key] for key in current}
                if current != expected or bool(done) != row["done"] or bool(env.check_success()) != row["check_success"]:
                    raise RuntimeError(f"Independent sensor/physics replay mismatch at {row['step']}")
            verification = {
                "schema": REPLAY_SCHEMA, "verified": True, "policy_loaded": policy_context_loaded,
                "policy_inference_performed": False,
                "mathematical_controller_loaded": False, "init_state": args.init_state,
                "actions": len(rows), "replayed_check_success": bool(env.check_success()),
                "recorded_check_success": bool(rows[-1]["check_success"]),
                "all_observation_hashes_verified": True,
                "all_action_delta_identities_verified": True,
                "corrected_action_float32_roundtrip_max_abs": corrected_roundtrip_max_abs,
                "corrected_action_float32_roundtrip_max_bound": corrected_roundtrip_max_bound,
                "float32_epsilon": float32_epsilon,
                "corrected_action_float32_roundtrip_bound": "eps*(abs(raw)+abs(delta)) elementwise",
                "nonzero_delta_actions": sum(bool(row["delta_action_l2"] > 0) for row in rows),
                "trajectory_bytes": trajectory_path.stat().st_size,
                "trajectory_sha256": base.sha_file(trajectory_path),
                "wall_seconds": time.perf_counter()-replay_started,
                "single_episode_not_success_rate": True,
            }
            base.atomic_json(output / "independent_replay_verification.json", verification)
            report_path = output / "report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                report["final_independent_replay"] = verification
                base.atomic_json(report_path, report)
            base.save_frame(output / "replay_final_agentview.png", base.upright(obs["agentview_image"]))
            base.save_frame(output / "replay_final_wrist.png", base.upright(obs["robot0_eye_in_hand_image"]))
            print(json.dumps(verification, indent=2), flush=True)
            return

        controller = CompassPickPlace(env)
        if metadata.get("controller_state"):
            controller.restore(metadata["controller_state"])
        # Inspection of an arbitrary historical row still uses full assimilation.
        # Ordinary resume restores the last committed mathematical entities and
        # assimilates only the final replayed observation, avoiding O(T^2) work
        # across repeated timeout/resume invocations.
        committed_path = rows[-1]["math"]["path"].get("entity") if rows else None
        fast_entity_restore = bool(
            rows and args.inspect_replay_step < 0
            and committed_path and "observations" in committed_path)
        entity_restore = {"mode": "full_historical_geometry_assimilation"}
        if fast_entity_restore:
            entity_restore = controller.restore_episode_evidence(rows)
        else:
            started = time.perf_counter()
            controller.assimilate(obs)
            timings["geometry"] += time.perf_counter()-started
        if args.inspect_geometry:
            if rows or not controller.initialized:
                raise RuntimeError("Initial geometry inspection requires a fresh observable episode")
            from aletheia_grasp_frame import surface_frame
            source = controller.latest["geometry"]["targets"][controller.source]
            measured = surface_frame(np.asarray(source["combined_surface"]["basis"]),
                                     controller.initial_rim, controller.latest["robot"]["eef"])
            inspection = {
                "schema": "aletheia-libero-initial-geometry-v1", "init_state": args.init_state,
                "settled": settled, "robot": plain(controller.latest["robot"]),
                "surface": plain({key: source["combined_surface"][key] for key in (
                    "center", "basis", "minimum", "maximum", "singular_values", "principal_curvatures")}),
                "rim": plain({key: source["combined_rim"][key] for key in (
                    "position", "radius_metres", "radial_residual_rms_metres")}),
                "measured_frame": plain(measured), "controller_state": controller.state(),
                "ground_truth_object_positions_used": False,
            }
            base.atomic_json(output / "initial_geometry.json", inspection)
            base.save_frame(output / "initial_agentview.png", base.upright(obs["agentview_image"]))
            base.save_frame(output / "initial_wrist.png", base.upright(obs["robot0_eye_in_hand_image"]))
            print(json.dumps(inspection, indent=2), flush=True)
            return
        replay_started = time.perf_counter()
        replay_geometry_started = timings["geometry"]
        resume_agentview_hash_mismatches = []
        for row in rows:
            obs, replay_reward, replay_done, _ = env.step(np.asarray(row["executed_action"]).tolist())
            current = observation_hashes(obs, np.asarray(env.get_sim_state()))
            expected = {key: row[key] for key in current}
            differences = {key: {"committed": expected[key], "replayed": current[key]}
                           for key in current if current[key] != expected[key]}
            outcome_matches = bool(replay_done) == row["done"] and bool(env.check_success()) == row["check_success"]
            # macOS MuJoCo can reproduce physics, depth, segmentation and the wrist
            # image bit-exactly while the agentview RGB buffer differs. Never hide
            # that exception: only resume may continue, with a per-step sidecar;
            # independent --verify-only remains strict for every observation hash.
            if outcome_matches and set(differences) == {"agentview_raw_sha256"}:
                resume_agentview_hash_mismatches.append({"step": row["step"], **differences})
                print(json.dumps({"resume_render_exception": row["step"], **differences}), flush=True)
            elif differences or not outcome_matches:
                raise RuntimeError(f"Resume replay mismatch at {row['step']}: " + json.dumps({
                    "sensor_hashes": differences, "committed_done": row["done"],
                    "replayed_done": bool(replay_done), "committed_success": row["check_success"],
                    "replayed_success": bool(env.check_success())}))
            if not fast_entity_restore:
                geometry_started = time.perf_counter()
                controller.assimilate(obs)
                timings["geometry"] += time.perf_counter()-geometry_started
            if row["step"] == args.inspect_replay_step:
                robot = controller.latest["robot"]
                source = controller.latest["geometry"]["targets"][controller.source]
                grasp = controller._grasp_position()
                sim = env.sim
                left = np.asarray(sim.data.body_xpos[sim.model.body_name2id("gripper0_leftfinger")])
                right = np.asarray(sim.data.body_xpos[sim.model.body_name2id("gripper0_rightfinger")])
                osc = env.env.robots[0].controller
                joint_position = np.asarray(osc.joint_pos)
                joint_range = np.asarray(sim.model.jnt_range[osc.joint_index])
                robot_dynamics = {
                    "joint_names": [sim.model.joint_id2name(int(index)) for index in osc.joint_index],
                    "joint_position": joint_position,
                    "joint_range": joint_range,
                    "joint_limit_margin": np.minimum(joint_position-joint_range[:, 0], joint_range[:, 1]-joint_position),
                    "joint_velocity": np.asarray(osc.joint_vel),
                    "jacobian_position_singular_values": np.linalg.svd(osc.J_pos, compute_uv=False),
                    "jacobian_orientation_singular_values": np.linalg.svd(osc.J_ori, compute_uv=False),
                    "goal_position": np.asarray(osc.goal_pos),
                    "controller_position": np.asarray(osc.ee_pos),
                    "controller_position_error": np.asarray(osc.goal_pos)-np.asarray(osc.ee_pos),
                    "controller_torques": np.asarray(osc.torques),
                    "actuator_minimum": np.asarray(osc.actuator_min),
                    "actuator_maximum": np.asarray(osc.actuator_max),
                    "kp": np.asarray(osc.kp),
                }
                inspection = {
                    "schema": "aletheia-libero-replay-grasp-geometry-v2", "step": row["step"],
                    "robot_dynamics": plain(robot_dynamics),
                    "phase": row["decision"]["phase"], "check_success": row["check_success"],
                    "committed_decision": plain(row["decision"]), "robot": plain(robot),
                    "left_finger_body": left, "right_finger_body": right,
                    "current_grasp_eef_target": grasp,
                    "finger_midpoint_minus_eef_target": robot["fingers"]-grasp,
                    "finger_midpoint_horizontal_error_metres": float(np.linalg.norm(robot["fingers"][:2]-grasp[:2])),
                    "measured_rim": plain({key: source["combined_rim"][key] for key in (
                        "position", "radius_metres", "radial_residual_rms_metres")}),
                    "measured_surface": plain({key: source["combined_surface"][key] for key in (
                        "center", "basis", "minimum", "maximum", "singular_values")}),
                    "physics_and_non_agentview_replay_verified_through_step": True,
                    "agentview_rgb_exception": bool(resume_agentview_hash_mismatches),
                    "ground_truth_object_positions_used": False,
                }
                base.atomic_json(output / f"replay_geometry_step_{row['step']:04d}.json", plain(inspection))
                base.save_frame(output / f"replay_geometry_step_{row['step']:04d}_agentview.png",
                                base.upright(obs["agentview_image"]))
                base.save_frame(output / f"replay_geometry_step_{row['step']:04d}_wrist.png",
                                base.upright(obs["robot0_eye_in_hand_image"]))
                print(json.dumps(plain(inspection), indent=2), flush=True)
                return
        if fast_entity_restore:
            geometry_started = time.perf_counter()
            controller.assimilate(obs)
            timings["geometry"] += time.perf_counter()-geometry_started
        base.atomic_json(output / "resume_replay_verification.json", {
            "schema": "aletheia-libero-resume-replay-v1", "committed_steps_replayed": len(rows),
            "physics_and_non_agentview_hashes_verified": True,
            "all_observation_hashes_verified": not resume_agentview_hash_mismatches,
            "agentview_rgb_only_mismatches": resume_agentview_hash_mismatches,
            "controller_entity_restore": entity_restore,
            "strict_independent_replay_remains_required": True,
        })
        timings["replay"] = (time.perf_counter()-replay_started
                             -(timings["geometry"]-replay_geometry_started))
        if rows and (rows[-1]["check_success"] or rows[-1]["done"] or len(rows) >= LIMIT):
            raise RuntimeError("Episode is already complete")

        load_started = time.perf_counter()
        policy, preprocess, postprocess, policy_details = base.load_policy()
        timings["model_load"] = time.perf_counter()-load_started
        policy_details["model_load_seconds_this_invocation"] = policy_details.pop("load_seconds")
        current_id = None
        while new_actions < args.max_new_actions and len(rows) < LIMIT:
            step = len(rows)
            need_policy = current_id is None
            if not need_policy:
                previous = rows[-1]
                offset = step-previous["policy_start_step"]
                # Reproject the stored learned chunk on every new sensor state.
                trial = controller.project_chunk(np.asarray(chunks[current_id]["postprocessed"]))
                need_policy = (trial["phase"] != previous["decision"]["phase"]
                               or offset >= previous["decision"]["horizon"])
            if need_policy:
                normalized, postprocessed, elapsed = base.predict(policy, preprocess, postprocess, obs, task.language, step)
                timings["inference"] += elapsed
                current_id = len(chunks)
                decision = controller.project_chunk(postprocessed)
                chunks[current_id] = {
                    "normalized": torch.from_numpy(normalized), "postprocessed": torch.from_numpy(postprocessed),
                    "corrected": torch.from_numpy(decision["corrected_chunk"]),
                    "delta": torch.from_numpy(decision["delta_chunk"]),
                    "inference_seconds": elapsed, "noise_seed": SEED+step,
                }
                policy_start = step
                print(json.dumps({"policy_chunk": current_id, "step": step, "phase": decision["phase"],
                                  "horizon": decision["horizon"], "inference_seconds": elapsed}), flush=True)
            else:
                policy_start = rows[-1]["policy_start_step"]
                decision = trial
            chunk_offset = step-policy_start
            raw = np.asarray(chunks[current_id]["postprocessed"])[chunk_offset].astype(np.float32)
            delta = np.asarray(decision["delta_chunk"])[chunk_offset].astype(np.float32)
            executed = (raw+delta).astype(np.float32)
            decision_public = plain({key: value for key, value in decision.items()
                                     if key not in ("corrected_chunk", "delta_chunk")})
            math_public = compact_assimilation(controller.latest)
            sensor_started = time.perf_counter()
            obs, reward, done, info = env.step(executed.tolist())
            timings["physics"] += time.perf_counter()-sensor_started
            success = bool(env.check_success())
            sim_state = np.asarray(env.get_sim_state()).copy()
            row = {
                "step": step, "policy_chunk_id": current_id, "policy_start_step": policy_start,
                "chunk_offset": chunk_offset, "raw_action": raw, "delta_action": delta,
                "corrected_action_chunk": np.asarray(decision["corrected_chunk"]),
                "delta_action_chunk": np.asarray(decision["delta_chunk"]),
                "executed_action": executed, "delta_action_l2": float(np.linalg.norm(delta)),
                "inference_seconds": float(chunks[current_id]["inference_seconds"]) if chunk_offset == 0 else 0.,
                "reward": float(reward), "done": bool(done), "check_success": success,
                "info": {key: str(value) for key, value in dict(info).items()},
                "state": base.proprio_state(obs), "sim_state": sim_state,
                "decision": decision_public, "math": math_public,
            }
            row.update(observation_hashes(obs, sim_state))
            started = time.perf_counter()
            rows.append(row)
            new_actions += 1
            serializable_chunks = {key: dict(value) for key, value in chunks.items()}
            base.atomic_torch(trajectory_path, tensor_tree({
                "schema": SCHEMA, "seed": SEED, "init_state": args.init_state,
                "metadata": {"settled": settled, "init_sha256": sha(initial),
                             "controller_state": controller.state()},
                "policy_chunks": serializable_chunks,
                "rows": rows,
            }))
            base.atomic_json(output / "report.json", summary(
                rows, args.init_state, task, output, policy_details, timings, runs))
            base.save_frame(output / "latest_agentview.png", base.upright(obs["agentview_image"]))
            base.save_frame(output / "latest_wrist.png", base.upright(obs["robot0_eye_in_hand_image"]))
            timings["serialization"] += time.perf_counter()-started
            print(json.dumps({"step": step+1, "phase": decision["phase"], "horizon": decision["horizon"],
                              "delta_l2": row["delta_action_l2"], "executed": executed.tolist(),
                              "eef": obs["robot0_eef_pos"].tolist(), "aperture": float(
                                  obs["robot0_gripper_qpos"][0]-obs["robot0_gripper_qpos"][1]),
                              "target": np.asarray(decision["target_position"]).tolist(), "stuck": decision["stuck"],
                              "check_success": success}), flush=True)
            started = time.perf_counter()
            controller.assimilate(obs)
            timings["geometry"] += time.perf_counter()-started
            if done or success:
                break
        base.atomic_json(output / "report.json", summary(
            rows, args.init_state, task, output, policy_details, timings, runs))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env.close()
        if not args.verify_only:
            runs.append({"started_unix": time.time(), "wall_seconds": time.perf_counter()-invocation_started,
                         "new_actions": new_actions, "actions_after_run": len(rows),
                         "timings": timings, "error": error})
            base.atomic_json(output / "run_history.json", runs)
            report_path = output / "report.json"
            if report_path.exists():
                report = json.loads(report_path.read_text())
                report["run_count"] = len(runs)
                report["completed_invocation_wall_seconds_sum"] = sum(run["wall_seconds"] for run in runs)
                base.atomic_json(report_path, report)


if __name__ == "__main__":
    main()

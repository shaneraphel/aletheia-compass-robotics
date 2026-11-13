# Copyright (C) 2026 Aletheia contributors
# SPDX-License-Identifier: GPL-3.0-only
#!/usr/bin/env python3
"""Run a resumable real LIBERO episode with the official LeRobot SmolVLA API.

One action is consumed from every freshly predicted 50-action chunk. Every
committed environment step is saved atomically. Resuming reconstructs the
controller and physics by replaying committed actions from the pinned init.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

ROOT = Path(os.environ.get("ALETHEIA_WORKSPACE", Path(__file__).resolve().parents[1]))
CHECKPOINT = Path(os.environ.get("SMOLVLA_CHECKPOINT", ROOT / "models/smolvla_libero"))
PROCESSORS = Path(os.environ.get(
    "SMOLVLA_PROCESSORS", ROOT / "models/SmolVLM2-500M-Video-Instruct-processors"))
INIT_INDEX = int(os.environ.get("LIBERO_INIT_STATE", "0"))
if not 0 <= INIT_INDEX < 50:
    raise ValueError("LIBERO_INIT_STATE must select one of the 50 official initial states")
OUTPUT = Path(os.environ.get(
    "LIBERO_OUTPUT", ROOT / f"results/libero_spatial_task0/smolvla_episode_seed7_init{INIT_INDEX}"))
EXPECTED_TASK = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
EXPECTED_INIT_SHA256 = "4e3eaf7a315bc3bafb6db4837c8eef0e5f8aea717782a2a3ea33139392d966ed"
ACTIVE_INIT_SHA256 = EXPECTED_INIT_SHA256
EXPECTED_WEIGHT_SHA256 = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
SMOLVLA_REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
SEED = 7
WAIT_STEPS = 10
OFFICIAL_SPATIAL_LIMIT = 220
EXTENDED_LIMIT = 280
WAIT_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
SCHEMA = "aletheia-libero-smolvla-resumable-episode-v1"


def sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def upright(image: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(image[::-1, ::-1])
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"Unexpected camera: shape={image.shape}, dtype={image.dtype}")
    return image


def _quat_to_axisangle_robosuite(quat: np.ndarray) -> np.ndarray:
    """Exact official vla-eval convention: [x,y,z,w], no antipodal flip."""
    q = quat.copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (q[:3] * 2.0 * math.acos(q[3]) / den).astype(np.float32)


def proprio_state(obs: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([
        obs["robot0_eef_pos"],
        _quat_to_axisangle_robosuite(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)


def load_initial_states(suite: Any, task_id: int) -> Any:
    """Scope the PyTorch >=2.6 compatibility override to LIBERO init loading."""
    original_load = torch.load

    @functools.wraps(original_load)
    def load_init(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = load_init
    try:
        return suite.get_task_init_states(task_id)
    finally:
        torch.load = original_load


def image_paths(output: Path, step: int) -> tuple[Path, Path]:
    return output / "frames" / f"{step:04d}_agentview.png", output / "frames" / f"{step:04d}_wrist.png"


def save_frame(path: Path, image: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.png")
    iio.imwrite(temporary, image)
    temporary.replace(path)


def torch_row(row: dict[str, Any]) -> dict[str, Any]:
    """Make trajectory tensors self-contained without pickle-only numpy objects."""
    result = dict(row)
    for key in ("raw_normalized_chunk", "postprocessed_chunk", "raw_action", "executed_action",
                "state", "sim_state", "agentview", "wrist"):
        result[key] = torch.from_numpy(np.ascontiguousarray(result[key]))
    return result


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("seed") != SEED or payload.get("init_state") != INIT_INDEX:
        raise RuntimeError("Trajectory schema, seed, or init-state index does not match this pinned episode")
    rows = payload["rows"]
    if len(rows) > EXTENDED_LIMIT:
        raise RuntimeError(f"Checkpoint has {len(rows)} rows, above the fixed {EXTENDED_LIMIT} limit")
    for index, row in enumerate(rows):
        if row["step"] != index:
            raise RuntimeError(f"Non-contiguous trajectory at step {index}")
        if np.asarray(row["executed_action"]).shape != (7,):
            raise RuntimeError(f"Invalid executed action at step {index}")
    return rows


def verify_replay(obs: dict[str, np.ndarray], env: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"replayed_actions": 0, "verified": True}
    last = rows[-1]
    actual = {
        "sim_state_sha256": sha_array(env.get_sim_state()),
        "state_sha256": sha_array(proprio_state(obs)),
        "agentview_raw_sha256": sha_array(obs["agentview_image"]),
        "wrist_raw_sha256": sha_array(obs["robot0_eye_in_hand_image"]),
    }
    expected = {key: last[key] for key in actual}
    if actual != expected:
        raise RuntimeError("Resume replay hash mismatch: " + json.dumps({"expected": expected, "actual": actual}, indent=2))
    # Images fed to the policy are upright; check their persisted bytes as well.
    if sha_array(upright(obs["agentview_image"])) != last["agentview_sha256"] or sha_array(
            upright(obs["robot0_eye_in_hand_image"])) != last["wrist_sha256"]:
        raise RuntimeError("Resume upright-camera hash mismatch")
    return {"replayed_actions": len(rows), "verified": True, "last_step": last["step"], "hashes": actual}


def load_policy() -> tuple[Any, Any, Any, dict[str, Any]]:
    if sha_file(CHECKPOINT / "model.safetensors") != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError("Pinned SmolVLA checkpoint SHA-256 mismatch")
    cfg = PreTrainedConfig.from_pretrained(CHECKPOINT)
    cfg.device = "mps"
    cfg.compile_model = False
    cfg.vlm_model_name = str(PROCESSORS)
    cfg.load_vlm_weights = False
    if cfg.chunk_size != 50 or cfg.n_action_steps != 50:
        raise RuntimeError("Checkpoint action-chunk configuration changed")
    started = time.perf_counter()
    policy = get_policy_class("smolvla").from_pretrained(CHECKPOINT, config=cfg, strict=True)
    policy.eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config, str(CHECKPOINT), preprocessor_overrides={
            "device_processor": {"device": "mps"},
            "tokenizer_processor": {"tokenizer_name": str(PROCESSORS)},
        },
    )
    policy.reset()
    details = {
        "checkpoint": str(CHECKPOINT), "revision": SMOLVLA_REVISION,
        "weights_sha256": EXPECTED_WEIGHT_SHA256, "load_seconds": time.perf_counter() - started,
        "policy_chunk_size": cfg.chunk_size, "policy_n_action_steps": cfg.n_action_steps,
        "actions_consumed_per_inference": 1, "action_queue_used": False,
        "resize_imgs_with_padding": list(cfg.resize_imgs_with_padding),
        "image_mapping": {"observation.images.camera1": "upright agentview_image",
                          "observation.images.camera2": "upright robot0_eye_in_hand_image"},
    }
    return policy, preprocess, postprocess, details


def predict(policy: Any, preprocess: Any, postprocess: Any, obs: dict[str, np.ndarray], task: str,
            step: int) -> tuple[np.ndarray, np.ndarray, float]:
    agentview = upright(obs["agentview_image"])
    wrist = upright(obs["robot0_eye_in_hand_image"])
    frame = {
        "observation.images.camera1": agentview,
        "observation.images.camera2": wrist,
        "observation.state": proprio_state(obs),
    }
    frame = prepare_observation_for_inference(frame, torch.device("mps"), task=task)
    frame = preprocess(frame)
    noise = torch.randn(
        (1, policy.config.chunk_size, policy.config.max_action_dim),
        generator=torch.Generator(device="cpu").manual_seed(SEED + step), dtype=torch.float32,
    ).to("mps")
    torch.mps.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        raw = policy.predict_action_chunk(frame, noise=noise)
        chunk = postprocess(raw)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    # SmolVLA slices its internal max_action_dim=32 chunk to the configured
    # seven LIBERO action dimensions before returning from predict_action_chunk.
    if tuple(raw.shape) != (1, 50, 7) or tuple(chunk.shape) != (1, 50, 7):
        raise RuntimeError(f"Unexpected action shapes: raw={tuple(raw.shape)}, post={tuple(chunk.shape)}")
    normalized = raw.squeeze(0).detach().cpu().numpy().astype(np.float32)
    actions = chunk.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(normalized).all() or not np.isfinite(actions).all():
        raise RuntimeError(f"Non-finite action chunk at step {step}")
    return normalized, actions, elapsed


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in row.items():
        if key in {"agentview", "wrist", "sim_state"}:
            continue
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def make_report(rows: list[dict[str, Any]], task: Any, policy_details: dict[str, Any],
                replay: dict[str, Any], settled_hashes: dict[str, str], new_actions: int) -> dict[str, Any]:
    times = [float(row["inference_seconds"]) for row in rows]
    done_rows = [row for row in rows if row["done"]]
    check_rows = [row for row in rows if row["check_success"]]
    first_done = done_rows[0]["step"] + 1 if done_rows else None
    first_check = check_rows[0]["step"] + 1 if check_rows else None
    finished = bool(done_rows or check_rows or len(rows) >= EXTENDED_LIMIT)
    return {
        "schema": SCHEMA,
        "status": "complete" if finished else "resumable",
        "episodes_attempted": 1 if rows else 0,
        "episodes_completed": int(finished),
        "suite": "libero_spatial", "task_id": 0, "task": task.name, "language": task.language,
        "goal_state": [["on", "akita_black_bowl_1", "plate_1"]],
        "init_state": INIT_INDEX, "init_state_sha256": ACTIVE_INIT_SHA256,
        "seed": SEED, "env_seed": SEED, "wait_steps": WAIT_STEPS,
        "settled_hashes": settled_hashes,
        "libero_commit": LIBERO_COMMIT, "policy": policy_details,
        "protocol": {
            "official_predict_action_chunk": True, "custom_model_forward": False,
            "select_action_used": False, "replan_every_environment_step": True,
            "chunk_shape_postprocessed": [50, 7], "chunk_first_action_only": True,
            "gripper_rule": "raw < 0 -> -1.0; otherwise +1.0",
            "proprio_state": "eef_pos[3] + official_robosuite_axis_angle_no_antipodal[3] + gripper_qpos[2]",
            "fixed_noise_seed_rule": "7 + zero_based_policy_step",
            "official_spatial_action_limit": OFFICIAL_SPATIAL_LIMIT,
            "extended_action_limit": EXTENDED_LIMIT,
        },
        "policy_actions_executed": len(rows), "new_actions_this_run": new_actions,
        "first_done_step": first_done, "first_check_success_step": first_check,
        "success_within_220_by_check_success": first_check is not None and first_check <= OFFICIAL_SPATIAL_LIMIT,
        "success_within_220_by_done": first_done is not None and first_done <= OFFICIAL_SPATIAL_LIMIT,
        "success_within_280_by_check_success": first_check is not None and first_check <= EXTENDED_LIMIT,
        "success_within_280_by_done": first_done is not None and first_done <= EXTENDED_LIMIT,
        "done_check_success_mismatch_steps": [row["step"] + 1 for row in rows if row["done"] != row["check_success"]],
        "inference_seconds_total": sum(times),
        "inference_seconds_mean": statistics.fmean(times) if times else None,
        "inference_seconds_median": statistics.median(times) if times else None,
        "inference_seconds_max": max(times) if times else None,
        "replay_verification": replay,
        "trajectory_file": str(OUTPUT / "trajectory.pt"),
        "single_episode_not_success_rate": True,
        "leaderboard_claim": False, "ready_for_multiagent_training": False,
        "rows": [public_row(row) for row in rows],
    }


def main() -> None:
    global ACTIVE_INIT_SHA256
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-actions", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true", help="Replay the committed episode without inference")
    args = parser.parse_args()
    if not 1 <= args.max_new_actions <= 20:
        raise ValueError("--max-new-actions must be between 1 and 20")
    if not args.verify_only and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for this pinned policy run")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "frames").mkdir(exist_ok=True)
    trajectory_path = OUTPUT / "trajectory.pt"
    rows = load_checkpoint(trajectory_path)
    if not args.verify_only and rows and (rows[-1]["done"] or rows[-1]["check_success"] or len(rows) >= EXTENDED_LIMIT):
        raise RuntimeError("Episode is already complete; refusing to append actions")
    if args.verify_only and not rows:
        raise RuntimeError("No committed episode is available for verification")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    if task.name != EXPECTED_TASK:
        raise RuntimeError(f"Task mismatch: {task.name}")
    initial_states = load_initial_states(suite, 0)
    if len(initial_states) != 50:
        raise RuntimeError(f"Expected 50 official init states, found {len(initial_states)}")
    initial = np.asarray(initial_states[INIT_INDEX])
    ACTIVE_INIT_SHA256 = sha_array(initial)
    if initial.shape != (92,) or (INIT_INDEX == 0 and ACTIVE_INIT_SHA256 != EXPECTED_INIT_SHA256):
        raise RuntimeError("Pinned init-state shape or SHA-256 mismatch")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    new_actions = 0
    policy_details: dict[str, Any] = {}
    replay: dict[str, Any] = {}
    try:
        env.seed(SEED)
        env.reset()
        obs = env.set_init_state(initial_states[INIT_INDEX])
        for _ in range(WAIT_STEPS):
            obs, _, _, _ = env.step(WAIT_ACTION)
        settled_hashes = {
            "sim_state_sha256": sha_array(env.get_sim_state()),
            "state_sha256": sha_array(proprio_state(obs)),
            "agentview_raw_sha256": sha_array(obs["agentview_image"]),
            "wrist_raw_sha256": sha_array(obs["robot0_eye_in_hand_image"]),
        }
        replay = verify_replay(obs, env, [])
        for row in rows:
            replay_obs, replay_reward, replay_done, _ = env.step(np.asarray(row["executed_action"], dtype=np.float32).tolist())
            if bool(replay_done) != row["done"] or bool(env.check_success()) != row["check_success"] or float(replay_reward) != row["reward"]:
                raise RuntimeError(f"Replay outcome mismatch at step {row['step']}")
            obs = replay_obs
        replay = verify_replay(obs, env, rows)
        save_frame(OUTPUT / "latest_agentview.png", upright(obs["agentview_image"]))
        save_frame(OUTPUT / "latest_wrist.png", upright(obs["robot0_eye_in_hand_image"]))
        print(json.dumps({"resume": replay, "settled": settled_hashes}, indent=2), flush=True)
        if args.verify_only:
            for row in rows:
                arrays = {key: np.asarray(row[key]) for key in (
                    "raw_normalized_chunk", "postprocessed_chunk", "raw_action", "executed_action",
                    "state", "sim_state", "agentview", "wrist")}
                if arrays["raw_normalized_chunk"].shape != (50, 7) or arrays["postprocessed_chunk"].shape != (50, 7):
                    raise RuntimeError(f"Stored action-chunk shape mismatch at step {row['step']}")
                if not np.array_equal(arrays["raw_action"], arrays["postprocessed_chunk"][0]):
                    raise RuntimeError(f"First action is not chunk[0] at step {row['step']}")
                if not np.array_equal(arrays["executed_action"][:6], arrays["raw_action"][:6]) or arrays["executed_action"][6] != (-1.0 if arrays["raw_action"][6] < 0 else 1.0):
                    raise RuntimeError(f"Action/gripper integrity mismatch at step {row['step']}")
                for key, hash_key in (("sim_state", "sim_state_sha256"), ("state", "state_sha256"),
                                      ("agentview", "agentview_sha256"), ("wrist", "wrist_sha256")):
                    if sha_array(arrays[key]) != row[hash_key]:
                        raise RuntimeError(f"Stored {key} hash mismatch at step {row['step']}")
                if not all(path.is_file() for path in image_paths(OUTPUT, row["step"])):
                    raise RuntimeError(f"Missing frame PNG at step {row['step']}")
            verification = {
                "schema": "aletheia-libero-episode-independent-replay-v1",
                "verified": True, "replayed_policy_actions": len(rows),
                "recorded_done": bool(rows[-1]["done"]), "replayed_done": bool(replay_done),
                "recorded_check_success": bool(rows[-1]["check_success"]),
                "replayed_check_success": bool(env.check_success()),
                "final_replay": replay, "stored_arrays_and_action_protocol_verified": True,
                "frame_pngs_verified": 2 * len(rows),
                "trajectory_file": str(trajectory_path), "trajectory_bytes": trajectory_path.stat().st_size,
                "trajectory_sha256": sha_file(trajectory_path),
                "single_episode_not_success_rate": True,
            }
            atomic_json(OUTPUT / "independent_replay_verification.json", verification)
            report_path = OUTPUT / "report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                report["final_independent_replay"] = verification
                atomic_json(report_path, report)
            print(json.dumps(verification, indent=2), flush=True)
            return

        policy, preprocess, postprocess, policy_details = load_policy()
        while new_actions < args.max_new_actions and len(rows) < EXTENDED_LIMIT:
            step = len(rows)
            normalized, actions, elapsed = predict(policy, preprocess, postprocess, obs, task.language, step)
            raw_action = actions[0].astype(np.float32)
            executed = raw_action.copy()
            executed[-1] = -1.0 if raw_action[-1] < 0 else 1.0
            obs, reward, done, info = env.step(executed.tolist())
            check_success = bool(env.check_success())
            agentview = upright(obs["agentview_image"])
            wrist = upright(obs["robot0_eye_in_hand_image"])
            sim_state = np.asarray(env.get_sim_state()).copy()
            if not np.isfinite(sim_state).all():
                raise RuntimeError(f"Non-finite simulator state at step {step}")
            row = {
                "step": step, "raw_normalized_chunk": normalized, "postprocessed_chunk": actions,
                "raw_action": raw_action, "executed_action": executed,
                "inference_seconds": elapsed, "noise_seed": SEED + step,
                "reward": float(reward), "done": bool(done), "check_success": check_success,
                "info": {key: str(value) for key, value in dict(info).items()},
                "state": proprio_state(obs), "sim_state": sim_state,
                "agentview": agentview, "wrist": wrist,
                "sim_state_sha256": sha_array(sim_state),
                "state_sha256": sha_array(proprio_state(obs)),
                "agentview_raw_sha256": sha_array(obs["agentview_image"]),
                "wrist_raw_sha256": sha_array(obs["robot0_eye_in_hand_image"]),
                "agentview_sha256": sha_array(agentview), "wrist_sha256": sha_array(wrist),
            }
            agent_path, wrist_path = image_paths(OUTPUT, step)
            save_frame(agent_path, agentview)
            save_frame(wrist_path, wrist)
            rows.append(row)
            new_actions += 1
            # The single atomic trajectory replace is the commit point. If interrupted
            # before it, the next invocation replays only the preceding committed steps.
            atomic_torch(trajectory_path, {
                "schema": SCHEMA, "seed": SEED, "task": task.name, "init_state": INIT_INDEX,
                "rows": [torch_row(item) for item in rows],
            })
            atomic_json(OUTPUT / "report.json", make_report(
                rows, task, policy_details, replay, settled_hashes, new_actions))
            save_frame(OUTPUT / "latest_agentview.png", agentview)
            save_frame(OUTPUT / "latest_wrist.png", wrist)
            print(json.dumps({"step": step + 1, "inference_seconds": elapsed,
                              "raw_action": raw_action.tolist(), "executed_action": executed.tolist(),
                              "reward": float(reward), "done": bool(done), "check_success": check_success,
                              "eef_pos": obs["robot0_eef_pos"].tolist()}), flush=True)
            if done or check_success:
                break
        report = make_report(rows, task, policy_details, replay, settled_hashes, new_actions)
        atomic_json(OUTPUT / "report.json", report)
        print(json.dumps({key: value for key, value in report.items() if key not in {"rows", "policy"}}, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()

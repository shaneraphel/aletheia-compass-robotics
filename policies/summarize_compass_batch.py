# Copyright (C) 2026 Aletheia contributors
# SPDX-License-Identifier: GPL-3.0-only
#!/usr/bin/env python3
"""Verify and summarize a contiguous Compass LIBERO init-state batch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prefix", default="aletheia_grip_site_frame_init")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for init_state in range(args.start, args.stop+1):
        directory = args.root/f"{args.prefix}{init_state}"
        report_path = directory/"report.json"
        if not report_path.exists():
            report_path = directory/"recorded_report.json"
        report = json.loads(report_path.read_text())
        trajectory = directory/"trajectory.pt"
        actual_sha = sha_file(trajectory)
        verification_path = directory/"independent_replay_verification.json"
        verification = json.loads(verification_path.read_text()) if verification_path.exists() else None
        replay_verified = bool(
            verification and verification.get("verified")
            and verification.get("recorded_check_success")
            and verification.get("replayed_check_success")
            and verification.get("all_observation_hashes_verified")
            and verification.get("all_action_delta_identities_verified")
            and verification.get("trajectory_sha256") == actual_sha)
        if report["check_success"]:
            assert report["status"] == "success" and replay_verified
            assert verification["actions"] == report["actions_executed"]
            assert verification["trajectory_bytes"] == trajectory.stat().st_size
        else:
            assert report["status"] == "complete_failed"
            assert report["actions_executed"] == report["official_spatial_action_limit"]
        rows.append({
            "init_state": init_state,
            "status": report["status"],
            "check_success": report["check_success"],
            "first_check_success_step": report["first_check_success_step"],
            "actions_executed": report["actions_executed"],
            "policy_inference_calls": report["policy_inference_calls"],
            "phases": report["phases"],
            "ground_truth_object_positions_used_for_control": report["ground_truth_object_positions_used_for_control"],
            "trajectory_bytes": trajectory.stat().st_size,
            "trajectory_sha256": actual_sha,
            "independent_replay_verified": replay_verified,
            "independent_replay": verification,
        })
    successes = [row for row in rows if row["check_success"]]
    result = {
        "schema": "aletheia-libero-fixed-protocol-batch-v1",
        "init_start": args.start,
        "init_stop": args.stop,
        "episodes": len(rows),
        "successful_episodes": len(successes),
        "failed_episodes": len(rows)-len(successes),
        "independently_verified_successes": sum(row["independent_replay_verified"] for row in rows),
        "all_successes_independently_verified": all(row["independent_replay_verified"] for row in successes),
        "official_spatial_action_limit": 220,
        "single_task_spatial_init_subset_not_leaderboard": True,
        "leaderboard_claim": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
    print(json.dumps([{key: row[key] for key in (
        "init_state", "status", "first_check_success_step", "actions_executed",
        "independent_replay_verified", "trajectory_sha256")} for row in rows], indent=2))


if __name__ == "__main__":
    main()

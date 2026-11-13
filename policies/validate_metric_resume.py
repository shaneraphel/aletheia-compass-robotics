# Copyright (C) 2026 Aletheia contributors
# SPDX-License-Identifier: GPL-3.0-only
#!/usr/bin/env python3
"""Regression checks for bounded Gram routes and committed-entity resume."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from aletheia_metric_control import MetricPath, PineKF, gram_dual_route, quadratic_surface
from compass_pick_place import CompassPickPlace


def dense_dual_route(xc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The original O(n^3) reference, used only on small test matrices."""
    gram = xc@xc.T
    diagonal = np.diag(gram)
    distances = np.maximum(diagonal[:, None]+diagonal[None, :]-2*gram, 0.0)
    centering = np.eye(len(xc))-np.ones((len(xc), len(xc)))/len(xc)
    return gram, -0.5*centering@distances@centering


def check_gram_routes() -> dict[str, float | int]:
    rng = np.random.default_rng(20261114)
    maximum_blocked_vs_dense = 0.0
    maximum_identity_error = 0.0
    cases = 0
    for n in (6, 7, 31, 128):
        for _ in range(4):
            points = rng.normal(size=(n, 3))
            xc = points-points.mean(axis=0)
            coordinate, dense_dual = dense_dual_route(xc)
            norms = np.einsum("ij,ij->i", xc, xc)
            coordinate_sum = xc.sum(axis=0)
            row_mean = norms+norms.mean()-2*(xc@coordinate_sum)/n
            grand_mean = 2*norms.mean()-2*float(coordinate_sum@coordinate_sum)/(n*n)
            # Independently reproduce the blocked formula and compare every entry
            # with the old dense J@D@J entity, not merely a scalar loss.
            distances = np.maximum(norms[:, None]+norms[None, :]-2*coordinate, 0.0)
            blocked_dual = -0.5*(distances-row_mean[:, None]-row_mean[None, :]+grand_mean)
            difference = float(np.abs(blocked_dual-dense_dual).max())
            maximum_blocked_vs_dense = max(maximum_blocked_vs_dense, difference)
            np.testing.assert_allclose(blocked_dual, dense_dual, rtol=0, atol=2e-12)
            check = gram_dual_route(xc, block_size=5)
            assert check["gram_dual_route_finite"]
            assert check["gram_dual_route_blocks"] == (n+4)//5
            maximum_identity_error = max(maximum_identity_error, check["gram_dual_route_max_abs"])
            assert check["gram_dual_route_max_abs"] < 2e-12
            chart = quadratic_surface(points)
            assert chart["valid"] and chart["gram_dual_route_finite"]
            assert "gram_coordinate_route" not in chart and "gram_distance_route" not in chart
            cases += 1
    return {
        "small_random_cases": cases,
        "blocked_vs_dense_max_abs": maximum_blocked_vs_dense,
        "gram_dual_route_max_abs": maximum_identity_error,
    }


def check_large_surface() -> dict[str, float | int]:
    # A 9000-point curved metric graph exercises the same scale as the late
    # init-3 plate cloud without allocating several 9000-by-9000 arrays.
    x, y = np.meshgrid(np.linspace(-0.08, 0.08, 90), np.linspace(-0.06, 0.06, 100))
    z = 0.7*x*x-0.4*y*y+0.2*x*y
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    started = time.perf_counter()
    chart = quadratic_surface(points)
    elapsed = time.perf_counter()-started
    assert chart["valid"] and chart["count"] == 9000
    assert chart["gram_dual_route_finite"] and chart["gram_dual_route_max_abs"] < 2e-12
    return {
        "points": 9000,
        "seconds": elapsed,
        "blocks": chart["gram_dual_route_blocks"],
        "gram_dual_route_max_abs": chart["gram_dual_route_max_abs"],
    }


def check_committed_resume(trajectory_path: Path) -> dict[str, object]:
    payload = torch.load(trajectory_path, map_location="cpu", weights_only=True)
    rows = payload["rows"]
    if not rows:
        raise ValueError("Committed trajectory is empty")
    initial = np.asarray(payload["metadata"]["controller_state"]["orientation_anchor"], dtype=np.float64)
    names = tuple(rows[-1]["math"]["targets"])
    checkpoints = sorted({0, 1, 2, 15, 16, 17, len(rows)//2, len(rows)-2, len(rows)-1})
    maximum_filter_error = 0.0
    maximum_path_entity_error = 0.0
    checked = 0
    for k in checkpoints:
        if not 0 <= k < len(rows):
            continue
        controller = object.__new__(CompassPickPlace)
        controller.names = names
        controller.filters = {name: PineKF(0.05) for name in names}
        controller.path = MetricPath()
        controller.orientation_anchor = initial.copy()
        stored_path = rows[k]["math"]["path"]
        if not stored_path.get("entity") or "observations" not in stored_path["entity"]:
            continue  # The runner deliberately falls back to full assimilation here.
        restore = controller.restore_episode_evidence(rows[:k+1])
        expected_positions = np.asarray(stored_path["entity"]["observations"], dtype=np.float64)
        assert len(controller.path.eef) == len(expected_positions)
        for actual, expected in zip(controller.path.eef, expected_positions):
            np.testing.assert_array_equal(actual, expected)
        for name in names:
            source_step = restore["pinekf_source_steps"][name]
            updates = rows[source_step]["math"]["targets"][name]["pinekf"]
            evidence = [update for update in updates if update.get("valid")][-1]
            maximum_filter_error = max(
                maximum_filter_error,
                float(np.max(np.abs(controller.filters[name].x-np.asarray(evidence["state"])))),
                float(np.max(np.abs(controller.filters[name].P-np.asarray(evidence["covariance"])))),
                float(np.max(np.abs(controller.filters[name].previous_innovation
                                    -np.asarray(evidence["innovation"])))),
            )
            np.testing.assert_array_equal(controller.filters[name].x, np.asarray(evidence["state"]))
            np.testing.assert_array_equal(controller.filters[name].P, np.asarray(evidence["covariance"]))
        assert stored_path["count"] == len(controller.path.eef)
        if len(controller.path.eef) >= 3:
            entity = controller.path._load_metric()(torch.from_numpy(np.asarray(controller.path.eef)), None)
            for key in ("space_center", "space_coordinates_Z", "gram_coordinate_route",
                        "gram_distance_route", "laplacian", "laplacian_eigenvalues",
                        "spectral_eta", "compass_roundtrip_max_abs"):
                actual = np.asarray(entity[key])
                expected = np.asarray(stored_path["entity"][key])
                maximum_path_entity_error = max(
                    maximum_path_entity_error, float(np.max(np.abs(actual-expected))))
                np.testing.assert_array_equal(actual, expected)
        checked += 1
    return {
        "trajectory": str(trajectory_path),
        "committed_rows": len(rows),
        "resume_checkpoints": checked,
        "pinekf_restore_max_abs": maximum_filter_error,
        "metric_path_entity_max_abs": maximum_path_entity_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    args = parser.parse_args()
    result = {
        "passed": True,
        "gram_routes": check_gram_routes(),
        "large_surface": check_large_surface(),
        "committed_resume": check_committed_resume(args.trajectory),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

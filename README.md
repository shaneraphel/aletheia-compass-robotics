# Aletheia Compass Robotics

Evidence-first geometric control for LIBERO, released under **GPL-3.0-only**. This repository contains mathematical control strategies, their recorded trajectories, and executable replay evidence. It does **not** contain the Aletheia mother model, private weights, DeepSeek weights, original SmolVLA weights, or a trained derivative small model.

## Measured results

All runs below use `libero_spatial` task 0 and seed 7, with the official 220-action limit.

### Preserved init-2 release

On official initial state 2:

- Original SmolVLA chunk-4 baseline: 220 actions, `check_success() = false`; independent replay also false, with observation hashes matched.
- Compass/PineKF geometric control: **87 actions**, `check_success() = true` at action 87; strict independent replay passed every check.
- Trajectory SHA-256: `d4aa0c7edb483839196bcaad238ec76f9aead5cd5f38d5f2804946673f9a7e9b`.

Init 2 is preserved as its own earlier release. It is **not pooled with init 3–19** to claim a result for one unchanged controller revision.

### Fixed-protocol init 3–19 batch

The subsequent controller revision was evaluated on the 17 explicitly selected initial states 3 through 19:

- **15 recorded successes**, at steps 83–207 inclusive.
- **All 15 successful trajectories passed strict independent replay**, including both RGB cameras, depth, instance segmentation, simulation state, action lineage, and final success.
- **2 failures retained without omission**: init 3 and init 16 each exhausted all 220 actions with `check_success() = false`.

```text
init  4: success at  90     init 12: success at  87
init  5: success at 166     init 13: success at 207
init  6: success at  91     init 14: success at  93
init  7: success at  88     init 15: success at  83
init  8: success at  91     init 17: success at  96
init  9: success at 109     init 18: success at  90
init 10: success at 111     init 19: success at  90
init 11: success at  96

init  3: complete_failed after 220 actions
init 16: complete_failed after 220 actions
```

This is a **15/17 result on one specified subset of one LIBERO Spatial task**. It is not the full 50-state evaluation, a leaderboard score, a generalization claim, or evidence of weight-training improvement. Init 3 has diagnostic resume evidence with three agent-view-only RGB differences at zero-based steps 128, 129, and 132; physics, other sensors, and the failed result matched. It is not presented as a strictly replay-verified success.

The preserved phase sequence is `orient → approach → descend → close → lift → transport → place`, with recovery branches visible in some trajectories. Every action in the 15 successful episodes has a nonzero mathematical correction. Ground-truth simulator object positions were not used for control.

## Mathematics that actually participated in control

- Dual-camera depth and instance segmentation produce metric point clouds; no simulator object-position truth is read for control.
- Point-cloud Gram duality, first/second fundamental forms, shape operator, principal curvatures, Gaussian curvature, and parallel-surface foci determine grasp geometry and descent bounds.
- Surface charts use a right-handed SO(3) convention. If the arbitrary SVD sign produces determinant `-1`, the normal is flipped before downstream orientation construction; no downstream geometric check is relaxed.
- The Gram dual-route identity is checked with algebraically obtained distance moments and bounded coordinate/distance blocks. It no longer stores two full `N×N` matrices or performs the dense `J @ D @ J` product. The pairwise-distance verification itself is retained.
- PineKF maintains explicit metric state, covariance, innovation, and Joseph-update entities.
- A Laplacian spectral path exposes trapped/decay/oscillatory bands; uncertainty-gated ultrametric and skew-exponential exits can alter free-space residuals when trapped.
- Panda `grip_site` geometry uses `approach = eef - fingers`; absolute orientation is `D = Q @ R`. The grasp/place map preserves the rigid bowl-center identity.
- Every raw action chunk is combined with a synthesized `delta_chunk`; the environment receives the float32 value of `raw + delta`.
- Resume restores the exact committed PineKF and original float64 `MetricPath` entities, then assimilates the current observation once. Replaying physics is still required; float32 `state[:3]` is not substituted for the original path observations.

![Real-time mathematical entities](figures/inloop_mathematics.png)

These quantities were recorded from the same decisions that generated the actions, not added as post-hoc trajectory decorations.

## Evidence map

- `evidence/init2/trajectory.pt`: preserved 87-step init-2 lineage and observation hashes.
- `evidence/init2/independent_replay_verification.json`: strict init-2 replay receipt.
- `evidence/init3_19/init{3..19}/trajectory.pt`: all 17 complete trajectories, including both failures.
- `evidence/init3_19/init*/recorded_report.json`: sanitized recorded episode reports.
- `evidence/init3_19/init*/independent_replay_verification.json`: strict receipts for the 15 successful trajectories.
- `evidence/init3_19/init*/*png`: recorded and replayed final camera frames where available.
- `evidence/init3_19/batch_summary.json`: independently recomputed trajectory hashes, per-episode outcomes, and replay receipts.
- `evidence/metric_resume_validation.json`: blocked-Gram and exact committed-entity restoration regression results.
- `figures/inloop_mathematics.pt`: tensor-valued in-loop mathematical entities from the preserved init-2 episode.
- `evidence/baseline_init2/reference_receipt.json`: concise failed-baseline reference; its 88 MB trajectory and original weights are intentionally not redistributed.
- `SHA256SUMS.json`: complete public-file hashes and explicit claim boundaries.

For strict replay, raw-action lineage, `raw + delta`, simulation state, proprioception, both RGB cameras, both depth maps, both instance-segmentation maps, `done`, and `check_success()` are checked at every step. The float32 round-trip residual is bounded elementwise by `eps*(abs(raw)+abs(delta))`; exact values are recorded in each receipt.

## Independent replay from this repository

Install LIBERO at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, plus the dependencies recorded in `environment.json`. The official LIBERO `SegmentationRenderEnv` is used without a local source patch. No model download is needed for replay because policy inference and the mathematical controller are disabled.

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO
git -C /tmp/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
export PYTHONPATH="$PWD/policies:/tmp/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export ALETHEIA_WORKSPACE="$PWD"
export LIBERO_OUTPUT="$PWD/evidence/init3_19/init4"
python -u policies/libero_compass_episode.py --init-state 4 --verify-only
```

The command rewrites the init-4 replay receipt with a new timing measurement. All action and observation hashes must remain identical. The complete checked-in receipt set is covered by `SHA256SUMS.json`.

Recompute the batch summary without running new policy actions:

```bash
python policies/summarize_compass_batch.py \
  --root evidence/init3_19 --prefix init --start 3 --stop 19 \
  --output evidence/init3_19/batch_summary.json
```

Run the lightweight geometry, Gram, spectral-entity, and exact-resume checks:

```bash
PYTHONPATH="$PWD/policies" python policies/validate_compass_release.py
PYTHONPATH="$PWD/policies" python policies/validate_metric_resume.py \
  evidence/init3_19/init4/trajectory.pt
```

For a new live episode, separately obtain the external SmolVLA checkpoint and processor files, then set `SMOLVLA_CHECKPOINT` and `SMOLVLA_PROCESSORS`. Their licenses remain with their respective upstream projects. No external weights are vendored here.

## Attribution and limits

The final absolute-kinematics and grasp-frame repair is explicitly attributed to `kimi-supervisor`. Four preserved Aletheia official-MLX drafts failed executable acceptance; they were not overwritten or represented as passing. This package therefore demonstrates a working geometric controller and reproducible physical successes, **not autonomous Aletheia code generation**.

No force sensor was available; no force closure, dexterous-hand result, world-model leaderboard position, or weight-learning improvement is claimed. Commercial licensing for new contributions can be discussed separately without removing the rights already granted by GPLv3. Private Aletheia assets are outside this release and are not licensed by it.

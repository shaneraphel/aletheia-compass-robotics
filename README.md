# Aletheia Compass Robotics

Evidence-first geometric control for LIBERO, released under **GPL-3.0-only**. This repository contains a small mathematical control strategy and its complete successful replay evidence. It does **not** contain the Aletheia mother model, private weights, DeepSeek weights, or original SmolVLA weights.

## Verified result

On `libero_spatial` task 0, official initial state 2, seed 7:

- Original SmolVLA chunk-4 baseline: 220 actions, `check_success() = false`; independent replay also false, with observation hashes matched.
- Compass/PineKF geometric control: **87 actions**, **`check_success() = true` at action 87**; strict independent replay passed every check.

This is **one successful initial state, not a success rate, leaderboard claim, generalization result, trained small model, or derivative-weight release**. The preserved phase sequence is `orient → approach → descend → close → lift → transport → place`. All 87 executed actions have nonzero mathematical corrections.

The independent replay executes the recorded actions without loading a policy or the mathematical controller. It verifies raw-action lineage, the `raw + delta` identity, simulation state, proprioception, both RGB cameras, both depth maps, both instance-segmentation maps, `done`, and `check_success()` at every step. Maximum nominal-corrected float32 round-trip residual is `1.1920928955078125e-07`, bounded elementwise by `eps*(abs(raw)+abs(delta))`; the largest observed bound was `3.65080722986022e-07`.

## Mathematics that actually participated in control

- Dual-camera depth and instance segmentation produce metric point clouds; no simulator object-position truth is read for control.
- Point-cloud Gram duality, first/second fundamental forms, shape operator, principal curvatures, Gaussian curvature and parallel-surface foci determine grasp geometry and descent bounds.
- PineKF maintains explicit metric state, covariance, innovation and Joseph-update entities.
- A Laplacian spectral path exposes trapped/decay/oscillatory bands; uncertainty-gated ultrametric and skew-exponential exits can alter free-space residuals when trapped.
- Panda `grip_site` geometry uses `approach = eef - fingers`; absolute orientation is `D = Q @ R`. The grasp/place map preserves the rigid bowl center identity.
- Every raw action chunk is combined with a synthesized `delta_chunk`; the environment receives the float32 value of `raw + delta`.

![Real-time mathematical entities](figures/inloop_mathematics.png)

These are the quantities recorded from the same decisions that generated the actions, not post-hoc trajectory decorations.

## Evidence map

- `evidence/init2/trajectory.pt`: complete 87-step lineage and observation hashes.
- `evidence/init2/recorded_report.json`: recorded episode report.
- `evidence/init2/independent_replay_verification.json`: strict independent replay receipt.
- `evidence/init2/fast_run.log` and `strict_replay.log`: sanitized execution logs.
- `evidence/init2/*png`: recorded and replayed final camera frames.
- `figures/inloop_mathematics.pt`: tensor-valued in-loop mathematical entities.
- `evidence/baseline_init2/reference_receipt.json`: concise failed-baseline reference; its 88 MB trajectory and original weights are intentionally not redistributed.
- `SHA256SUMS.json`: file hashes and explicit claim boundaries.

## Independent replay

Install LIBERO at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`, plus the dependencies recorded in `environment.json`. The official LIBERO `SegmentationRenderEnv` is used without a local source patch. No model download is needed for replay because policy inference is disabled.

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /tmp/LIBERO
git -C /tmp/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
export PYTHONPATH="$PWD/policies:/tmp/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export ALETHEIA_WORKSPACE="$PWD"
export LIBERO_OUTPUT="$PWD/evidence/init2"
python -u policies/libero_compass_episode.py --init-state 2 --verify-only
```

The command rewrites the replay receipt with a new timing measurement. All action and observation hashes must remain identical. The preserved source receipt is checksummed in `SHA256SUMS.json`.

Run the lightweight geometry and spectral-entity checks:

```bash
PYTHONPATH="$PWD/policies" python policies/validate_compass_release.py
```

For a new live episode, separately obtain the external SmolVLA checkpoint and processor files, then set `SMOLVLA_CHECKPOINT` and `SMOLVLA_PROCESSORS`. Their licenses remain with their respective upstream projects. No external weights are vendored here.

## Attribution and limits

The final absolute-kinematics and grasp-frame repair is explicitly attributed to `kimi-supervisor`. Four preserved Aletheia official-MLX drafts failed executable acceptance; they were not overwritten or represented as passing. This package therefore demonstrates a working geometric controller and reproducible physical success, **not autonomous Aletheia code generation**.

No force sensor was available; no force closure, dexterous-hand result, world-model leaderboard position, or weight-learning improvement is claimed. Commercial licensing for new contributions can be discussed separately without removing the rights already granted by GPLv3. Private Aletheia assets are outside this release and are not licensed by it.

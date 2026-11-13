# Third-party boundaries

- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- SmolVLA policy: `lerobot/smolvla_libero`, revision `a24f1f380461b665158d52acd140152c79e8b36d`, referenced by the harness but **not included**.
- SmolVLM2 processor artifacts: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, referenced but **not included**.
- NumPy, SciPy, PyTorch, LeRobot, robosuite, MuJoCo and their transitive dependencies retain their own licenses.

GPL-3.0-only applies to this repository's original controller adaptations and released mathematical-operator code. It does not relicense external simulators, model weights, datasets, or processor artifacts. The recorded trajectory contains actions, mathematical measurements and observation hashes, not model weights.

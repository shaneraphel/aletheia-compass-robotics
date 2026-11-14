# Third-party boundaries

- LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO at commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- SmolVLA policy: `lerobot/smolvla_libero`, revision `a24f1f380461b665158d52acd140152c79e8b36d`, referenced by the harness but **not included**.
- SmolVLM2 processor artifacts: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, referenced but **not included**.
- NumPy, SciPy, PyTorch, LeRobot, robosuite, MuJoCo and their transitive dependencies retain their own licenses.

GPL-3.0-only applies to this repository's original controller adaptations and released mathematical-operator code. It does not relicense external simulators, model weights, datasets, or processor artifacts. The recorded trajectory contains actions, mathematical measurements and observation hashes, not model weights.

- The separate Gram coding experiment uses `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` as its base-model lineage. The base model is not vendored and must be obtained separately.
- The locally retrieved base-model license is MIT; its notice is preserved at `evidence/gram_coding/DEEPSEEK_LICENSE.txt`.
- `evidence/gram_coding/down_proj_delta.pt` contains only a new FP32 total increment for layer 26, not original DeepSeek tensors. It is a public derivative experiment artifact, not the Aletheia mother model.
- PyTorch, Transformers, tokenizers, NumPy, and the base model retain their respective upstream licenses and terms.

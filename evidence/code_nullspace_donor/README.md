# Coding-nullspace donor composition: verified evidence

This is a **single-task coding composition result**, not a leaderboard claim. A nonzero donor-readout operator was composed with the already verified layer-26 weight realization while preserving its successful free-running code generation.

## Mathematical constraint

Let \(Z_{\mathrm{fit}}\) be the two-donor feature coordinates on the language fit states, \(Z_{\mathrm{code}}\) the same coordinates at every cached causal state of the successful coding trajectory, and \(\Gamma\) the measured remaining language cotangent. The readout solves

\[
\min_{\Delta W}\|\Delta W\|_F,\qquad
Z_{\mathrm{fit}}\Delta W^\top=\Gamma,\qquad
Z_{\mathrm{code}}\Delta W^\top=0.
\]

One numerical-rank dual solve gives

\[
\Delta W=
\begin{bmatrix}\Gamma\\0\end{bmatrix}^{\top}
\left(
\begin{bmatrix}Z_{\mathrm{fit}}\\Z_{\mathrm{code}}\end{bmatrix}
\begin{bmatrix}Z_{\mathrm{fit}}\\Z_{\mathrm{code}}\end{bmatrix}^{\top}
\right)^\dagger
\begin{bmatrix}Z_{\mathrm{fit}}\\Z_{\mathrm{code}}\end{bmatrix}.
\]

The code coordinates are actual layer-27 states from greedy cached generation, collected with zero readouts that do not modify the forward result. This is a geometric nullspace constraint, not a token rule, answer lookup table, candidate ranking procedure, or post-generation code edit.

## Independently measured result

- The source operator and generated code passed all local gate checks.
- Joint solve: **1985 constraints**, numerical rank **1985**.
- Language fit residual, relative Frobenius norm: **5.950311484273215e-07**.
- Coding nullspace residual, maximum absolute entry: **2.7082213892044155e-08**.
- A separate process regenerated **617 tokens**, ending at EOS under a **1024-token** allowance. The complete token-ID sequence is identical to the verified deep-base trajectory.
- The untouched generated source passed the unchanged tests: **16 random/repeated-point block cases** and all invalid-input contracts.
- Maximum dense-or-blocked identity error: **1.5987211554602254e-14**.
- The original base-head delta was not applied. All original weight hashes were restored after the experiment.

## Verify and regenerate

The exact generated source is [`../gram_coding/candidate.py`](../gram_coding/candidate.py). From this directory, verify its identity and execute the original tests with Python and NumPy:

```bash
python verify.py
```

To regenerate those same tokens from the public layer-26 increment and separately obtained DeepSeek-R1-Distill-Qwen-1.5B, use the existing command in [`../gram_coding/README.md`](../gram_coding/README.md#regenerate-from-weights). The additional constrained donor readout and its private source tensors are not distributed in this package; the zero-correction constraint preserves this released coding trajectory.

`summary.json` records the independent measurements and source hashes. `test_receipt.json` is the actual passing test receipt. The Aletheia mother engine and private model weights remain local. Original public contributions use GPL-3.0-only; external model lineage retains its upstream license.

#!/usr/bin/env python3
"""Verify the frozen candidate or regenerate it with the official Transformers trunk."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_LAZY_MODEL_ALLOWLIST", "auto,qwen2")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_source(raw: str) -> str:
    """Extract the generated function; never score or repair candidates."""
    text = "```python\n" + raw
    closed = re.findall(r"```(?:python|py)?\s*(.*?)```", text,
                        flags=re.DOTALL | re.IGNORECASE)
    for source in closed + [text[len("```python\n"):]]:
        source = source.strip() + "\n"
        try:
            compile(source, "<model-generated>", "exec")
        except SyntaxError:
            continue
        if "def blocked_gram_dual_route" in source:
            return source
    return text[len("```python\n"):].strip() + "\n"


def run_official_tests(candidate: Path) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(ROOT / "official_tests.py"), str(candidate.resolve())],
        cwd=ROOT, text=True, capture_output=True, timeout=120)
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        parsed = {"passed": False, "raw_stdout": completed.stdout}
    return {
        "passed": completed.returncode == 0 and parsed.get("passed") is True,
        "exit_code": completed.returncode,
        "parsed": parsed,
        "stderr": completed.stderr,
        "wall_seconds": time.perf_counter() - started,
        "candidate_sha256": sha256_bytes(candidate.read_bytes()),
        "official_tests_sha256": sha256_bytes((ROOT / "official_tests.py").read_bytes()),
        "supervisor_code_written": False,
    }


def repeated_suffix(ids: list[int], repeats: int = 4, max_period: int = 192) -> dict | None:
    for period in range(4, min(max_period, len(ids) // repeats) + 1):
        chunk = ids[-period:]
        if all(ids[-(k + 1) * period:-k * period if k else None] == chunk
               for k in range(1, repeats)):
            return {"period_tokens": period, "complete_repeats": repeats,
                    "period_token_ids": chunk}
    return None


def regenerate(args: argparse.Namespace) -> int:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class FP32Readout(torch.nn.Module):
        def __init__(self, weight: torch.Tensor, output_dtype: torch.dtype | None = None):
            super().__init__()
            self.weight = torch.nn.Parameter(weight.detach().float(), requires_grad=False)
            self.output_dtype = output_dtype

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            result = F.linear(hidden.float(), self.weight, None)
            return result if self.output_dtype is None else result.to(self.output_dtype)

    if not torch.backends.mps.is_available():
        raise RuntimeError("This recorded reproduction requires an Apple MPS device; revalidate other devices separately.")
    if args.max_new_tokens < 1024:
        raise ValueError("Use at least the recorded 1024-token generation budget.")
    args.out.mkdir(parents=True, exist_ok=True)
    prompt = (ROOT / "prompt.txt").read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), dtype=torch.bfloat16).to("mps").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    artifact = torch.load(args.delta, map_location="cpu", weights_only=True)
    if artifact["schema"] != "aletheia-public-down-proj-delta-v1":
        raise RuntimeError("Unexpected public delta schema")
    layer_index = int(artifact["layer_index"])
    layer = model.model.layers[layer_index]
    original_down = layer.mlp.down_proj
    original_head = model.lm_head
    original_down_hash = sha256_bytes(
        original_down.weight.detach().cpu().contiguous().view(torch.uint8).numpy())
    if original_down_hash != artifact["original_down_proj_sha256"]:
        raise RuntimeError("The base model does not match the recorded down-projection tensor")
    updated = original_down.weight.detach().float() + artifact["down_proj_delta_fp32"].float().to("mps")
    layer.mlp.down_proj = FP32Readout(updated, output_dtype=torch.bfloat16)
    model.lm_head = FP32Readout(original_head.weight.detach())
    ids: list[int] = []
    texts: list[str] = []
    hidden_rows = []
    periodic = None
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            current = torch.tensor([prompt_ids], device="mps")
            past = None
            for _ in range(args.max_new_tokens):
                result = model(current, past_key_values=past, use_cache=True,
                               output_hidden_states=True, return_dict=True)
                past = result.past_key_values
                logits = result.logits[0, -1].float().cpu()
                token_id = int(torch.nonzero(logits == logits.max(), as_tuple=False).flatten().min())
                ids.append(token_id)
                texts.append(tokenizer.decode([token_id], skip_special_tokens=False))
                hidden_rows.append(result.hidden_states[-1][0, -1].detach().float().cpu())
                current = torch.tensor([[token_id]], device="mps")
                periodic = repeated_suffix(ids)
                if periodic is not None or token_id == tokenizer.eos_token_id:
                    break
        elapsed = time.perf_counter() - started
        raw = tokenizer.decode(ids, skip_special_tokens=True)
        source = extract_source(raw)
        candidate = args.out / "candidate.py"
        candidate.write_text(source, encoding="utf-8")
        (args.out / "raw_output.txt").write_text(raw, encoding="utf-8")
        receipt = run_official_tests(candidate)
        hidden = torch.stack(hidden_rows)
        centered = hidden - hidden.mean(dim=0, keepdim=True)
        torch.save({
            "schema": "aletheia-metric-vector-space-v1",
            "coordinate_field": "R",
            "observations": hidden,
            "generated_token_ids": torch.tensor(ids, dtype=torch.long),
            "prompt_tokens": torch.tensor(len(prompt_ids)),
            "centered_coordinate_norms": centered.square().sum(dim=-1).sqrt(),
            "entity_source": "official cached BF16 trunk; FP32 layer-26 down projection and linear readout",
        }, args.out / "metric_vector_space.pt")
        report = {
            "schema": "aletheia-public-gram-reproduction-v1",
            "generated_tokens": len(ids),
            "generated_token_ids": ids,
            "generated_token_text": texts,
            "max_new_tokens": args.max_new_tokens,
            "generation_seconds": elapsed,
            "stopped_on_exact_period": periodic,
            "test_receipt": receipt,
            "matches_frozen_candidate": receipt["candidate_sha256"] == sha256_bytes((ROOT / "candidate.py").read_bytes()),
            "claim_boundary": "Exact candidate reproduction on the recorded software and MPS environment; other environments require fresh validation.",
        }
        (args.out / "reproduction.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.out / "test_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in (
            "generated_tokens", "matches_frozen_candidate")}, indent=2))
        print(json.dumps(receipt, indent=2))
        return 0 if receipt["passed"] and report["matches_frozen_candidate"] else 1
    finally:
        layer.mlp.down_proj = original_down
        model.lm_head = original_head


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Regenerate instead of only testing the frozen candidate")
    parser.add_argument("--model", type=Path, help="Local DeepSeek-R1-Distill-Qwen-1.5B directory")
    parser.add_argument("--delta", type=Path, default=ROOT / "down_proj_delta.pt")
    parser.add_argument("--out", type=Path, default=Path("gram_coding_reproduction"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if not args.generate:
        receipt = run_official_tests(ROOT / "candidate.py")
        print(json.dumps(receipt, indent=2))
        return 0 if receipt["passed"] else 1
    if args.model is None:
        parser.error("--model is required with --generate")
    return regenerate(args)


if __name__ == "__main__":
    raise SystemExit(main())

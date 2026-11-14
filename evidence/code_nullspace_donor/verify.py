#!/usr/bin/env python3
"""Verify the frozen model-generated source against the unchanged tests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
GRAM = HERE.parent / "gram_coding"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    summary = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
    candidate = (GRAM / "candidate.py").resolve()
    tests = (GRAM / "official_tests.py").resolve()
    if sha256(candidate) != summary["candidate_sha256"]:
        raise RuntimeError("Generated source hash mismatch")
    if sha256(tests) != summary["official_tests_sha256"]:
        raise RuntimeError("Unchanged test hash mismatch")
    completed = subprocess.run([sys.executable, str(tests), str(candidate)], cwd=GRAM,
                               text=True, capture_output=True, timeout=120)
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError:
        parsed = {"passed": False, "raw_stdout": completed.stdout}
    receipt = {"passed": completed.returncode == 0 and parsed.get("passed") is True,
               "exit_code": completed.returncode, "parsed": parsed,
               "candidate_sha256": sha256(candidate), "official_tests_sha256": sha256(tests),
               "candidate_modified": False, "tests_modified": False}
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

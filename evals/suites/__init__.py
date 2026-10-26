"""Eval suites. Each module has run(runs: int) -> dict with:
- "metrics": {name: float}  the scorecard numbers (rates are 0-1)
- "cases":   [dict]         one record per case (and run), saved in the results JSON
- "failures": [str]         short human-readable lines for the failed cases
"""

import json
from pathlib import Path

DATASETS = Path(__file__).resolve().parents[1] / "datasets"


def load_jsonl(name: str) -> list[dict]:
    lines = (DATASETS / name).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip() and not line.startswith("//")]

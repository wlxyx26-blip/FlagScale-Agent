from __future__ import annotations

import json
from pathlib import Path

from rewardkit import criterion

SCORES = Path("/logs/verifier/programmatic-scores.json")


@criterion
def assistant_quality(workspace: Path) -> float:
    del workspace
    try:
        value = json.loads(SCORES.read_text(encoding="utf-8")).get(
            "assistant_quality", 0.0
        )
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0

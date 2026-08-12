#!/usr/bin/env bash
set -uo pipefail

mkdir -p /logs/verifier /logs/artifacts /logs/verifier/rewardkit

TASK_PYTHON="/opt/imdb-bert-env/bin/python"
PROGRAMMATIC="/logs/verifier/programmatic-scores.json"
RK_OUTPUT="/logs/verifier/rewardkit/reward.json"
FINAL_REWARD="/logs/verifier/reward.json"
REPORT="/logs/artifacts/evaluation_report.json"


write_zero_programmatic() {
  local reason="$1"

  printf '%s\n' \
    '{"correct": 0.0, "assistant_quality": 0.0}' \
    > "$PROGRAMMATIC"

  printf '%s\n' "$reason" \
    > /logs/verifier/verifier-error.txt
}


# ============================================================
# 1. Deterministic verifier
# ============================================================

if [ ! -x "$TASK_PYTHON" ]; then
  write_zero_programmatic \
    "Missing training environment Python: $TASK_PYTHON"
else
  set +e

  "$TASK_PYTHON" /tests/test_outputs.py \
    > >(tee /logs/verifier/test-stdout.txt) \
    2> >(tee /logs/verifier/test-stderr.txt >&2)

  verifier_status=$?

  set -e

  if [ "$verifier_status" -ne 0 ] || \
     [ ! -s "$PROGRAMMATIC" ]; then

    write_zero_programmatic \
      "Verifier failed with exit code $verifier_status"
  fi
fi


# ============================================================
# 2. RewardKit
#
# Important:
# mcs-5 is a gateway alias that LiteLLM does not know natively.
# rewardkit_runner.py registers the alias before RewardKit runs.
# ============================================================

rewardkit_status=127

# Avoid accidentally consuming a stale RewardKit result.
rm -f \
  "$RK_OUTPUT" \
  /logs/verifier/rewardkit/reward-details.json \
  /logs/verifier/rewardkit/exit-code.txt

if command -v uv >/dev/null 2>&1; then
  set +e

  LITELLM_LOCAL_MODEL_COST_MAP=True \
  uv run \
    --no-project \
    --with "harbor-rewardkit==0.1.7" \
    python /tests/rewardkit_runner.py \
      /tests \
      --workspace /app \
      --output "$RK_OUTPUT" \
    > >(tee /logs/verifier/rewardkit/stdout.txt) \
    2> >(tee /logs/verifier/rewardkit/stderr.txt >&2)

  rewardkit_status=$?

  set -e
else
  echo "uv is unavailable; cannot launch RewardKit wrapper" \
    > /logs/verifier/rewardkit/stderr.txt
fi

printf '%s\n' "$rewardkit_status" \
  > /logs/verifier/rewardkit/exit-code.txt


# ============================================================
# 3. Merge scores
# ============================================================

MERGE_PYTHON="$TASK_PYTHON"

if [ ! -x "$MERGE_PYTHON" ]; then
  MERGE_PYTHON="$(command -v python3 || true)"
fi

if [ -z "$MERGE_PYTHON" ]; then
  printf '%s\n' \
    '{"correct":0.0,"quality":0.0,"assistant_quality":0.0,"final_scores":0.0,"reward":0.0}' \
    > "$FINAL_REWARD"

  exit 0
fi


"$MERGE_PYTHON" - \
  "$PROGRAMMATIC" \
  "$RK_OUTPUT" \
  "$FINAL_REWARD" \
  "$REPORT" \
  "$rewardkit_status" <<'PY'

import json
import math
import sys

from pathlib import Path
from typing import Any


programmatic_path = Path(sys.argv[1])
rewardkit_path = Path(sys.argv[2])
reward_path = Path(sys.argv[3])
report_path = Path(sys.argv[4])

rewardkit_status = int(sys.argv[5])


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )

        return value if isinstance(value, dict) else {}

    except Exception:
        return {}


def score(value: Any) -> float:
    try:
        number = float(value)

    except (TypeError, ValueError):
        return 0.0

    if not math.isfinite(number):
        return 0.0

    return max(0.0, min(1.0, number))


programmatic = load(programmatic_path)
rewardkit = load(rewardkit_path)


correct = score(
    rewardkit.get(
        "correct",
        programmatic.get("correct", 0.0),
    )
)


assistant_quality = score(
    rewardkit.get(
        "assistant_quality",
        programmatic.get(
            "assistant_quality",
            0.0,
        ),
    )
)


quality = score(
    rewardkit.get(
        "quality",
        0.0,
    )
)


# Gate:
# task failure => final score = 0
# task success => final score = LLM judge quality
final_scores = (
    quality
    if correct > 0.0
    else 0.0
)


# Harbor requires flat numeric top-level rewards.
final_reward = {
    "correct": correct,
    "quality": quality,
    "assistant_quality": assistant_quality,
    "final_scores": final_scores,
    "reward": final_scores,
}


reward_path.write_text(
    json.dumps(
        final_reward,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)


report = load(report_path)

report["reward_summary"] = {
    "content": {
        "correct": correct,
        "quality": quality,
        "assistant_quality": assistant_quality,
    },
    "final_scores": final_scores,
}

report["rewardkit_status"] = rewardkit_status


report_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)

report_path.write_text(
    json.dumps(
        report,
        indent=2,
        ensure_ascii=False,
    )
    + "\n",
    encoding="utf-8",
)

PY


exit 0
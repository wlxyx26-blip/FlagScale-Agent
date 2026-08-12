from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TRACE_LOCK = threading.Lock()

_SECRET_KEY_PATTERN = re.compile(
    r"(api[_-]?key|authorization|auth[_-]?token|"
    r"access[_-]?token|password|secret)",
    flags=re.IGNORECASE,
)

_SECRET_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(
        r"Bearer\s+[A-Za-z0-9._~+/=-]+",
        flags=re.IGNORECASE,
    ),
    # Redact credentials embedded in proxy or service URLs.
    re.compile(
        r"(?P<scheme>https?://)[^\s/@:]+:[^\s/@]+@",
        flags=re.IGNORECASE,
    ),
]


def _redact_text(value: str) -> str:
    result = value

    for pattern in _SECRET_VALUE_PATTERNS:
        if "scheme" in pattern.groupindex:
            result = pattern.sub(
                lambda match: f"{match.group('scheme')}***:***@",
                result,
            )
        else:
            result = pattern.sub("***REDACTED***", result)

    return result


def _make_json_safe(value: Any) -> Any:
    """转换SDK对象并递归移除密钥。"""

    if value is None or isinstance(
        value,
        (bool, int, float),
    ):
        return value

    if isinstance(value, str):
        return _redact_text(value)

    if isinstance(value, dict):
        result: dict[str, Any] = {}

        for key, item in value.items():
            key_text = str(key)

            if _SECRET_KEY_PATTERN.search(key_text):
                result[key_text] = "***REDACTED***"
            else:
                result[key_text] = _make_json_safe(item)

        return result

    if isinstance(value, (list, tuple, set)):
        return [
            _make_json_safe(item)
            for item in value
        ]

    # Anthropic/OpenAI SDK对象通常支持model_dump
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _make_json_safe(
                model_dump(mode="json")
            )
        except TypeError:
            return _make_json_safe(model_dump())

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _make_json_safe(to_dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return _make_json_safe(vars(value))
        except Exception:
            pass

    return _redact_text(str(value))


class TraceLogger:
    def __init__(self) -> None:
        self.enabled = (
            os.getenv(
                "FLAGSCALE_TRACE_ENABLED",
                "0",
            )
            == "1"
        )

        self.path = Path(
            os.getenv(
                "FLAGSCALE_TRACE_PATH",
                "/logs/agent/flagscale-events.jsonl",
            )
        )

        self.session_id = os.getenv(
            "FLAGSCALE_TRACE_SESSION_ID",
            str(uuid.uuid4()),
        )
        self._sequence_id = 0

        if self.enabled:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    def emit(
        self,
        event_type: str,
        **payload: Any,
    ) -> None:
        """Append one JSONL event without allowing tracing to break the agent."""
        if not self.enabled:
            return

        try:
            safe_payload = _make_json_safe(payload)

            with _TRACE_LOCK:
                self._sequence_id += 1
                event = {
                    "schema_version": 1,
                    "sequence_id": self._sequence_id,
                    "event_id": str(uuid.uuid4()),
                    "session_id": self.session_id,
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "event_type": event_type,
                    **safe_payload,
                }

                line = json.dumps(
                    event,
                    ensure_ascii=False,
                    default=str,
                )

                self.path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                with self.path.open(
                    "a",
                    encoding="utf-8",
                ) as file:
                    file.write(line + "\n")
                    file.flush()

        except Exception as exc:
            # Trace collection is observational and must never abort the task.
            print(
                "[TraceLogger] failed to write "
                f"{event_type}: {type(exc).__name__}: {exc}",
                flush=True,
            )


trace_logger = TraceLogger()
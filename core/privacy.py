from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Message


PII_PATTERNS: dict[str, tuple[str, str]] = {
    "EMAIL": (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
    "PHONE": (
        r"(\+91[\-\s]?)?[6-9]\d{9}|\+?1?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}",
        "[PHONE]",
    ),
    "STUDENT_ID": (r"\b\d{6,12}\b", "[STUDENT_ID]"),
    "AADHAAR": (r"\b\d{4}\s\d{4}\s\d{4}\b", "[AADHAAR]"),
}


def mask_pii(text: str) -> tuple[str, dict[str, str]]:
    masked_text = text
    pii_map: dict[str, str] = {}
    counters = {pii_type: 0 for pii_type in PII_PATTERNS}

    for pii_type, (pattern, _) in PII_PATTERNS.items():

        def replace_match(match: re.Match[str]) -> str:
            counters[pii_type] += 1
            token = f"[{pii_type}_{counters[pii_type]}]"
            pii_map[token] = match.group(0)
            return token

        masked_text = re.sub(pattern, replace_match, masked_text)

    return masked_text, pii_map


def restore_pii(masked_text: str, pii_map: dict[str, str]) -> str:
    restored_text = masked_text
    for token, value in pii_map.items():
        restored_text = restored_text.replace(token, value)
    return restored_text


def _log_dir() -> Path:
    configured = os.getenv("LOG_DIR", "./logs")
    path = Path(configured)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _token_types(pii_map: dict[str, str]) -> list[str]:
    token_types: list[str] = []
    for token in pii_map:
        token_type = token.strip("[]").rsplit("_", 1)[0]
        if token_type not in token_types:
            token_types.append(token_type)
    return token_types


def _write_pii_audit(session_id: str, pii_map: dict[str, str]) -> None:
    log_dir = _log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    token_types = _token_types(pii_map)
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"{timestamp}: PII detected in session {session_id} — types: {token_types}\n"
    with (log_dir / "pii_audit.log").open("a", encoding="utf-8") as log_file:
        log_file.write(line)


class PIIMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:
        if request.method != "POST" or request.url.path != "/query":
            return await call_next(request)

        body = await request.body()
        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            async def original_receive() -> Message:
                return {"type": "http.request", "body": body, "more_body": False}

            return await call_next(Request(request.scope, original_receive))

        query = payload.get("query")
        if isinstance(query, str):
            masked_query, pii_map = mask_pii(query)
            payload["query"] = masked_query
            if pii_map:
                session_id = str(payload.get("session_id") or "unknown")
                _write_pii_audit(session_id, pii_map)

        new_body = json.dumps(payload).encode("utf-8")

        async def receive() -> Message:
            return {"type": "http.request", "body": new_body, "more_body": False}

        return await call_next(Request(request.scope, receive))

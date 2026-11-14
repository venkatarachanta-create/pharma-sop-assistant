import json
from datetime import datetime, timezone
from pathlib import Path
from src.config.settings import settings


def log_audit_event(
    query: str,
    status: str,
    response: str,
    sources: list[str],
    latency_ms: float,
) -> None:
    log_path: Path = settings.audit_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "status": status,
        "response_preview": response[:300],
        "sources": sources,
        "latency_ms": round(latency_ms, 2),
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

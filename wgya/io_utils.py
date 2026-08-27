"""Idempotent result writing. Every experiment writes a JSON file whose name
embeds a hash of its config, so re-running with the same config overwrites
the same file and changing anything produces a new one."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path


def config_hash(config: dict) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def file_sha256(path, n=12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def write_results(out_dir, name: str, config: dict, results) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h = config_hash(config)
    path = out_dir / f"{name}_{h}.json"
    payload = {
        "config": config,
        "config_hash": h,
        "written_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def newest_result(dir_, pattern, name_filter=None):
    """Newest result file by its embedded written_at timestamp, falling
    back to mtime for files without one. Selecting on mtime alone breaks
    on fresh clones/rsyncs, where every file's mtime ties."""
    cands = [
        p for p in Path(dir_).glob(pattern) if name_filter is None or name_filter(p)
    ]

    def key(p):
        try:
            return (json.loads(p.read_text()).get("written_at", ""), p.stat().st_mtime)
        except (OSError, json.JSONDecodeError):
            return ("", p.stat().st_mtime)

    return max(cands, key=key) if cands else None


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)

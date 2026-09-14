"""Export local Codex sessions whose working directory is this project.

The source JSONL files under ``C:\\Users\\lizy\\.codex\\sessions`` are
treated as read-only.  The exporter writes a readable transcript, a compact
visible-message JSONL file, a gzip-compressed event archive, a source manifest,
and a reproducibility summary under the task's result directory.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(r"D:\Project_Files\python_project\project_20260821").resolve()
CODEX_DATA_ROOT = Path(r"C:\Users\lizy\.codex").resolve()
SESSION_ROOT = CODEX_DATA_ROOT / "sessions"
TASK_NAME = "codex_session_export_project_20260821"
RESULT_ROOT = PROJECT_ROOT / "results" / TASK_NAME
LOG_PATH = PROJECT_ROOT / "work_logs" / TASK_NAME / "execution_log.txt"
HANDOFF_PATH = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"
TARGET_CWD = str(PROJECT_ROOT).rstrip("\\/").casefold()
OUTPUT_NAMES = (
    "session_transcript.txt",
    "session_messages.jsonl",
    "session_events.jsonl.gz",
    "session_manifest.csv",
    "session_export_summary.txt",
)


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        handle.write(text)


def _normalise_cwd(value: object) -> str:
    return str(value or "").replace("/", "\\").rstrip("\\").casefold()


def _json_first_line(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        line = handle.readline()
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("session_meta is not a JSON object")
    return value


def _discover_matching_sessions() -> list[tuple[Path, dict[str, Any]]]:
    if not SESSION_ROOT.is_dir():
        raise FileNotFoundError(f"Codex session directory is missing: {SESSION_ROOT}")
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(SESSION_ROOT.rglob("*.jsonl"), key=lambda item: str(item).casefold()):
        try:
            first = _json_first_line(path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        payload = first.get("payload")
        if not isinstance(payload, dict):
            continue
        if _normalise_cwd(payload.get("cwd")) == TARGET_CWD:
            matches.append((path, first))
    if not matches:
        raise RuntimeError(f"No Codex sessions match project cwd: {PROJECT_ROOT}")
    return matches


def _safe_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _extract_message(event: dict[str, Any], source_file: str) -> dict[str, Any] | None:
    if event.get("type") != "response_item":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    role = payload.get("role")
    if role not in {"user", "assistant", "developer", "system"}:
        return None
    blocks = payload.get("content")
    parts: list[str] = []
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", ""))
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
            elif block_type:
                parts.append(f"[{block_type}]")
    if not parts:
        return None
    return {
        "source_file": source_file,
        "session_id": str(payload.get("session_id") or ""),
        "ordinal": event.get("ordinal"),
        "timestamp": event.get("timestamp"),
        "role": str(role),
        "response_type": str(payload.get("type") or ""),
        "text": "\n".join(parts),
    }


def _event_archive_record(
    source_file: str,
    session_id: str,
    line_number: int,
    event: object,
) -> dict[str, Any]:
    return {
        "source_file": source_file,
        "session_id": session_id,
        "line_number": line_number,
        "event": event,
    }


def _iter_export(
    sessions: Iterable[tuple[Path, dict[str, Any]]],
    archive_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str], int, int]:
    manifest_rows: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    parse_errors = 0
    archive_bytes = 0
    with gzip.open(archive_path, "wt", encoding="utf-8", compresslevel=9, newline="\n") as archive:
        for path, first in sessions:
            relative = path.relative_to(SESSION_ROOT).as_posix()
            payload = first.get("payload") if isinstance(first.get("payload"), dict) else {}
            session_id = str(payload.get("session_id") or payload.get("id") or "")
            before = path.stat()
            digest = hashlib.sha256()
            line_count = 0
            local_counts: Counter[str] = Counter()
            local_parse_errors = 0
            with path.open("rb") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    digest.update(raw_line)
                    line_count += 1
                    try:
                        event = json.loads(raw_line.decode("utf-8", errors="replace"))
                    except json.JSONDecodeError:
                        parse_errors += 1
                        local_parse_errors += 1
                        event = {
                            "_parse_error": True,
                            "raw_line_utf8": raw_line.decode("utf-8", errors="replace"),
                        }
                    event_type = (
                        str(event.get("type", "parse_error"))
                        if isinstance(event, dict)
                        else "non_object"
                    )
                    local_counts[event_type] += 1
                    event_counts[event_type] += 1
                    wrapper = _event_archive_record(relative, session_id, line_number, event)
                    archive.write(
                        json.dumps(wrapper, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    if isinstance(event, dict):
                        message = _extract_message(event, relative)
                        if message is not None:
                            if not message["session_id"]:
                                message["session_id"] = session_id
                            messages.append(message)
            after = path.stat()
            manifest_rows.append(
                {
                    "source_file": relative,
                    "absolute_source_path": str(path),
                    "session_id": session_id,
                    "start_timestamp": first.get("timestamp"),
                    "cwd": payload.get("cwd"),
                    "source": payload.get("source"),
                    "originator": payload.get("originator"),
                    "bytes": int(before.st_size),
                    "line_count": line_count,
                    "sha256": digest.hexdigest(),
                    "changed_during_export": bool(
                        before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
                    ),
                    "parse_error_count": local_parse_errors,
                    "event_counts": json.dumps(
                        dict(sorted(local_counts.items())), ensure_ascii=False
                    ),
                }
            )
    archive_bytes = archive_path.stat().st_size
    return manifest_rows, messages, event_counts, parse_errors, archive_bytes


def _message_sort_key(message: dict[str, Any]) -> tuple[str, str, int, str]:
    ordinal = message.get("ordinal")
    try:
        ordinal_value = int(ordinal)
    except (TypeError, ValueError):
        ordinal_value = 0
    return (
        str(message.get("timestamp") or ""),
        str(message.get("session_id") or ""),
        ordinal_value,
        str(message.get("source_file") or ""),
    )


def _write_messages(messages: list[dict[str, Any]], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for message in messages:
            handle.write(json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_transcript(messages: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "Codex project session export",
        f"Project cwd: {PROJECT_ROOT}",
        f"Generated: {_now()}",
        "Source: local .codex/sessions JSONL records; visible role messages only.",
        "The complete event archive is session_events.jsonl.gz.",
        "",
    ]
    for index, message in enumerate(messages, start=1):
        lines.extend(
            [
                f"===== Message {index} =====",
                f"role: {message['role']}",
                f"timestamp: {message.get('timestamp') or ''}",
                f"session_id: {message.get('session_id') or ''}",
                f"source_file: {message.get('source_file') or ''}",
                f"ordinal: {message.get('ordinal')}",
                "----- content -----",
                str(message["text"]),
                "",
            ]
        )
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _write_manifest(rows: list[dict[str, Any]], output: Path) -> None:
    columns = [
        "source_file",
        "absolute_source_path",
        "session_id",
        "start_timestamp",
        "cwd",
        "source",
        "originator",
        "bytes",
        "line_count",
        "sha256",
        "changed_during_export",
        "parse_error_count",
        "event_counts",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    sessions: list[tuple[Path, dict[str, Any]]],
    rows: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    event_counts: Counter[str],
    parse_errors: int,
    archive_bytes: int,
    output: Path,
) -> None:
    role_counts = Counter(str(message["role"]) for message in messages)
    source_bytes = sum(int(row["bytes"]) for row in rows)
    changed = [row["source_file"] for row in rows if row["changed_during_export"]]
    lines = [
        f"Task: {TASK_NAME}",
        f"Generated: {_now()}",
        f"Project cwd filter: {PROJECT_ROOT}",
        f"Source root: {SESSION_ROOT}",
        "",
        "Matched session files",
        f"count: {len(sessions)}",
        f"source bytes: {source_bytes}",
        f"archive bytes (gzip): {archive_bytes}",
        f"source files changed during export: {len(changed)}",
        f"malformed JSONL lines: {parse_errors}",
        "",
        "Visible message counts",
        f"total: {len(messages)}",
        f"user: {role_counts.get('user', 0)}",
        f"assistant: {role_counts.get('assistant', 0)}",
        f"developer: {role_counts.get('developer', 0)}",
        f"system: {role_counts.get('system', 0)}",
        "",
        "All event counts in matched source files",
    ]
    lines.extend(f"{key}: {value}" for key, value in sorted(event_counts.items()))
    lines.extend(
        [
            "",
            "Session manifest",
        ]
    )
    for row in rows:
        lines.append(
            f"{row['source_file']} | session={row['session_id']} | "
            f"bytes={row['bytes']} | lines={row['line_count']} | sha256={row['sha256']}"
        )
    if changed:
        lines.extend(["", "WARNING: source files changed while being read:", *changed])
    lines.extend(
        [
            "",
            "Output files",
            *[str(RESULT_ROOT / name) for name in OUTPUT_NAMES],
            "",
            "Interpretation",
            "session_transcript.txt contains visible user/assistant/developer/system "
            "role messages.",
            "session_events.jsonl.gz preserves every parsed source event with "
            "source-file and line metadata, including tool calls, outputs, context, "
            "compaction and token records.",
            "This is a local Codex session export, not an account-level ChatGPT data export.",
            "Source session files were read-only; no source file was moved, edited or deleted.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    started = datetime.now(UTC)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _append(
        LOG_PATH,
        f"\n[{_now()}] START {TASK_NAME}\n"
        f"Project cwd filter: {PROJECT_ROOT}\nSource root: {SESSION_ROOT}\n",
    )
    try:
        sessions = _discover_matching_sessions()
        archive_path = RESULT_ROOT / "session_events.jsonl.gz"
        rows, messages, event_counts, parse_errors, archive_bytes = _iter_export(
            sessions, archive_path
        )
        messages.sort(key=_message_sort_key)
        _write_transcript(messages, RESULT_ROOT / "session_transcript.txt")
        _write_messages(messages, RESULT_ROOT / "session_messages.jsonl")
        _write_manifest(rows, RESULT_ROOT / "session_manifest.csv")
        _write_summary(
            sessions,
            rows,
            messages,
            event_counts,
            parse_errors,
            archive_bytes,
            RESULT_ROOT / "session_export_summary.txt",
        )
        actual = tuple(sorted(path.name for path in RESULT_ROOT.iterdir() if path.is_file()))
        if actual != tuple(sorted(OUTPUT_NAMES)):
            raise RuntimeError(f"Export output gate failed: {actual}")
        elapsed = (datetime.now(UTC) - started).total_seconds()
        log_summary = (
            f"Completed {TASK_NAME}: matched_sessions={len(sessions)}, "
            f"visible_messages={len(messages)}, source_bytes={sum(int(r['bytes']) for r in rows)}, "
            f"archive_bytes={archive_bytes}, elapsed={elapsed:.3f}s. "
            "Source .codex session files were read-only.\n"
        )
        _append(LOG_PATH, f"[{_now()}] COMPLETE\n{log_summary}")
        _append(
            HANDOFF_PATH,
            f"\n\n{_now()} | Codex session export for project_20260821\n{log_summary}",
        )
        print(log_summary, end="")
        print(f"Result root: {RESULT_ROOT}")
    except Exception as exc:
        _append(
            LOG_PATH,
            f"[{_now()}] FAILED\n{type(exc).__name__}: {exc}\n"
            "No source session file was modified or deleted.\n",
        )
        raise


if __name__ == "__main__":
    main()

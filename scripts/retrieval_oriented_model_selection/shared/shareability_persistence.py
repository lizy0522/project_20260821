"""Human-readable checkpoint and validated small-array cache contracts."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .shareability_search import SearchState

CACHE_KINDS = (
    "basis_column",
    "observed_basis",
    "state_model",
    "fingerprint",
    "oracle",
    "candidate_score",
    "screening_score",
)


@dataclass(frozen=True)
class CacheKey:
    kind: str
    state_id: int | None
    side: str | None
    basis_ids: tuple[str, ...]
    bandwidth_label: str
    segment_fingerprint: str
    ridge_lambda: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in CACHE_KINDS or not self.bandwidth_label or not self.segment_fingerprint:
            raise ValueError("unknown cache kind or missing bandwidth/segment contract")
        if self.kind in {"basis_column", "observed_basis"} and self.ridge_lambda is not None:
            raise ValueError("basis columns cannot depend on Ridge lambda")
        if (
            self.kind in {"state_model", "fingerprint", "candidate_score", "screening_score"}
            and self.ridge_lambda is None
        ):
            raise ValueError("fitted model/fingerprint/score key needs Ridge lambda")
        object.__setattr__(self, "basis_ids", tuple(sorted(self.basis_ids)))

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.__dict__, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class MetadataCache:
    """Write-once file cache with interprocess lock and atomic metadata commit."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @contextmanager
    def _lock(self, key: CacheKey) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{key.digest}.lock"
        deadline = time.monotonic() + 10.0
        while True:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"cache key lock timed out: {key.digest}") from None
                time.sleep(0.02)
        try:
            os.close(descriptor)
            yield
        finally:
            path.unlink(missing_ok=True)

    def _publish_locked(self, key: CacheKey, array: np.ndarray) -> None:
        value = np.asarray(array)
        if value.dtype == object or not np.all(np.isfinite(value)):
            raise ValueError("cache requires finite non-object numeric arrays")
        data = self.root / f"{key.digest}.npy"
        metadata = self.root / f"{key.digest}.json"
        if data.exists() or metadata.exists():
            raise FileExistsError("incomplete or already committed cache entry")
        token = uuid.uuid4().hex
        temp_data = self.root / f".{key.digest}.{token}.npy"
        temp_metadata = self.root / f".{key.digest}.{token}.json"
        try:
            with temp_data.open("xb") as handle:
                np.save(handle, value, allow_pickle=False)
            digest = hashlib.sha256(temp_data.read_bytes()).hexdigest()
            temp_metadata.write_text(
                json.dumps(
                    {
                        "key": key.__dict__,
                        "shape": value.shape,
                        "dtype": str(value.dtype),
                        "sha256": digest,
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(temp_data, data)
            os.replace(temp_metadata, metadata)
        finally:
            temp_data.unlink(missing_ok=True)
            temp_metadata.unlink(missing_ok=True)

    def save(self, key: CacheKey, array: np.ndarray) -> None:
        with self._lock(key):
            self._publish_locked(key, array)

    def get_or_compute(self, key: CacheKey, build: Callable[[], np.ndarray]) -> np.ndarray:
        """Only lock owner computes; other workers wait and reuse committed entry."""

        with self._lock(key):
            if (self.root / f"{key.digest}.json").is_file():
                return self.load(key)
            value = build()
            self._publish_locked(key, value)
            return self.load(key)

    def load(self, key: CacheKey) -> np.ndarray:
        metadata = json.loads((self.root / f"{key.digest}.json").read_text(encoding="utf-8"))
        if metadata["key"] != json.loads(json.dumps(key.__dict__)):
            raise ValueError("cache key mismatch")
        path = self.root / f"{key.digest}.npy"
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("cache data checksum mismatch")
        value = np.load(path, allow_pickle=False)
        if list(value.shape) != metadata["shape"] or str(value.dtype) != metadata["dtype"]:
            raise ValueError("cache array shape/dtype mismatch")
        return value


def save_checkpoint(state: SearchState, path: Path, *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            )
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Fails atomically if checkpoint already exists.
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path) -> SearchState:
    return SearchState.from_dict(json.loads(path.read_text(encoding="utf-8")))


@contextmanager
def _artifact_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 10.0
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"screening artifact lock timed out: {path}") from None
            time.sleep(0.02)
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def save_screening_artifact(
    root: Path,
    stem: str,
    metadata: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    *,
    replace: bool = False,
) -> tuple[Path, Path]:
    """Atomically persist a compact screening score NPZ plus checked metadata."""
    if not stem or any(token in stem for token in ("/", "\\")) or not arrays:
        raise ValueError("screening artifact needs a simple stem and nonempty arrays")
    base = Path(root) / stem
    data_path = base.with_suffix(".npz")
    metadata_path = base.with_suffix(".json")
    lock_path = base.with_suffix(".lock")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    if any(not name or value.dtype == object for name, value in normalized.items()):
        raise ValueError("screening artifact arrays must be named non-object arrays")
    if any(
        np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value))
        for value in normalized.values()
    ):
        raise ValueError("screening artifact numeric arrays must be finite")
    with _artifact_lock(lock_path):
        if (data_path.exists() or metadata_path.exists()) and not replace:
            raise FileExistsError(data_path if data_path.exists() else metadata_path)
        token = uuid.uuid4().hex
        temp_data = base.parent / f".{stem}.{token}.npz"
        temp_metadata = base.parent / f".{stem}.{token}.json"
        try:
            with temp_data.open("xb") as handle:
                np.savez_compressed(handle, **normalized)
            data_sha = hashlib.sha256(temp_data.read_bytes()).hexdigest()
            payload = {
                "metadata": dict(metadata),
                "arrays": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in normalized.items()
                },
                "sha256": data_sha,
            }
            temp_metadata.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_data, data_path)
            os.replace(temp_metadata, metadata_path)
        finally:
            temp_data.unlink(missing_ok=True)
            temp_metadata.unlink(missing_ok=True)
    return data_path, metadata_path


def load_screening_artifact(
    root: Path,
    stem: str,
    expected_metadata: Mapping[str, object],
) -> dict[str, np.ndarray]:
    """Load a screening cache only when checksum and all expected fields agree."""
    base = Path(root) / stem
    data_path = base.with_suffix(".npz")
    metadata_path = base.with_suffix(".json")
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != payload.get("sha256"):
        raise ValueError("screening artifact checksum mismatch")
    recorded = payload.get("metadata")
    if not isinstance(recorded, dict) or any(
        recorded.get(key) != value for key, value in expected_metadata.items()
    ):
        raise ValueError("screening artifact metadata mismatch")
    with np.load(data_path, allow_pickle=False) as loaded:
        values = {name: loaded[name].copy() for name in loaded.files}
    declared = payload.get("arrays", {})
    if set(values) != set(declared):
        raise ValueError("screening artifact array names mismatch")
    for name, value in values.items():
        description = declared[name]
        if list(value.shape) != description["shape"] or str(value.dtype) != description["dtype"]:
            raise ValueError("screening artifact shape/dtype mismatch")
    return values


__all__ = [
    "CACHE_KINDS",
    "CacheKey",
    "MetadataCache",
    "load_checkpoint",
    "load_screening_artifact",
    "save_checkpoint",
    "save_screening_artifact",
]

"""Immutable candidate, bandwidth, and Ridge contracts for shareability search."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from low_bandwidth_behavior_analysis.shared import SampleRateSpec

K_MAX = 20
STATE_COUNT = 425
REAL_B_THRESHOLD_DB = -40.0


@dataclass(frozen=True, slots=True)
class ParallelExecutionSpec:
    """Runtime-only policy; deliberately excluded from CandidateModelSpec identity."""

    max_workers: int = 10
    start_method: str = "spawn"
    blas_threads_per_worker: int = 1
    cpu_affinity: None = None
    gpu_enabled: bool = False
    allow_nested_parallelism: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or self.max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer")
        if (
            isinstance(self.blas_threads_per_worker, bool)
            or not isinstance(self.blas_threads_per_worker, int)
            or self.blas_threads_per_worker < 1
        ):
            raise ValueError("blas_threads_per_worker must be a positive integer")
        if self.start_method != "spawn" or self.cpu_affinity is not None:
            raise ValueError("only spawn without CPU affinity is supported")
        if self.gpu_enabled or self.allow_nested_parallelism:
            raise ValueError("GPU and nested multiprocessing are not supported")

    def to_dict(self, *, effective_workers: int) -> dict[str, int | str | bool | None]:
        return {
            "requested_workers": self.max_workers,
            "effective_workers": effective_workers,
            "start_method": self.start_method,
            "blas_threads_per_worker": self.blas_threads_per_worker,
            "cpu_affinity": self.cpu_affinity,
            "gpu_enabled": self.gpu_enabled,
            "nested_parallelism": self.allow_nested_parallelism,
        }


@dataclass(frozen=True, slots=True)
class ObservationBandwidthSpec:
    label: str
    sample_rate: SampleRateSpec

    @property
    def bandwidth_hz(self) -> int:
        return self.sample_rate.observation_bandwidth_hz

    @property
    def bandwidth_ratio_to_b(self) -> float:
        return self.sample_rate.normalized_bandwidth_n

    @property
    def sample_rate_hz(self) -> int:
        return self.sample_rate.fs_out_hz

    def __post_init__(self) -> None:
        if not self.label or not isinstance(self.sample_rate, SampleRateSpec):
            raise ValueError("label and existing SampleRateSpec are required")

    def to_dict(self) -> dict[str, object]:
        return {"label": self.label, "sample_rate_index_k": self.sample_rate.sample_rate_index_k}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ObservationBandwidthSpec:
        return cls(str(data["label"]), SampleRateSpec(int(data["sample_rate_index_k"])))


@dataclass(frozen=True, slots=True)
class RidgeGridSpec:
    coarse_values: tuple[float, ...]
    dense_values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coarse_values", tuple(self.coarse_values))
        object.__setattr__(self, "dense_values", tuple(self.dense_values))
        if not self.coarse_values or any(
            not math.isfinite(value) or value < 0
            for value in self.coarse_values + self.dense_values
        ):
            raise ValueError("explicit finite, nonnegative Ridge values are required")

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(dict.fromkeys(self.coarse_values + self.dense_values))

    @classmethod
    def with_log_refinement(
        cls,
        coarse_values: tuple[float, ...],
        *,
        lower: float,
        upper: float,
        count: int,
    ) -> RidgeGridSpec:
        """Caller-defined refinement; no formal scientific lambda range is frozen."""

        if lower <= 0 or upper <= lower or count < 2:
            raise ValueError("positive refinement bounds and at least two points required")
        step = (math.log(upper) - math.log(lower)) / (count - 1)
        dense = tuple(math.exp(math.log(lower) + i * step) for i in range(count))
        return cls(coarse_values, dense)


@dataclass(frozen=True, slots=True)
class StructureSearchSpec:
    """A single scientific lambda and seed; dense Ridge is a separate phase."""

    seed_basis_id: str
    ridge_lambda: float = 1e-8
    max_support_size: int = 20
    beam_width: int = 3
    global_dmax: int = 4

    def __post_init__(self) -> None:
        if not self.seed_basis_id or self.ridge_lambda != 1e-8:
            raise ValueError("structure stage requires the frozen seed and 1e-8 Ridge")
        if self.global_dmax != 4 or not 1 <= self.max_support_size <= K_MAX or self.beam_width != 3:
            raise ValueError("invalid frozen shareability structure search contract")

    def candidate(
        self, observation: ObservationBandwidthSpec, basis_ids: tuple[str, ...]
    ) -> CandidateModelSpec:
        if self.seed_basis_id not in basis_ids:
            raise ValueError("all structure supports must contain x[n]")
        return CandidateModelSpec(observation, basis_ids, self.ridge_lambda)


@dataclass(frozen=True, slots=True)
class CandidateScreeningSpec:
    """Deterministic proposal policy; it never changes the exact objective."""

    enabled: bool = True
    method: str = "somp_residual_correlation"
    shortlist_size: int = 500
    exploit_size: int = 400
    explore_size: int = 100
    aend_weight: float = 0.5
    c2_weight: float = 0.5
    state_aggregation: str = "mean"
    deterministic: bool = True
    stratification_version: str = "order_interaction_max_delay_v1"

    def __post_init__(self) -> None:
        if self.method != "somp_residual_correlation" or self.state_aggregation != "mean":
            raise ValueError("only deterministic mean SOMP residual correlation is supported")
        if not self.deterministic:
            raise ValueError("formal screening must be deterministic")
        if self.shortlist_size < 1 or self.exploit_size < 0 or self.explore_size < 0:
            raise ValueError("screening sizes must be nonnegative with a positive shortlist")
        if self.exploit_size + self.explore_size != self.shortlist_size:
            raise ValueError("exploit_size + explore_size must equal shortlist_size")
        if (
            not math.isfinite(self.aend_weight)
            or not math.isfinite(self.c2_weight)
            or self.aend_weight < 0
            or self.c2_weight < 0
            or not math.isclose(self.aend_weight + self.c2_weight, 1.0, abs_tol=1e-12)
        ):
            raise ValueError("Aend/C2 weights must be finite, nonnegative and sum to one")
        if not self.stratification_version:
            raise ValueError("screening stratification version is required")

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "shortlist_size": self.shortlist_size,
            "exploit_size": self.exploit_size,
            "explore_size": self.explore_size,
            "aend_weight": self.aend_weight,
            "c2_weight": self.c2_weight,
            "state_aggregation": self.state_aggregation,
            "deterministic": self.deterministic,
            "stratification_version": self.stratification_version,
        }

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateModelSpec:
    bandwidth_spec: ObservationBandwidthSpec
    basis_ids: tuple[str, ...]
    ridge_lambda: float

    def __post_init__(self) -> None:
        if (
            not self.basis_ids
            or len(self.basis_ids) > K_MAX
            or len(set(self.basis_ids)) != len(self.basis_ids)
        ):
            raise ValueError("selected support must contain 1..20 unique basis IDs")
        if not all(isinstance(item, str) and item for item in self.basis_ids):
            raise ValueError("basis IDs must be nonempty strings")
        if not math.isfinite(self.ridge_lambda) or self.ridge_lambda < 0:
            raise ValueError("Ridge lambda must be finite and nonnegative")
        object.__setattr__(self, "basis_ids", tuple(sorted(self.basis_ids)))

    @property
    def k_selected(self) -> int:
        return len(self.basis_ids)

    @property
    def candidate_id(self) -> str:
        payload = self.to_dict()
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "bandwidth_spec": self.bandwidth_spec.to_dict(),
            "basis_ids": list(self.basis_ids),
            "k_selected": self.k_selected,
            "ridge_lambda": self.ridge_lambda,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CandidateModelSpec:
        return cls(
            ObservationBandwidthSpec.from_dict(data["bandwidth_spec"]),  # type: ignore[arg-type]
            tuple(data["basis_ids"]),  # type: ignore[arg-type]
            float(data["ridge_lambda"]),
        )


__all__ = [
    "K_MAX",
    "STATE_COUNT",
    "REAL_B_THRESHOLD_DB",
    "ObservationBandwidthSpec",
    "RidgeGridSpec",
    "CandidateModelSpec",
    "ParallelExecutionSpec",
    "StructureSearchSpec",
    "CandidateScreeningSpec",
]

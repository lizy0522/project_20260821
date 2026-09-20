"""Small, active-support statistics for segment-local shareability fitting.

The project's Ridge objective is mean squared error plus lambda times the
coefficient norm: the normal equations therefore use G + n_rows * lambda I.
No dictionary-wide Gram matrix is materialized.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def canonical_support(basis_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not basis_ids or len(set(basis_ids)) != len(basis_ids):
        raise ValueError("support must contain distinct basis IDs")
    return tuple(sorted(basis_ids))


def support_id(basis_ids: tuple[str, ...] | list[str]) -> str:
    payload = json.dumps(canonical_support(basis_ids), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ParentSupportStats:
    basis_ids: tuple[str, ...]
    gram: np.ndarray
    rhs: np.ndarray
    n_rows: int

    def __post_init__(self) -> None:
        k = len(self.basis_ids)
        if k < 1 or len(set(self.basis_ids)) != k or self.n_rows < 1:
            raise ValueError("invalid active support or number of rows")
        if self.gram.shape != (k, k) or self.rhs.shape != (k,):
            raise ValueError("Gram/RHS shape differs from support")

    @classmethod
    def from_design(
        cls, ids: tuple[str, ...], phi: np.ndarray, y: np.ndarray
    ) -> ParentSupportStats:
        if phi.ndim != 2 or phi.shape != (y.size, len(ids)):
            raise ValueError("design/target shape mismatch")
        return cls(ids, phi.conj().T @ phi, phi.conj().T @ y, phi.shape[0])

    def without(self, basis_id: str) -> ParentSupportStats:
        if basis_id not in self.basis_ids or len(self.basis_ids) < 2:
            raise ValueError("cannot remove this basis")
        idx = self.basis_ids.index(basis_id)
        keep = [i for i in range(len(self.basis_ids)) if i != idx]
        return ParentSupportStats(
            tuple(self.basis_ids[i] for i in keep),
            self.gram[np.ix_(keep, keep)],
            self.rhs[keep],
            self.n_rows,
        )

    def append(
        self, basis_id: str, cross: np.ndarray, energy: float, rhs: complex
    ) -> ParentSupportStats:
        if basis_id in self.basis_ids or cross.shape != (len(self.basis_ids),):
            raise ValueError("invalid child basis or cross statistics")
        k = len(self.basis_ids)
        gram = np.empty((k + 1, k + 1), dtype=np.complex128)
        gram[:k, :k] = self.gram
        gram[:k, k] = cross
        gram[k, :k] = cross.conj()
        gram[k, k] = energy
        return ParentSupportStats(
            self.basis_ids + (basis_id,), gram, np.r_[self.rhs, rhs], self.n_rows
        )


@dataclass(slots=True)
class ParentFactor:
    stats: ParentSupportStats
    ridge_lambda: float
    factor: tuple[np.ndarray, bool]
    coefficients: np.ndarray

    @classmethod
    def build(cls, stats: ParentSupportStats, ridge_lambda: float) -> ParentFactor:
        if ridge_lambda < 0:
            raise ValueError("negative regularization")
        a = stats.gram.copy()
        a.flat[:: len(stats.basis_ids) + 1] += stats.n_rows * ridge_lambda
        factor = cho_factor(a, lower=True, check_finite=True)
        return cls(stats, ridge_lambda, factor, cho_solve(factor, stats.rhs))

    def expand_block(
        self,
        ids: tuple[str, ...],
        phi_parent: np.ndarray,
        columns: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, list[ParentSupportStats], list[str]]:
        """Solve all child normal equations using one parent factor/multi-RHS.

        columns is a single state's candidate block; the caller streams states
        and must not construct a state-by-sample-by-candidate tensor.
        """
        parent = self.stats
        if (
            columns.shape != (parent.n_rows, len(ids))
            or phi_parent.shape != (parent.n_rows, len(parent.basis_ids))
            or y.shape != (parent.n_rows,)
        ):
            raise ValueError("candidate columns must already be effective segment rows")
        if len(set(ids)) != len(ids) or set(ids) & set(parent.basis_ids):
            raise ValueError("duplicate child basis")
        return self.expand_from_statistics(ids, *block_statistics(phi_parent, columns, y))

    def expand_from_statistics(
        self,
        ids: tuple[str, ...],
        cross: np.ndarray,
        energy: np.ndarray,
        rhs: np.ndarray,
        *,
        tolerance: float = 1e-10,
    ) -> tuple[np.ndarray, list[ParentSupportStats], list[str]]:
        """Update children; suspect Schur pivots fall back to full Hermitian solve."""
        k, block = cross.shape
        if (
            k != len(self.stats.basis_ids)
            or len(ids) != block
            or energy.shape != (block,)
            or rhs.shape != (block,)
        ):
            raise ValueError("candidate statistics shape mismatch")
        if set(ids) & set(self.stats.basis_ids) or len(set(ids)) != block:
            raise ValueError("duplicate candidate basis")
        z = cho_solve(self.factor, cross)
        schur = (
            energy + self.stats.n_rows * self.ridge_lambda - np.einsum("kb,kb->b", cross.conj(), z)
        )
        residual = rhs - cross.conj().T @ self.coefficients
        output = np.empty((block, k + 1), dtype=np.complex128)
        children: list[ParentSupportStats] = []
        fallbacks: list[str] = []
        for i, basis_id in enumerate(ids):
            child = self.stats.append(basis_id, cross[:, i], float(energy[i]), rhs[i])
            children.append(child)
            pivot = schur[i]
            scale = max(1.0, float(abs(energy[i]) + self.stats.n_rows * self.ridge_lambda))
            if (
                not np.isfinite(pivot)
                or abs(pivot.imag) > tolerance * scale
                or pivot.real <= tolerance * scale
            ):
                # Same normal-equation objective, without the unstable Schur division.
                full = child.gram.copy()
                full.flat[:: k + 2] += child.n_rows * self.ridge_lambda
                output[i] = np.linalg.solve(full, child.rhs)
                fallbacks.append(basis_id)
            else:
                alpha = residual[i] / pivot.real
                output[i, :k] = self.coefficients - z[:, i] * alpha
                output[i, k] = alpha
        return output, children, fallbacks


def block_statistics(
    phi_parent: np.ndarray, columns: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if phi_parent.shape[0] != columns.shape[0] or columns.shape[0] != y.size:
        raise ValueError("segment effective rows differ")
    cross = phi_parent.conj().T @ columns
    energy = np.einsum("nb,nb->b", columns.conj(), columns).real
    rhs = columns.conj().T @ y
    return cross, energy, rhs


def coefficient_distance(
    query: np.ndarray, lut: np.ndarray, common_b_gram: np.ndarray
) -> np.ndarray:
    """Unadjusted query-reference CNMSE; numerical ranking needs regression gating."""
    h = common_b_gram
    q = np.einsum("ik,kl,il->i", query.conj(), h, query).real
    lut_energy = np.einsum("ik,kl,il->i", lut.conj(), h, lut).real
    if np.any(q <= 0):
        raise ValueError("query fingerprint energy must be positive")
    squared = q[:, None] + lut_energy[None, :] - 2 * np.real(query.conj() @ h @ lut.T)
    tol = 32 * np.finfo(float).eps * (q[:, None] + lut_energy[None, :] + 1)
    squared[np.abs(squared) <= tol] = 0
    squared = np.maximum(squared, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(squared / q[:, None])

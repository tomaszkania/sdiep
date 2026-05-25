#!/usr/bin/env python3
"""Generate the NLA-oriented application experiments for the SDIEP paper."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import csv

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RealisationMetrics:
    """Diagnostics for a computed Schur realisation.

    Parameters
    ----------
    experiment:
        Name of the experiment.
    basis:
        Name of the basis used to construct ``Q``.
    n:
        Matrix dimension.
    trace_sum:
        Prescribed trace sum ``1 + sum(lambda_j)``.
    slem:
        Second-largest eigenvalue modulus among non-Perron eigenvalues.
    min_entry:
        Minimum entry of ``P = Q diag(lambda) Q.T``.
    row_residual_inf:
        Infinity norm of row-sum residuals.
    symmetry_residual_fro:
        Frobenius norm of the skew-symmetric part.
    spectral_residual_inf:
        Infinity norm between sorted computed and prescribed eigenvalues.
    iterations_to_tol:
        First iteration at which the monitored error is below the tolerance;
        ``-1`` records a failed comparator.
    """

    experiment: str
    basis: str
    n: int
    trace_sum: float
    slem: float
    min_entry: float
    row_residual_inf: float
    symmetry_residual_fro: float
    spectral_residual_inf: float
    iterations_to_tol: int


def sylvester_hadamard(n: int) -> Array:
    """Return the Sylvester Hadamard matrix of order ``n``.

    Parameters
    ----------
    n:
        A positive power of two.

    Returns
    -------
    numpy.ndarray
        The ``n`` by ``n`` Sylvester Hadamard matrix with entries ``+-1``.
    """
    if n < 1 or n & (n - 1):
        raise ValueError("n must be a positive power of two")
    h = np.array([[1.0]], dtype=float)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def canonical_cycle_basis(n: int) -> Array:
    """Construct the canonical real cycle-walk basis.

    Parameters
    ----------
    n:
        Dimension of the basis.

    Returns
    -------
    numpy.ndarray
        Orthogonal matrix with first column ``1/sqrt(n)``.
    """
    grid = np.arange(n, dtype=float)
    q = np.empty((n, n), dtype=float)
    q[:, 0] = 1.0 / np.sqrt(n)
    for j in range(1, n):
        q[:, j] = np.sqrt(2.0 / n) * np.sin(2.0 * np.pi * j * grid / n + np.pi / 4.0)
    return q


def phase_cycle_basis(n: int) -> Array:
    """Construct the phase-optimised cycle basis.

    Parameters
    ----------
    n:
        Dimension of the basis.

    Returns
    -------
    numpy.ndarray
        Orthogonal matrix with first column ``1/sqrt(n)``.
    """
    grid = np.arange(n, dtype=float)
    columns: list[Array] = [np.ones(n, dtype=float) / np.sqrt(n)]
    for j in range(1, (n - 1) // 2 + 1):
        n_prime = n // int(np.gcd(j, n))
        effective_spacing = (np.pi / 2.0) * int(np.gcd(n_prime, 4)) / n_prime
        phi = effective_spacing / 2.0
        theta = 2.0 * np.pi * j * grid / n + phi
        columns.append(np.sqrt(2.0 / n) * np.sin(theta))
        columns.append(np.sqrt(2.0 / n) * np.cos(theta))
    if n % 2 == 0:
        columns.append(((-1.0) ** np.arange(n, dtype=float)) / np.sqrt(n))
    q = np.column_stack(columns)
    if q.shape != (n, n):
        raise RuntimeError(f"unexpected basis shape {q.shape}; expected {(n, n)}")
    return q


def schur_matrix(q: Array, eigenvalues: Array) -> Array:
    """Form ``Q diag(eigenvalues) Q.T`` without materialising the diagonal matrix.

    Parameters
    ----------
    q:
        Orthogonal basis matrix.
    eigenvalues:
        Prescribed eigenvalues in the column order of ``q``.

    Returns
    -------
    numpy.ndarray
        Symmetric Schur realisation candidate.
    """
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("q must be a square matrix")
    if eigenvalues.shape != (q.shape[0],):
        raise ValueError("eigenvalues must have shape (n,)")
    return (q * eigenvalues[np.newaxis, :]) @ q.T


def spectral_residual(p: Array, eigenvalues: Array) -> float:
    """Compute an infinity-norm eigenvalue residual after sorting.

    Parameters
    ----------
    p:
        Symmetric matrix whose eigenvalues are checked.
    eigenvalues:
        Prescribed eigenvalues.

    Returns
    -------
    float
        Maximum absolute difference between sorted spectra.
    """
    computed = np.linalg.eigvalsh(p)
    return float(np.max(np.abs(np.sort(computed) - np.sort(eigenvalues))))


def total_variation_curve(p: Array, steps: int) -> Array:
    """Compute total-variation distance from uniform for a point start.

    Parameters
    ----------
    p:
        Row-stochastic transition matrix.
    steps:
        Number of iterations to simulate.

    Returns
    -------
    numpy.ndarray
        Distances for iterations ``0, ..., steps``.
    """
    n = p.shape[0]
    distribution = np.zeros(n, dtype=float)
    distribution[0] = 1.0
    uniform = np.ones(n, dtype=float) / n
    values: list[float] = []
    for _ in range(steps + 1):
        values.append(float(0.5 * np.sum(np.abs(distribution - uniform))))
        distribution = distribution @ p
    return np.asarray(values, dtype=float)


def consensus_error_curve(p: Array, x0: Array, steps: int) -> Array:
    """Compute relative average-consensus errors.

    Parameters
    ----------
    p:
        Symmetric row-stochastic update matrix.
    x0:
        Initial values at the nodes.
    steps:
        Number of iterations to simulate.

    Returns
    -------
    numpy.ndarray
        Relative Euclidean errors to the average for iterations ``0, ..., steps``.
    """
    mean = float(np.mean(x0))
    target = np.full_like(x0, mean)
    denom = float(np.linalg.norm(x0 - target))
    if denom == 0.0:
        raise ValueError("x0 must not already be consensus")
    x = x0.copy()
    values: list[float] = []
    for _ in range(steps + 1):
        values.append(float(np.linalg.norm(x - target) / denom))
        x = p @ x
    return np.asarray(values, dtype=float)


def first_below(values: Array, tol: float) -> int:
    """Return the first index at which ``values`` is below ``tol``.

    Parameters
    ----------
    values:
        Monitored non-negative values.
    tol:
        Target tolerance.

    Returns
    -------
    int
        First qualifying index, or ``-1`` if no index qualifies.
    """
    hits = np.flatnonzero(values <= tol)
    return int(hits[0]) if hits.size else -1


def write_metrics(path: Path, metrics: Iterable[RealisationMetrics]) -> None:
    """Write realisation metrics to a CSV file.

    Parameters
    ----------
    path:
        Output path.
    metrics:
        Metric rows to write.
    """
    rows = list(metrics)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(RealisationMetrics.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def diagnostics(
    experiment: str,
    basis: str,
    q: Array,
    eigenvalues: Array,
    iterations_to_tol: int,
) -> RealisationMetrics:
    """Collect standard diagnostics for a Schur realisation.

    Parameters
    ----------
    experiment:
        Experiment name.
    basis:
        Basis name.
    q:
        Orthogonal basis matrix.
    eigenvalues:
        Prescribed spectrum.
    iterations_to_tol:
        Iteration count from the corresponding application test.

    Returns
    -------
    RealisationMetrics
        Numerical diagnostics.
    """
    p = schur_matrix(q, eigenvalues)
    return RealisationMetrics(
        experiment=experiment,
        basis=basis,
        n=p.shape[0],
        trace_sum=float(np.sum(eigenvalues)),
        slem=float(np.max(np.abs(eigenvalues[1:]))),
        min_entry=float(np.min(p)),
        row_residual_inf=float(np.max(np.abs(p @ np.ones(p.shape[0]) - np.ones(p.shape[0])))),
        symmetry_residual_fro=float(np.linalg.norm(p - p.T, ord="fro")),
        spectral_residual_inf=spectral_residual(p, eigenvalues),
        iterations_to_tol=iterations_to_tol,
    )


def run_markov_application(data_dir: Path, figure_dir: Path) -> list[RealisationMetrics]:
    """Run the reversible Markov-chain mixing experiment.

    Parameters
    ----------
    data_dir:
        Directory where CSV outputs are written.
    figure_dir:
        Directory where figure PDFs are written.

    Returns
    -------
    list[RealisationMetrics]
        Diagnostics for the tested spectra.
    """
    n = 16
    q = sylvester_hadamard(n) / np.sqrt(n)
    trace_sum = 0.05
    rates = [0.25, 0.50, 0.75, 0.90]
    steps = 80
    tol = 1e-3
    t = np.arange(steps + 1)
    metrics: list[RealisationMetrics] = []
    curves: list[tuple[float, Array]] = []

    for rate in rates:
        eigenvalues = np.empty(n, dtype=float)
        eigenvalues[0] = 1.0
        eigenvalues[1] = -rate
        eigenvalues[2:] = -(1.0 - trace_sum - rate) / (n - 2)
        p = schur_matrix(q, eigenvalues)
        tv = total_variation_curve(p, steps)
        curves.append((rate, tv))
        metrics.append(
            diagnostics(
                experiment="markov_mixing",
                basis=f"Walsh-Hadamard r={rate:.2f}",
                q=q,
                eigenvalues=eigenvalues,
                iterations_to_tol=first_below(tv, tol),
            )
        )

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.8, 4.3))
    for rate, tv in curves:
        plt.semilogy(t, tv, label=fr"target SLEM ${rate:.2f}$")
    plt.axhline(tol, linestyle="--", linewidth=1, label=fr"tolerance ${tol:g}$")
    plt.xlabel(r"iteration $m$")
    plt.ylabel(r"total-variation distance to uniform")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "fig_exp6_markov_mixing.pdf")
    plt.close()

    write_metrics(data_dir / "exp6_markov_metrics.csv", metrics)
    return metrics


def run_consensus_application(data_dir: Path, figure_dir: Path) -> list[RealisationMetrics]:
    """Run the average-consensus application experiment.

    Parameters
    ----------
    data_dir:
        Directory where CSV outputs are written.
    figure_dir:
        Directory where figure PDFs are written.

    Returns
    -------
    list[RealisationMetrics]
        Diagnostics for phase and canonical matrices.
    """
    n = 24
    trace_sum = 0.495
    spike = 1.0 - trace_sum
    eigenvalues = np.zeros(n, dtype=float)
    eigenvalues[0] = 1.0
    eigenvalues[1] = -spike

    q_phase = phase_cycle_basis(n)
    q_canonical = canonical_cycle_basis(n)
    p_phase = schur_matrix(q_phase, eigenvalues)

    rng = np.random.default_rng(20260525)
    x0 = rng.normal(size=n)
    steps = 40
    tol = 1e-6
    errors = consensus_error_curve(p_phase, x0, steps)
    reference = np.maximum(errors[1], 1e-16) * np.abs(eigenvalues[1]) ** np.arange(steps + 1)
    reference[0] = 1.0

    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.8, 4.3))
    plt.semilogy(np.arange(steps + 1), errors, label="phase-optimised matrix")
    plt.semilogy(
        np.arange(steps + 1),
        reference,
        linestyle="--",
        label=fr"spectral rate ${abs(eigenvalues[1]):.3f}^m$",
    )
    plt.axhline(tol, linestyle=":", linewidth=1, label=fr"tolerance ${tol:g}$")
    plt.xlabel(r"iteration $m$")
    plt.ylabel(r"relative disagreement norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "fig_exp7_consensus.pdf")
    plt.close()

    metrics = [
        diagnostics(
            experiment="consensus",
            basis="phase-optimised cycle",
            q=q_phase,
            eigenvalues=eigenvalues,
            iterations_to_tol=first_below(errors, tol),
        ),
        diagnostics(
            experiment="consensus",
            basis="canonical cycle",
            q=q_canonical,
            eigenvalues=eigenvalues,
            iterations_to_tol=-1,
        ),
    ]
    write_metrics(data_dir / "exp7_consensus_metrics.csv", metrics)
    curve_data = np.column_stack([np.arange(steps + 1), errors, reference])
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        data_dir / "exp7_consensus_curve.csv",
        curve_data,
        delimiter=",",
        header="iteration,relative_disagreement,spectral_reference",
        comments="",
    )
    return metrics


def run_application_experiments(data_dir: Path, figure_dir: Path) -> list[RealisationMetrics]:
    """Generate all application figures and CSV files.

    Parameters
    ----------
    data_dir:
        Directory where CSV outputs are written.
    figure_dir:
        Directory where figure PDFs are written.

    Returns
    -------
    list[RealisationMetrics]
        Diagnostics from all application experiments.
    """
    all_metrics: list[RealisationMetrics] = []
    all_metrics.extend(run_markov_application(data_dir, figure_dir))
    all_metrics.extend(run_consensus_application(data_dir, figure_dir))
    write_metrics(data_dir / "exp_application_metrics.csv", all_metrics)
    return all_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Directory for CSV outputs.")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Directory for figure PDFs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the application experiments from the command line."""
    args = parse_args()
    run_application_experiments(args.out_dir, args.figure_dir)


if __name__ == "__main__":
    main()

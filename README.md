# sdiep-schur

Reference implementation and reproducible experiments for Schur-template constructions in the symmetric doubly stochastic inverse eigenvalue problem (SDIEP).

This repository accompanies the paper

> *Dimension-dependent bounds for the SDIEP via phase optimisation and Paley-type constructions*.

The NLA-oriented revision keeps this title and adds the numerical linear algebra material requested by the editors: a structured Schur realisation workflow, certification diagnostics, dense timing tests, Monte-Carlo tests, and application experiments for reversible Markov chains and average-consensus iterations.

## What the software does

Given a basis `Q` and a Suleĭmanova spectrum `Λ = diag(1, λ₂, …, λ_n)`, the package can:

- build the candidate matrix `P(Λ) = Q Λ Qᵀ`;
- compute the coherence certificate `M(Q)`;
- evaluate the explicit sufficient thresholds `δ_n` and `δ_n^(ph)`;
- sample random Suleĭmanova lists with a prescribed trace sum;
- reproduce the numerical experiments and publication figures.

## Repository layout

| Path | Purpose |
|---|---|
| `src/sdiep_schur/` | Core package |
| `scripts/reproduce_paper.py` | Regenerates all CSV data and figures used in the paper |
| `scripts/nla_application_experiments.py` | Regenerates the Markov-chain and consensus application tests |
| `tests/test_core.py` | Lightweight regression tests |
| `data/` | Generated CSV outputs |
| `paper/` | Manuscript sources, response letter, and compiled PDFs |
| `paper/figures/` | Figure files used by the manuscript |
| `paper/data/` | CSV tables used in the manuscript archive |
| `notebooks/` | Optional exploratory notebooks |

## Installation

```bash
pip install -e .
pip install -e .[test]
```

## Quick start

```python
import numpy as np
from sdiep_schur import CycleBasis, delta_cycle

n = 12
Q = CycleBasis.create(n)
lambda_diag = np.zeros(n)
lambda_diag[0] = 1.0
lambda_diag[1] = -0.5
P = Q.compute_P(lambda_diag)

print(Q.coherence())
print(delta_cycle(n))
print(P.min())
```

## Reproducibility

To regenerate the paper data and figures:

```bash
python scripts/reproduce_paper.py --out-dir data --figure-dir paper/figures --paper-data-dir paper/data
```

To rerun only the NLA application experiments:

```bash
python scripts/nla_application_experiments.py --out-dir data --figure-dir paper/figures
```

To compile the manuscript from the repository root:

```bash
latexmk -pdf -cd -jobname=main_clean paper/main_clean.tex
```

The compiled revised manuscript is tracked as `paper/main_clean.pdf`, and the response letter is tracked as `paper/response_to_editor.pdf`.

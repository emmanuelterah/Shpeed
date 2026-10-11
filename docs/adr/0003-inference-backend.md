# 0003. Inference backend for Bayesian models

- Status: Proposed (decide at the end of Phase 3)
- Date: 2026-10-11
- Roadmap: P3-12

## Context

The pace model is refitted after every session of a race weekend, and the post-qualifying run has a time budget.

## Options

- **PyMC (NUTS):** most familiar, good diagnostics.
- **NumPyro (JAX):** much faster sampling; more setup.
- **Variational inference:** fastest, with approximation error to measure.

## Decision criteria

Time for a full weekend update on a laptop, posterior predictive check results, and R-hat / ESS on held-out races. Record the measurements here before accepting.

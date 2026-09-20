# Chamberland Case Study Provenance

## Source model

Chamberland, E., Moravveji, S., Doyon, N., Duchesne, S. (2024). A
computational model of Alzheimer's disease at the nano, micro, and
macroscales. Frontiers in Neuroinformatics.

Parameter and equation ground truth: Moravveji, S., Sadia, H., Doyon,
N., Duchesne, S. (2025). Sensitivity analysis of a mathematical model
of Alzheimer's disease progression unveils important causal pathways.
Frontiers in Neuroinformatics. This 2025 paper's accompanying code
release is the oracle used throughout this case study's verification,
copied unmodified into `reference/authors_orig/`.

## What was found wrong in the earlier reconstruction

An earlier reconstruction of this model, used throughout the original
phase1 through phase10 pipeline now in `legacy/`, was built without
access to the authors' actual parameter values or code. Once the 2025
paper and its code release were found, direct comparison surfaced
three real deviations, not just missing calibration:

1. Time unit: the earlier reconstruction ran on age in years. The
real model runs on age in days. A three order of magnitude timescale
error, not a units label issue, since the rate constants themselves
were also on the wrong scale.

2. Neuron death hazard: the earlier reconstruction used a
Michaelis-Menten saturation term for the intracellular NFT contribution
to neuron death. The real model uses a sigmoid with its own steepness
exponent, `n`.

3. Amyloid oligomer to plaque conversion: the earlier reconstruction's
conversion term was linear in oligomer concentration. The real
equation squares it.

## Verification method

Three gates, defined before running them:

Gate 1, static equivalence: every parameter, every initial condition,
and randomized right hand side evaluations compared directly against
the reference implementation. 480 of 480 checks pass. An additional
1900 row diagnostic decomposition of the RHS checks down to individual
state variables is also available and also passes in full, but is not
counted as additional validation units, see `gate1_recovery.py` for
the exact counting convention.

Gate 2, numerical trajectory equivalence: full lifespan trajectories
integrated independently by the reference implementation and this
reconstruction, compared by state and by demographic and genotype
configuration. Three of four configurations reproduce at machine
precision. The fourth, APOE4 positive males, diverges to roughly
1e-3 relative scale in several states. This was confirmed to be a
property of the source model, not the reconstruction, by perturbing
the reference implementation's own initial conditions by a tiny
relative amount on the same configuration and observing a comparable
divergence, see `gate2_self_perturbation.py`.

Gate 3, published behavior reproduction: three criteria defined before
running, trajectory direction behavior, the APOE4 effect on amyloid
plaque, and age dependent neuronal loss. 3 of 3 pass.

## Files

- `reference/chamberland_reference.py`: the clean room reconstruction.
- `reference/authors_orig/`: unmodified copies of the reference
  implementation, used only for verification, not part of this
  project's own contribution.
- `gate1_recovery.py`, `gate2_recovery.py`,
  `gate2_self_perturbation.py`, `gate3_recovery.py`: verification
  scripts, run in that order.
- `figures/make_figure_c1.py` through `make_figure_c3.py`: figure
  generation, run after the corresponding gate script.

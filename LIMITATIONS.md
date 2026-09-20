# Limitations

## Chamberland case study

The APOE4 positive male configuration does not reproduce the reference
implementation at machine precision the way the other three
demographic and genotype configurations do. This was traced to several
state variables sitting near numerical equilibrium at values below the
solver's own absolute tolerance in that specific configuration.
Perturbing the original authors' own code by a tiny relative amount on
the same configuration reproduces a comparable divergence, so this is
treated as a property of the source model's numerical behavior in that
regime, not an error in the reconstruction. Quantitative claims about
that specific configuration should be read with this in mind.

## Vaughan case study

Vaughan's fitted model parameters and initial conditions are not
publicly available as a complete set for any patient cluster. The
exploratory ensemble used in this project samples plausible ranges
around a non-canonical fixture, using published cluster specific
ranges where they exist for nine parameters, and broad exploratory
ranges elsewhere. This ensemble is not a reconstruction of Vaughan's
own fitted patient clusters and should not be described as one.
Conclusions drawn from it describe how ImmunoGraph's outputs behave
under parameter uncertainty within the published equation structure,
not predictions about any specific patient population.

## Normalization

The sensitivity based graph construction method normalizes each
state's local sensitivity by that state's own characteristic rate of
change over its simulated trajectory. This prevents instantaneous
singularities at points where a state's derivative crosses zero, but
remains ill conditioned for states that sit near equilibrium across
most of their trajectory, since the normalizing quantity itself
becomes small. This was found during the normalization audit of the
Vaughan ensemble and is documented rather than patched, since no
principled alternative scale is available given Vaughan's
non-dimensional units. Weighted magnitude statistics for affected
states should be read with this limitation in mind. Rank based and
topology based statistics were checked and found substantially more
robust to this effect.

## V5 held out comparison

The correspondence set between the Vaughan model and the Helmy cohort
is limited to three cytokines, IL-1, IL-12, and IL-10, based on which
states have direct, unambiguous coverage in both places. Per patient
sample sizes in the Helmy data for this comparison are small, nine,
six, and five patients respectively. The reported agreement statistics
describe how a large ensemble of parameterizations relates to a fixed,
independently derived human ordering, not a probability distribution
over the true underlying biological parameters, and should not be read
as evidence from independent biological trials.

## General

Substantial portions of this project's code, verification scripts, and
analysis were developed with AI assistance across multiple sessions.
See `docs/DEVELOPMENT_LOG.md` for what was independently verified and
how. Independent human review of the core scientific claims is
recommended before this work is cited or built upon.

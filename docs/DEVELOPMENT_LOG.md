# Development Log

Chronological account of how this project's understanding of itself
changed. Written so the reasoning behind each correction is visible,
not just the fact that a correction happened.

## 1. Initial audit

The original composite backbone (Chamberland aging model plus an added
TBI cytokine module) was reviewed before any new work began. Its own
build log already flagged its parameters as placeholder values, not
calibrated against the source paper's numbers. The original model
authors' own paper describes some of its outputs, including lifetime
microglia activation percentage, as clearly inaccurate in places. A
literature review confirmed the graph construction method itself was a
real, defensible contribution, but situated in an existing space of
adjacent prior art rather than an empty one: ecology's Jacobian based
community matrix tradition, gene regulatory network to graph
correspondence theory, and a concurrent independent preprint using a
similar time indexed Jacobian approach on different data.

## 2. Parameter and equation audit

A 2025 follow up paper by the original model's authors was found,
performing a sensitivity analysis of the same model and releasing both
a literature sourced parameter appendix and a public code repository.
This directly contradicted the earlier assumption that the model's
real parameters were unrecoverable. Comparing the actual source code
against the project's existing reconstruction found two genuine
equation deviations: the neuron death term used a Michaelis-Menten
form where the source model uses a sigmoid with its own steepness
exponent, and the amyloid oligomer to plaque conversion term was
missing a square. Most significantly, the reconstruction ran on age in
years while the real model runs on age in days, a three order of
magnitude timescale error.

## 3. Clean room rebuild and gates

A new implementation was built directly from the authors' released
code rather than patched from the old reconstruction. Three gates were
defined before running them: static equivalence against the reference
implementation, numerical trajectory equivalence, and reproduction of
published behaviors. Gate 1 initially reported 2280 rather than the
expected 480 total checks, traced to a state level decomposition of
each vector level RHS check being counted separately in the export.
Resolved by keeping two separate exports, a canonical count matching
the original validation unit definition, and a diagnostic
decomposition explicitly labeled as such. Gate 2 found three of four
demographic and genotype configurations reproduce at machine
precision. The fourth, APOE4 positive males, diverges, traced to a
near equilibrium state whose value sits below the solver's own
absolute tolerance. Perturbing the original authors' own code by a
tiny relative amount on the same configuration reproduced the same
order of divergence, confirming this is a property of the source
model, not an error in the reconstruction.

## 4. Retiring the composite

The Chamberland and Vaughan composite splice was reviewed for whether
it could be repaired with better parameters. It could not. Chamberland
is a fully dimensioned, chronic, decades scale model. Vaughan is an
explicitly non-dimensional, acute, days scale model fit to patient
cytokine data. The splice had also reused Vaughan's algebraic resting
microglia quantity as if it were Chamberland's dynamical microglia
state, a structural mismatch beyond parameter values. Decision: treat
Chamberland and Vaughan as two independent case studies rather than
force a merge.

## 5. Vaughan provenance audit

Vaughan's 52 parameters were found to be fit separately per patient
outcome cluster through repeated randomized optimization, not a single
canonical set. The paper's only public numeric supplement covers
ranges and averages for a subset of parameters flagged as most
different between cluster pairs, not a complete parameter vector for
any cluster. Initial conditions were never published at all. Before
using the Helmy human cohort as an independent check downstream, its
independence from Vaughan's own patient cohort was verified directly:
different institution, different country, different biospecimen,
different investigators, no shared cohort identifiers.

## 6. Sampling protocol bugs

While designing the exploratory parameter sampling protocol, an early
version of the D0 mixture multiplied a nonzero draw against a zero
anchor value, which would have stayed zero on every draw. Caught before
implementation. A wording issue was also caught: Hill exponent sampling
ranges had been described as biologically plausible with no source
actually establishing that, corrected to an explicit exploratory
mathematical range. Retrieving the actual supplementary numeric file,
rather than relying on the paper's prose summary of it, surfaced a
ninth constrained parameter the prose alone had not named, and
confirmed the published ranges were genuine per cluster marginal
distributions rather than artifacts of how they had been tabulated.

## 7. Harness verification

The ensemble sampling harness was smoke tested before generating the
production ensemble. An early test run showed 64 out of 64 samples
passing admissibility, which only proves the bookkeeping works, not
that the rejection logic itself fires. A deliberately pathological
parameter set was constructed and pushed through the filter, and was
correctly rejected with a specific, traceable reason, confirming the
rejection path is real.

## 8. Normalization audit

Applying the graph construction method to the frozen ensemble surfaced
one node whose mean weighted in-degree was inflated by roughly six
orders of magnitude relative to every other node. Traced to a specific
mechanism: a state that sits near its own equilibrium for most of its
simulated trajectory produces a near zero derivative scale, which the
existing normalization divides by, amplifying floating point level
differences into large apparent effects. Confirmed stable across all
five simulated timepoints. Confirmed as a property of the normalization
method itself, not a bug, by reproducing comparable divergence when
perturbing the same configuration by a tiny amount and rerunning the
unmodified method. Documented as a known limitation of quantile
derivative normalization near trajectory wide equilibrium, rather than
patched by raising the floor, since Vaughan's non-dimensional scale
gives no principled value to raise it to.

## 9. Directional axis correction

A synthetic sanity check, run specifically because a reported network
statistic looked internally inconsistent, revealed that every
computation labeled in-degree throughout the network analysis had
actually summed the array's axes in the direction corresponding to
out-degree. Caught before any conclusion was drawn from the mislabeled
statistic. Verified with a minimal reproducible synthetic example
before touching any real data. Every downstream degree computation was
rewritten with explicit, unambiguous function names, weighted in,
weighted out, topology in, topology out, so the same class of error
cannot recur silently. Recomputing the corrected statistics changed
the reported finding in a meaningful way: the original claim of a
single dominant hub became a more specific and more interesting
finding, one node acting as a persistent receiver throughout the
simulated window, a different node acting as an early broadcaster
whose reach declines over the same window.

## 10. Packaging discipline

Before generating final figures, an inventory of already completed
work found that several of the project's strongest results existed
only as console output from prior work sessions, never saved to a
file. Every one of these was recovered by rerunning the exact original
code against the exact frozen data, and checked against the previously
reported numbers before any figure was built from the recovery. Two
further small bugs were caught at this stage: a string slicing
collision that mislabeled which data points in one figure were
censored, and an imprecise summary claim that described two data
points as matching when one was in fact an exact tie rather than a
clean match. Both were corrected before the figures were finalized.

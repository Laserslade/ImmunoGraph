# Vaughan Case Study Provenance

## Source model

Vaughan, L.E., Ranganathan, P.R., Kumar, R.G., Wagner, A.K., Rubin,
J.E. (2018). A mathematical model of neuroinflammation in severe
clinical traumatic brain injury. Journal of Neuroinflammation.

## What is and is not recoverable

Vaughan's 52 parameters, 45 rate and shape parameters plus 7 initial
conditions, were fit separately for three patient outcome clusters
through repeated, randomized Nelder-Mead optimization, producing a 100
member fitted ensemble per cluster. None of these full parameter
vectors or ensembles are public.

The paper's only numeric supplement gives ranges and averages for the
subset of parameters found to differ significantly between cluster
pairs, not a complete vector for any single cluster. This subset
covers 9 of the 45 parameters: kpn1, kpn12, mu_M1, mu_M2, mu_n10,
mu_n12, a_inf2, hn, and vn. hn is continuous in the source model, not
integer valued as an earlier draft of this project's sampling protocol
assumed before the supplement was directly consulted. Initial
conditions have no published numeric support at all, for any
parameter.

Given this, this case study is treated as a structural and parameter
uncertainty case study, not a quantitative reconstruction of Vaughan's
fitted patients. See `case_study_vaughan/vaughan_sampling_protocol.csv`
for the full parameter by parameter sampling rule, and
`vaughan_additional_file1_extraction.csv` for the source preserved
extraction of the nine published cluster ranges.

## Independence of the Helmy cohort

Before using the Helmy 2011 cohort as an independent check in V5, its
independence from Vaughan's own patient cohort was verified directly.
Different institution, Cambridge versus Pittsburgh. Different country.
Different biospecimen, cerebral microdialysate and paired plasma
versus cerebrospinal fluid. Different investigators. No shared cohort
identifiers found in either paper.

## Bugs found during construction

An early version of the D0 initial condition sampling mixture
multiplied its nonzero draw against a zero valued anchor, which would
have produced zero on every draw regardless of the random component.
Caught before implementation, fixed with a separate, explicit positive
anchor value.

The exploratory Hill exponent sampling range had been described as
biologically plausible in early drafts. No source establishes that.
Corrected to explicit exploratory mathematical range wording.

The ensemble admissibility filter was smoke tested with a normal
sample batch that produced zero rejections, which only confirms
bookkeeping, not that the rejection logic fires. A deliberately
pathological parameter set was constructed and pushed through the
filter separately, and was correctly rejected with a specific,
traceable reason.

The in-degree and out-degree axis convention was found to be reversed
throughout an early version of the network analysis, caught by a
synthetic sanity check before any conclusion was drawn from it. See
`docs/DEVELOPMENT_LOG.md` section 9 for detail. Fixed by introducing
explicit, unambiguous degree functions in `reference/vaughan_degree_utils.py`.

## Files

- `reference/vaughan_reference.py`: structural implementation.
- `reference/vaughan_degree_utils.py`: explicit directional degree
  functions, introduced after the axis bug above was found.
- `reference/vaughan_ensemble_harness.py`,
  `reference/vaughan_immunograph_ensemble.py`: sampling and graph
  construction.
- `vaughan_sampling_protocol.csv`,
  `vaughan_additional_file1_extraction.csv`: provenance tables.
- `build_graph_cache.py`, `normalization_audit.py`,
  `directional_hub_dynamics.py`, `compute_v4_outcomes.py`,
  `v4_full_analysis.py`, `v5_helmy_comparison.py`: analysis scripts,
  run in that order.
- `figures/make_figure_v1.py` through `make_figure_v6.py`: figure
  generation, run after the corresponding analysis script.

## Normalization limitation

See `LIMITATIONS.md` for the near equilibrium normalization limitation
found during this case study's construction.

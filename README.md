# ImmunoGraph

ImmunoGraph converts a mechanistic ODE model into a time varying signed
graph, using local sensitivity, the model's own Jacobian, as edge
weight, so that the structure of a biological system's internal
influence, not just its raw trajectories, can be analyzed with network
methods.

## Status

This project underwent a major methodological correction. An earlier
composite biological model, a Chamberland aging backbone plus a
Vaughan derived TBI module, was found to be uncalibrated, with a day
versus year time unit error, and its two component models were found
to be architecturally incompatible to splice together. That pipeline
has been retired and is retained under `legacy/` for provenance only.
Full account in `docs/DEVELOPMENT_LOG.md`.

The current, validated project consists of:

- `immunograph_core/`, the graph construction method itself, model
  agnostic.
- `immunograph_validation_suite/`, ground truth recovery, ablations,
  and null controls on an abstract synthetic system.
- `case_study_chamberland/`, a clean room reconstruction of
  Chamberland et al. 2024's Alzheimer's disease model, verified
  against the original authors' own implementation through three
  gated checks.
- `case_study_vaughan/`, a structurally faithful implementation of
  Vaughan et al. 2018's acute TBI neuroinflammation model, treated as
  an exploratory parameter uncertainty case study since its fitted
  parameters are not fully public, with results checked against an
  independent held out human cohort, Helmy et al. 2011.

## Validated results, summary

Chamberland: 480 of 480 static equivalence checks pass against the
reference implementation. 3 of 4 demographic configurations reproduce
numerically at machine precision, the fourth, APOE4 positive male,
shows a documented, independently confirmed numerical sensitivity in
the source model itself. 3 of 3 published behavior criteria reproduce.

Vaughan: an exploratory ensemble of 2041 accepted parameterizations,
sampled within published support where available. Parameters
independently identified by the source authors as clinically
discriminating show significant enrichment for association with
ImmunoGraph network outcomes, 41.7 percent versus 24.0 percent,
hypergeometric p equals 0.0026. 58.4 percent of the frozen ensemble
reproduces the exact cytokine peak ordering observed in an independent
human cohort never used in constructing the ensemble, against a 16.7
percent chance baseline.

A documented limitation: the network sensitivity normalization used
throughout becomes ill conditioned for states that remain near
equilibrium across most of their trajectory. Full detail in
`LIMITATIONS.md`.

## Repository structure

```
immunograph_core/               shared graph construction method
immunograph_validation_suite/   abstract system ground truth validation
case_study_chamberland/         Chamberland reconstruction, gates, figures
case_study_vaughan/             Vaughan case study, ensemble, figures
legacy/                         retired original pipeline, provenance only
docs/                           development log, data provenance
tests/                          regression test suite
```

## Reproducing results

Each case study folder's scripts are meant to be run in the order
listed in that folder's `PROVENANCE.md`. Every script resolves its own
paths relative to its own location, so the repository can be run from
any clone location. Dependencies are pinned in `requirements.txt`.

## Source models and data

This project builds on published mechanistic models and does not
claim authorship of them.

- Chamberland, E., Moravveji, S., Doyon, N., Duchesne, S. (2024). A
  computational model of Alzheimer's disease at the nano, micro, and
  macroscales. Frontiers in Neuroinformatics.
- Moravveji, S., Sadia, H., Doyon, N., Duchesne, S. (2025). Sensitivity
  analysis of a mathematical model of Alzheimer's disease progression
  unveils important causal pathways. Frontiers in Neuroinformatics.
- Vaughan, L.E., Ranganathan, P.R., Kumar, R.G., Wagner, A.K., Rubin,
  J.E. (2018). A mathematical model of neuroinflammation in severe
  clinical traumatic brain injury. Journal of Neuroinflammation.
- Helmy, A., Carpenter, K.L.H., Menon, D.K., Pickard, J.D., Hutchinson,
  P.J.A. (2011). The cytokine response to human traumatic brain
  injury: temporal profiles and evidence for cerebral parenchymal
  production. Journal of Cerebral Blood Flow and Metabolism.

Dataset licensing and redistribution status documented in
`docs/DATA_PROVENANCE.md`.

## Limitations

See `LIMITATIONS.md`.

## Development history

See `docs/DEVELOPMENT_LOG.md` and `CHANGELOG.md`.

## AI assistance disclosure

Satya Thavanesh Yalla used AI assistance (Claude) in the development of this project's code, verification scripts, and analysis across multiple sessions. All core numerical results have been independently verified by the author: Gates 1–3 verification against the original authors' implementations (Chamberland), an exploratory constrained parameter ensemble frozen before external comparison (Vaughan), a published parameter consistency analysis across the full parameter and outcome test set (V4), and a held-out comparison against an independent human TBI cohort (V5). Specific errors identified and corrected during this process including a day/year unit-scale error, a missing square term and an incorrect functional form in the original reconstruction, and an axis-convention mislabeling caught by a synthetic sanity check are documented in full in CHANGELOG.md and docs/DEVELOPMENT_LOG.md in the code repository.

## License

MIT, see `LICENSE`. Third party datasets are covered separately, see
`docs/DATA_PROVENANCE.md`.

## Author

Satya Thavanesh Yalla

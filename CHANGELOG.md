# Changelog

## Unreleased

### Added
- Gate 1, 2, and 3 verification of the Chamberland clean room
  reconstruction against the original authors' own implementation.
- Vaughan structural implementation with an exploratory, published
  support constrained parameter ensemble, frozen before external
  comparison.
- V4 published parameter consistency analysis across the full
  parameter and outcome test set.
- V5 held out comparison against an independent human TBI cohort.
- Shared graph construction core used by both case studies.
- Development log, limitations document, and data provenance
  document.

### Changed
- Retired the Chamberland and Vaughan composite splice. Treated as two
  independent case studies rather than one merged model.
- Corrected in-degree and out-degree axis convention throughout the
  Vaughan network analysis. Previously mislabeled, caught by a
  synthetic sanity check before any conclusion was drawn from it.
- Replaced biologically plausible wording around exploratory Hill
  exponent sampling ranges with explicit exploratory mathematical
  range wording, since no source establishes biological plausibility
  for those ranges.

### Fixed
- Day versus year time unit error in the original Chamberland
  reconstruction, a three order of magnitude timescale mismatch.
- Missing square term in the amyloid oligomer to plaque conversion
  equation in the original reconstruction.
- Incorrect functional form for the neuron death hazard term in the
  original reconstruction.
- Zero anchor bug in the initial D0 sampling mixture design, caught
  before implementation.
- String slicing bug that mislabeled which Helmy cohort data points
  were left censored in the V5 validation figure.

### Deprecated
- Original phase1 through phase10 pipeline, moved to `legacy/`.
  Retained for provenance only, not part of current validated results.

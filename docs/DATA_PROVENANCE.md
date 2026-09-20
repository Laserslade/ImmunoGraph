# Data Provenance

## External datasets referenced by this project

### Helmy et al. 2011 (used in Vaughan case study V5)

Source: Helmy, A., Carpenter, K.L.H., Menon, D.K., Pickard, J.D.,
Hutchinson, P.J.A. (2011). The cytokine response to human traumatic
brain injury: temporal profiles and evidence for cerebral parenchymal
production. Journal of Cerebral Blood Flow and Metabolism.

What is used: a small extracted subset of patient level peak cytokine
timing for three cytokines, IL-1, IL-12, and IL-10, drawn from the
paper's own supplementary material.

Licensing and redistribution status: not yet confirmed. Do not commit
the extracted subset to this repository until this is resolved.
`case_study_vaughan/v5_helmy_comparison.py` should either read from a
locally provided copy the user fetches themselves from the original
paper's supplementary material, or from a redistributed copy only once
permission or an applicable license has been confirmed.

## Model source code and parameters

### Chamberland reference implementation

Source: parameters, equations, and initial condition logic in
`case_study_chamberland/reference/authors_orig/` are the original
model authors' own released code, accompanying Moravveji et al. 2025,
copied unmodified and used only as a verification oracle. Not modified
in any way by this project. See that paper and its associated
repository for the original license terms.

### Vaughan model structure

The equations implemented in `case_study_vaughan/reference/vaughan_reference.py`
are a structurally faithful reimplementation of the equations published
in Vaughan et al. 2018, written independently from the published paper
text, not copied from any release of the original authors' own code,
since no such release was located. Parameter values used in this
project's exploratory ensemble are not the original authors' fitted
values, see `case_study_vaughan/PROVENANCE.md` for what is and is not
recoverable.

## Datasets referenced elsewhere in earlier project phases but not
## currently part of the validated case studies

iPSC microglia cytokine data and mouse TBI transcriptomic data were
sourced and structured during earlier phases of this project for a
planned external validation experiment that was later superseded by
the Vaughan V5 comparison against Helmy. Their licensing status was
never fully resolved. Do not redistribute derived files from these
sources in this repository until that is checked, regardless of
whether they appear in any historical branch or the legacy folder.

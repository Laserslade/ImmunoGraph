"""
models/composite_backbone.py

Phase 3 of the Neuroimmune Graph Project: the first composite-model
composition step.

Adds the missing cytokine detail (IL-1beta, IL-12, IL-4) from:

    Vaughan, L.E., Ranganathan, P.R., Kumar, R.G., Wagner, A.K., and
    Rubin, J.E. (2018). "A mathematical model of neuroinflammation in
    severe clinical traumatic brain injury." Journal of Neuroinflammation,
    15:345. Equations (3)-(9) reproduced faithfully below.

...to the Chamberland backbone (models/chamberland_backbone.py), following
the module-composition strategy agreed in the project findings log,
Section 3.6: pick one base model, and add missing components as a module
that plugs into a *specific existing compartment* -- not a wholesale
merge of two full systems.

-------------------------------------------------------------------------
HOW THE SEAM IS HANDLED (the part that makes this a composition, not a
"mash")
-------------------------------------------------------------------------
Two of the TBI model's variables already exist in the Chamberland
backbone, and are NOT duplicated:

  - TBI's resting microglia (mr), M1-like microglia (M1), and M2-like
    microglia (M2) map directly onto Chamberland's M_NA, M_pro, and
    M_anti. No new microglia pool is created. The TBI model's IL-1beta/
    IL-12/IL-4-driven activation and polarization terms (Rm1, Rm2, Rms)
    are added as EXTRA flux on top of Chamberland's existing Fo/ABo/TGF-
    beta/TNF-alpha-driven flux for those same three state variables.

  - TBI's IL-10 (IL10) is the same molecule as Chamberland's I10. No new
    IL-10 pool is created. The TBI model's Th2/M2-driven IL-10 production
    terms (Rt, Ra) are added as EXTRA production on top of Chamberland's
    existing microglia/macrophage-driven I10 production, sharing the same
    single decay term.

Genuinely new state variables (no Chamberland equivalent): IL-1beta
(IL1), IL-12 (IL12), IL-4 (IL4).

TBI's "resting microglia quasi-steady-state" and "Th2 cell" abstractions,
and its "tissue damage" variable (D), are deliberately NOT imported --
they are TBI-injury-specific constructs outside this project's target
variable list, and Chamberland already has an explicit M_NA state (no
quasi-steady-state assumption needed) and no analogous damage variable.
This is a deliberate scope decision, not an oversight.

-------------------------------------------------------------------------
SEAM VALIDATION (Phase 3 definition of done)
-------------------------------------------------------------------------
The composite derivative is built as:

    composite_dydt = [chamberland_dydt (19,) + seam_correction (19,),
                       new_module_dydt (3,)]

If every TBI-module coupling constant is set to zero, seam_correction is
identically zero and the composite system's first 19 equations reduce
EXACTLY to the pure Chamberland backbone. This is checked automatically
in phase3_composite_model.py as the seam validation step -- if that check
fails, the composition has changed the base model's behavior, which
would defeat the purpose of module-based composition.

-------------------------------------------------------------------------
PARAMETER VALUES
-------------------------------------------------------------------------
As with Phase 1, the TBI paper's exact numeric parameter values (52
parameters, Table 1) were not retrievable as a machine-readable table
from available sources. The equation *structure* below is exact
(reproduced directly from the paper's Eqs. 3-9). Numeric values are
placeholders, flagged the same way as Phase 1's DEFAULT_PARAMS.
"""

import numpy as np

from models.chamberland_backbone import DEFAULT_PARAMS as CHAMBERLAND_PARAMS
from models.chamberland_backbone import N_STATES as N_CHAMBERLAND_STATES
from models.chamberland_backbone import STATE_NAMES as CHAMBERLAND_STATE_NAMES
from models.chamberland_backbone import _mm, chamberland_rhs, insulin  # noqa: F401

NEW_STATE_NAMES = ["IL1", "IL12", "IL4"]
STATE_NAMES = CHAMBERLAND_STATE_NAMES + NEW_STATE_NAMES
N_STATES = len(STATE_NAMES)

# Index of the shared/seam variables within the 19-variable Chamberland
# sub-vector, used to add extra flux at the right positions.
_IDX = {name: i for i, name in enumerate(CHAMBERLAND_STATE_NAMES)}

# -------------------------------------------------------------------------
# TBI module parameters (PLACEHOLDER -- see module docstring)
# -------------------------------------------------------------------------
TBI_MODULE_PARAMS = {
    # Rm1 / Rm2: resting-microglia activation cues
    "kn1": 1.0, "kn12": 1.0, "xn": 2.0, "bn": 1.0, "a_inf1": 1.0,
    "kn4": 1.0, "kn10": 1.0, "zn": 2.0, "yn": 1.0,

    # Rms: M1 -> M2 polarization by IL-4/IL-10
    "tau_n4": 1.0, "tau_n10": 1.0, "gn": 2.0, "mn": 1.0,

    # Rp: pro-inflammatory cytokine release by M1
    "kM1base": 0.05, "hn": 2.0, "vn": 1.0,
    "kpn1": 0.2, "mu_n1": 0.3,
    "kpn12": 0.2, "mu_n12": 0.3,

    # Rt: Th2-mediated IL-4/IL-10 induction
    "ktbase": 0.02, "ktn12": 0.5, "cn": 2.0, "rn": 1.0, "a_inf2": 1.0,
    "ktn10": 0.1, "ktn4": 0.1,

    # Ra: anti-inflammatory cytokine release by M2
    "kM2base": 0.05, "kc4": 1.0, "qn": 2.0, "wn": 1.0,
    "kpn10": 0.1, "kpn4": 0.1,
    "mu_n4": 0.3,

    # Coupling strength of the module's microglia flux relative to
    # Chamberland's existing flux, in case the module needs to be
    # dialed down/up without touching every rate constant individually.
    # 1.0 = full strength, 0.0 = module fully disabled (seam validation).
    "module_coupling": 1.0,
}


def default_params():
    """Combined parameter dict: Chamberland backbone + TBI module."""
    params = dict(CHAMBERLAND_PARAMS)
    params.update(TBI_MODULE_PARAMS)
    return params


def composite_rhs(t, y, params=None):
    """
    Right-hand side of the composite (Chamberland + TBI cytokine module)
    system, 22 state variables total.

    y[:19]  -- Chamberland backbone states, same order as
               chamberland_backbone.STATE_NAMES
    y[19:22] -- [IL1, IL12, IL4], the new module states
    """
    if params is None:
        params = default_params()
    p = params
    coupling = p.get("module_coupling", 1.0)

    y_chamberland = y[:N_CHAMBERLAND_STATES]
    IL1, IL12, IL4 = y[N_CHAMBERLAND_STATES:N_CHAMBERLAND_STATES + 3]

    # Chamberland's own dynamics, entirely unmodified.
    base_dydt = chamberland_rhs(t, y_chamberland, params)

    # Seam variables read from the shared state vector.
    M_NA = y_chamberland[_IDX["M_NA"]]
    M_pro = y_chamberland[_IDX["M_pro"]]
    M_anti = y_chamberland[_IDX["M_anti"]]
    I10 = y_chamberland[_IDX["I10"]]

    # --- TBI model helper terms (Eqs. from Vaughan et al. 2018) ---
    def hill(x, k, n):
        xk = x ** n
        return xk / (k ** n + xk) if (x >= 0) else 0.0

    activation_signal = p["kn1"] * IL1 + p["kn12"] * IL12
    inhibition_signal_sq = (1 + ((I10 + IL4) / p["a_inf1"]) ** 2)
    Rm1 = hill(activation_signal, p["bn"], p["xn"]) / inhibition_signal_sq

    m2_signal = p["kn4"] * IL4 + p["kn10"] * I10
    Rm2 = hill(m2_signal, p["yn"], p["zn"])

    rms_signal = p["tau_n4"] * IL4 + p["tau_n10"] * I10
    Rms = hill(rms_signal, p["mn"], p["gn"])

    Rp_numer = p["kM1base"] * M_pro + M_pro * hill(IL12, p["vn"], p["hn"])
    Rp = Rp_numer / inhibition_signal_sq

    rt_inhibition_sq = (1 + (I10 / p["a_inf2"]) ** 2)
    Rt_numer = p["ktbase"] + hill(IL4 + p["ktn12"] * IL12, p["rn"], p["cn"])
    Rt = Rt_numer / rt_inhibition_sq

    Ra_numer = p["kM2base"] * M_anti + M_anti * hill(p["kc4"] * IL4, p["wn"], p["qn"])
    Ra = Ra_numer / rt_inhibition_sq

    # --- Seam corrections: EXTRA flux added to Chamberland's existing
    # equations for M_NA, M_pro, M_anti, and I10 (see module docstring).
    seam = np.zeros(N_CHAMBERLAND_STATES)
    activation_flux = coupling * (Rm1 + Rm2) * M_NA
    seam[_IDX["M_NA"]] += -activation_flux
    seam[_IDX["M_pro"]] += coupling * Rm1 * M_NA - coupling * Rms * M_pro
    seam[_IDX["M_anti"]] += coupling * Rm2 * M_NA + coupling * Rms * M_pro
    seam[_IDX["I10"]] += coupling * (p["ktn10"] * Rt + p["kpn10"] * Ra)

    composite_base_dydt = base_dydt + seam

    # --- New module states: IL-1beta, IL-12, IL-4 ---
    dIL1 = coupling * p["kpn1"] * Rp - p["mu_n1"] * IL1
    dIL12 = coupling * p["kpn12"] * Rp - p["mu_n12"] * IL12
    dIL4 = coupling * (p["ktn4"] * Rt + p["kpn4"] * Ra) - p["mu_n4"] * IL4

    return np.concatenate([composite_base_dydt, [dIL1, dIL12, dIL4]])


def default_initial_state():
    """Chamberland's default initial state, extended with zero for the
    three new cytokines (no IL-1beta/IL-12/IL-4 at simulated age 30)."""
    from models.chamberland_backbone import default_initial_state as chamberland_y0
    y0 = np.concatenate([chamberland_y0(), np.zeros(3)])
    return y0

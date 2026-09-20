"""
vaughan_reference.py

Faithful structural implementation of the acute-TBI neuroinflammation
model from:

    Vaughan, L.E., Ranganathan, P.R., Kumar, R.G., Wagner, A.K., Rubin, J.E.
    (2018). "A mathematical model of neuroinflammation in severe clinical
    traumatic brain injury." J Neuroinflammation 15:345.

STATUS AND SCOPE (read before using)
-------------------------------------------------------------------------
This is a SECOND, INDEPENDENT case study for ImmunoGraph, not part of
the Chamberland composite. There is no splice with chamberland_reference.py
and none should be added (see project decision log).

This module implements Vaughan's published EQUATION STRUCTURE exactly.
It does NOT reproduce any specific patient cluster's fitted behavior,
because that information is not publicly available:

  - The paper fit all 52 parameters (45 rate/shape parameters plus
    initial conditions for M1, M2, IL1, IL12, IL10, IL4, D) separately
    per patient cluster (1, 2A, 2B) via Nelder-Mead, then built a
    100-member ensemble per cluster from randomized re-optimization.
    None of that -- full vectors or ensembles -- is public.
  - Additional file 1 (the paper's only released numeric supplement)
    gives ranges/averages for only the subset of parameters flagged as
    most dissimilar between cluster pairs (Table 4 criterion), not all
    52, and not as complete per-cluster vectors.
  - Initial conditions were also fit per cluster and are not published
    at all, for any parameter subset.

Given that, DEFAULT_PARAMS and DEFAULT_INITIAL_STATE below are explicitly
NON-CANONICAL placeholders -- reasonable enough to make the system run
and demonstrate ImmunoGraph's machinery on it, but NOT a claim about any
real patient or cluster. Every parameter is tagged in PARAMETER_STATUS
(see bottom of file) as one of:
    'definition_only'   -- biological meaning known (Table 1), no
                            published numeric value at all
    'partial_range'      -- appears in Additional file 1 for at least
                            one cluster pairwise comparison, so a
                            plausible range/average exists, but not a
                            complete per-cluster vector
No parameter is tagged 'fitted_value_available', because none are.

Time unit: Vaughan does not assign literal physical units to its rate
constants (explicitly disclaimed in the paper's own limitations
section), but the model was fit to data in 6-hour bins over 5 days
post-injury, so t is treated here as HOURS, spanning roughly 0-120.
This is a scale convention for simulation purposes, not a claim that
the parameters are dimensionally grounded the way Chamberland's are.

State vector (7 variables, matching the paper's Eqs. 3-9):
    0  M1    M1-like (pro-inflammatory) microglia   [arbitrary units]
    1  M2    M2-like (anti-inflammatory) microglia  [arbitrary units]
    2  IL1   IL-1beta                                [arbitrary units]
    3  IL12  IL-12                                    [arbitrary units]
    4  IL10  IL-10                                    [arbitrary units]
    5  IL4   IL-4                                     [arbitrary units]
    6  D     secondary tissue damage                  [arbitrary units]

mr (resting microglia) is NOT a state -- it is solved algebraically at
every timestep from Eq. 2, exactly as the source paper does. Reusing a
dynamical state for it (as the old composite_backbone.py did) would be
a structural deviation from the published model; this file deliberately
does not do that.
"""

import numpy as np

STATE_NAMES = ["M1", "M2", "IL1", "IL12", "IL10", "IL4", "D"]
N_STATES = len(STATE_NAMES)
IDX = {name: i for i, name in enumerate(STATE_NAMES)}


def _hill(numerator_terms, exponent, half_sat):
    """Generic Hill-function saturation: x^n / (h^n + x^n)."""
    x = numerator_terms
    return (x ** exponent) / (half_sat ** exponent + x ** exponent)


def resting_microglia(IL1, IL12, IL4, IL10, p):
    """
    Eq. 2: quasi-steady-state resting microglia. NOT a dynamical state --
    solved algebraically at every call, exactly as the source model does.
    """
    Rm1 = Rm1_term(IL1, IL12, IL4, IL10, p)
    Rm2 = Rm2_term(IL4, IL10, p)
    return p["s_mr"] / (Rm1 + Rm2 + p["mu_mr"])


def Rm1_term(IL1, IL12, IL4, IL10, p):
    """Eq. (Rm1): cytokine-driven M1 activation signal."""
    drive = p["kn1"] * IL1 + p["kn12"] * IL12
    numer = _hill(drive, p["xn"], p["bn"])
    denom = 1.0 + ((IL10 + IL4) / p["a_inf1"]) ** 2
    return numer / denom


def Rm2_term(IL4, IL10, p):
    """Eq. (Rm2): cytokine-driven M2 activation signal."""
    drive = p["kn4"] * IL4 + p["kn10"] * IL10
    return _hill(drive, p["zn"], p["yn"])


def Rms_term(IL4, IL10, p):
    """Eq. (Rms): M1 -> M2 polarization shift signal."""
    drive = p["tau_n4"] * IL4 + p["tau_n10"] * IL10
    return _hill(drive, p["gn"], p["mn"])


def Rp_term(M1, IL12, D, IL10, IL4, p):
    """
    Eq. (Rp): pro-inflammatory cytokine release by M1.
    Includes the tissue-damage feedback (kcd * D) inside the Hill drive --
    this term was dropped in the old composite_backbone.py and is
    restored here per the published equation.
    """
    drive = IL12 + p["kcd"] * D
    numer = p["kM1base"] * M1 + M1 * _hill(drive, p["hn"], p["vn"])
    denom = 1.0 + ((IL10 + IL4) / p["a_inf1"]) ** 2
    return numer / denom


def Rt_term(IL4, IL12, IL10, p):
    """Eq. (Rt): Th2-mediated IL-4/IL-10 production signal."""
    drive = IL4 + p["ktn12"] * IL12
    numer = p["ktbase"] + _hill(drive, p["cn"], p["rn"])
    denom = 1.0 + (IL10 / p["a_inf2"]) ** 2
    return numer / denom


def Ra_term(M2, IL4, IL10, p):
    """Eq. (Ra): M2-mediated anti-inflammatory cytokine production signal."""
    drive = p["kc4"] * IL4
    numer = p["kM2base"] * M2 + M2 * _hill(drive, p["qn"], p["wn"])
    denom = 1.0 + (IL10 / p["a_inf2"]) ** 2
    return numer / denom


def vaughan_rhs(t, y, p):
    """
    Right-hand side of the 7-equation Vaughan model, reproducing Eqs.
    2-9 of the source paper exactly, including D and the algebraic mr.
    t is in hours (see module docstring on time-unit convention).
    """
    M1, M2, IL1, IL12, IL10, IL4, D = y
    dydt = np.zeros(N_STATES)

    mr = resting_microglia(IL1, IL12, IL4, IL10, p)
    Rm1 = Rm1_term(IL1, IL12, IL4, IL10, p)
    Rm2 = Rm2_term(IL4, IL10, p)
    Rms = Rms_term(IL4, IL10, p)
    Rp = Rp_term(M1, IL12, D, IL10, IL4, p)
    Rt = Rt_term(IL4, IL12, IL10, p)
    Ra = Ra_term(M2, IL4, IL10, p)

    # Eq. 3: M1
    dydt[IDX["M1"]] = Rm1 * mr - Rms * M1 - p["mu_M1"] * M1

    # Eq. 4: M2
    dydt[IDX["M2"]] = Rm2 * mr + Rms * M1 - p["mu_M2"] * M2

    # Eq. 5: IL-1beta
    dydt[IDX["IL1"]] = p["kpn1"] * Rp - p["mu_n1"] * IL1

    # Eq. 6: IL-12
    dydt[IDX["IL12"]] = p["kpn12"] * Rp - p["mu_n12"] * IL12

    # Eq. 7: IL-10
    dydt[IDX["IL10"]] = p["ktn10"] * Rt + p["kpn10"] * Ra - p["mu_n10"] * IL10

    # Eq. 8: IL-4
    dydt[IDX["IL4"]] = p["ktn4"] * Rt + p["kpn4"] * Ra - p["mu_n4"] * IL4

    # Eq. 9: tissue damage D
    pro_drive = (p["alpha_n12"] * IL12 + p["alpha_n1"] * IL1) / (1.0 + (IL10 / p["a_inf2"]) ** 2)
    dydt[IDX["D"]] = pro_drive + p["r_M1"] * M1 - p["gamma_M1"] * M1 * D - p["gamma_M2"] * M2 * D

    return dydt


# -------------------------------------------------------------------------
# NON-CANONICAL default parameters -- see module docstring. Every key
# below is classified in PARAMETER_STATUS. Values are order-of-magnitude
# placeholders chosen only to make the system integrate to a stable,
# qualitatively sane trajectory (bounded, non-negative, settles within
# the ~120h simulation window) -- NOT fitted to any cluster or patient.
# -------------------------------------------------------------------------

DEFAULT_PARAMS = {
    # Resting microglia (Eq. 1-2)
    "s_mr": 1.0, "mu_mr": 0.05,
    # Rm1
    "kn1": 1.0, "kn12": 1.0, "xn": 2, "bn": 1.0, "a_inf1": 5.0,
    # Rm2
    "kn4": 1.0, "kn10": 1.0, "zn": 2, "yn": 1.0,
    # Rms
    "tau_n4": 1.0, "tau_n10": 1.0, "gn": 2, "mn": 1.0,
    # M1/M2 decay
    "mu_M1": 0.05, "mu_M2": 0.03,
    # Rp
    "kM1base": 0.1, "hn": 2, "vn": 1.0, "kcd": 0.1,
    # IL1/IL12 production & decay
    "kpn1": 1.0, "mu_n1": 0.2, "kpn12": 1.0, "mu_n12": 0.2,
    # Rt
    "ktbase": 0.1, "ktn12": 0.5, "cn": 2, "rn": 1.0, "a_inf2": 5.0,
    # Ra
    "kM2base": 0.1, "kc4": 1.0, "qn": 2, "wn": 1.0,
    # IL10/IL4 production & decay
    "ktn10": 0.5, "kpn10": 0.5, "mu_n10": 0.2,
    "ktn4": 0.5, "kpn4": 0.5, "mu_n4": 0.2,
    # Damage (Eq. 9)
    "alpha_n12": 0.05, "alpha_n1": 0.05, "r_M1": 0.02,
    "gamma_M1": 0.01, "gamma_M2": 0.02,
}

DEFAULT_INITIAL_STATE = np.array([
    2.0,   # M1  -- paper enforces M1>M2 pre-day-2 as a fit heuristic;
    0.5,   # M2     placeholder respects that ordering, nothing more.
    0.1,   # IL1
    0.1,   # IL12
    0.1,   # IL10
    0.1,   # IL4
    0.0,   # D
])

# -------------------------------------------------------------------------
# Parameter provenance map (V2). Every key in DEFAULT_PARAMS appears here.
# 'definition_only'  -> biological meaning from Table 1, no published number
# 'partial_range'    -> appears in Additional file 1 for >=1 cluster pair
#                       (range/average only, not a full vector)
# -------------------------------------------------------------------------
PARAMETER_STATUS = {k: "definition_only" for k in DEFAULT_PARAMS}

# 'partial_range': Additional file 1 is confirmed (by the paper's own
# text) to tabulate ranges/averages for the parameters behind Table 4 /
# Fig. 5, for at least one pairwise cluster comparison. The file itself
# hasn't been opened yet, so exact numeric ranges aren't in hand -- this
# tier only asserts that a real range exists to retrieve.
for _k in ["kpn1", "kpn12", "mu_M1", "mu_M2", "mu_n10", "mu_n12", "a_inf2", "hn", "vn"]:
    PARAMETER_STATUS[_k] = "partial_range"

# 'direction_only': the paper's discussion text states which cluster is
# higher/lower for these parameters, with no magnitude, independent of
# whether Additional file 1 also covers them. Where a parameter is both
# 'partial_range' (has a numeric range in Additional file 1) and
# discussed directionally in text, 'partial_range' is kept as the
# stronger classification above and is not overwritten here.
_DIRECTIONAL_ONLY = ["a_inf2"]  # a_inf2: text notes cluster1 > 2A directionally;
                                  # already 'partial_range' above via Table 4, kept as-is.
# No additional parameters carry directional-only information beyond
# those already captured by partial_range in this paper's reported
# comparisons -- the body text's qualitative claims (higher/lower)
# track the same parameter set Table 4 flags, they don't extend to any
# parameter lacking a Table 4 / Additional-file-1 entry.

INITIAL_CONDITION_STATUS = "unavailable_for_all_states"  # fit per cluster, never published

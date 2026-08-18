"""
models/chamberland_backbone.py

Phase 1 backbone model for the Neuroimmune Graph Project.

This module implements the mechanistic ODE system from:

    Chamberland, E., Moravveji, S., Doyon, N., and Duchesne, S. (2024).
    "A computational model of Alzheimer's disease at the nano, micro,
    and macroscales." Frontiers in Neuroinformatics, 18:1348113.

It was chosen as the project's base backbone because it is the only
published mechanistic model found (see project findings log, Section 3.8)
that already unifies neurons, astrocytes, microglia (three activation
states), macrophages (two activation states), and the core cytokines
TNF-alpha, TGF-beta, and IL-10 in a single coupled system.

-------------------------------------------------------------------------
IMPORTANT NOTE ON PARAMETER VALUES
-------------------------------------------------------------------------
The equation *structure* below (which variables interact, and in what
functional form -- linear production, Michaelis-Menten saturation, Hill-
type activation, etc.) is reconstructed faithfully from the mechanisms
described in the source paper and its companion sensitivity-analysis
appendix (which lists every parameter and its literature source, but not
its numeric value in machine-readable form).

The numeric parameter values in DEFAULT_PARAMS below are PLACEHOLDER
values, chosen to be biologically plausible in relative magnitude (e.g.
degradation rates faster than production rates where that's expected)
so that the system is stable and produces interpretable trajectories.
They are NOT yet calibrated against the original paper's Supplementary
Material A. Anyone using this module for anything beyond a structural /
qualitative demonstration should replace DEFAULT_PARAMS with the
original paper's calibrated values first.

This is flagged explicitly (rather than silently assumed) per the
project's Validation Strategy: the goal of Phase 1 is a qualitatively
sane, structurally faithful implementation -- not a quantitatively
calibrated one.
-------------------------------------------------------------------------

State vector (19 variables, in order):
    0  ABi     intracellular amyloid-beta
    1  ABm     extracellular amyloid-beta monomers
    2  ABo     extracellular amyloid-beta oligomers
    3  ABp     extracellular amyloid-beta plaques
    4  G       GSK-3
    5  tau     phosphorylated tau
    6  Fi      intracellular neurofibrillary tangles (NFTs)
    7  Fo      extracellular NFTs
    8  N       neuron density
    9  A       activated astrocytes
    10 M_NA    non-activated microglia
    11 M_pro   pro-inflammatory microglia
    12 M_anti  anti-inflammatory microglia
    13 Mh_pro  pro-inflammatory macrophages
    14 Mh_anti anti-inflammatory macrophages
    15 Tb      TGF-beta
    16 Ta      TNF-alpha
    17 I10     IL-10
    18 P       MCP-1
"""

import numpy as np

STATE_NAMES = [
    "ABi", "ABm", "ABo", "ABp", "G", "tau", "Fi", "Fo", "N", "A",
    "M_NA", "M_pro", "M_anti", "Mh_pro", "Mh_anti", "Tb", "Ta", "I10", "P",
]

N_STATES = len(STATE_NAMES)


# -------------------------------------------------------------------------
# Default parameters (PLACEHOLDER -- see module docstring)
# -------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # Demographic switches (0/1). AP = 1 means APOE4 carrier.
    "AP": 0.0,

    # Amyloid-beta
    "lam_ABi": 0.5, "delta_AP_ABi": 0.3, "d_ABi": 0.3,
    "lam_ABom": 0.4, "delta_AP_ABom": 0.3, "lam_AABom": 0.05,
    "kappa_ABom_ABoo": 0.02, "delta_AP_ABomo": 0.3, "d_ABom": 0.2,
    "kappa_ABoo_ABop": 0.05, "d_ABoo": 0.15,
    "d_Manti_ABop": 0.1, "d_Mhanti_ABop": 0.1, "K_ABop": 1.0,
    "delta_AP_dp": 0.3,

    # GSK-3 / tau / NFTs
    "Ins0": 1.0, "lam_InsG": 0.2, "d_G": 0.3, "G0": 1.0,
    "lam_tau": 0.3, "lam_Gtau": 0.2, "kappa_tauFi": 0.01, "d_tau": 0.25,
    "d_Fi": 0.2, "kappa_MFo": 0.15, "K_Manti_Fo": 1.0, "d_Fo": 0.2,

    # Neurons
    "d_FiN": 0.05, "K_Fi": 1.0, "d_TaN": 0.05, "K_Ta": 1.0, "K_I10": 1.0,

    # Astrocytes
    "A_max": 1.0, "kappa_TaA": 0.1, "kappa_ABopA": 0.1, "d_A": 0.1,

    # Microglia
    "d_Mpro": 0.15, "d_Manti": 0.15,
    "kappa_FoM": 0.2, "K_Fo": 1.0,
    "kappa_ABooM": 0.2, "K_ABoo": 1.0,
    "kappa_TbMpro": 0.2, "K_TbM": 1.0,
    "kappa_TaManti": 0.2, "K_TaM": 1.0,
    "K_TaAct": 1.0, "K_I10Act": 1.0,

    # Macrophages
    "kappa_PMhat": 0.2, "K_P": 1.0, "Mhat_max": 1.0,
    "kappa_TbMhpro": 0.2, "K_TbMh": 1.0,
    "kappa_TaMhanti": 0.2, "K_TaMh": 1.0,
    "d_Mhpro": 0.15, "d_Mhanti": 0.15,

    # Cytokines / chemokines
    "kappa_MantiTb": 0.1, "kappa_MhantiTb": 0.1, "d_Tb": 0.3,
    "kappa_MantiI10": 0.1, "kappa_MhantiI10": 0.1, "d_I10": 0.3,
    "kappa_MproTa": 0.1, "kappa_MhproTa": 0.1, "d_Ta": 0.3,
    "kappa_MproP": 0.1, "kappa_MhproP": 0.1, "kappa_AP_mcp": 0.1, "d_P": 0.3,
}


def insulin(t, params):
    """
    Age-dependent insulin concentration, normalized to Ins0 at age 30.
    Declines slowly with age, consistent with the source paper's framing
    of insulin resistance increasing GSK-3 activity over the lifespan.
    t is age in years.
    """
    Ins0 = params["Ins0"]
    return Ins0 * np.exp(-0.01 * max(t - 30.0, 0.0))


def _mm(x, K):
    """Michaelis-Menten saturation term: x / (K + x), safe for x >= 0."""
    return x / (K + x) if (x + K) > 0 else 0.0


def chamberland_rhs(t, y, params=None):
    """
    Right-hand side of the 19-equation Chamberland backbone system.

    Parameters
    ----------
    t : float
        Time (age, in years).
    y : array-like, shape (19,)
        State vector, in the order given by STATE_NAMES.
    params : dict, optional
        Parameter dictionary. Defaults to DEFAULT_PARAMS.

    Returns
    -------
    dydt : np.ndarray, shape (19,)
    """
    if params is None:
        params = DEFAULT_PARAMS

    p = params
    AP = p["AP"]

    (ABi, ABm, ABo, ABp, G, tau, Fi, Fo, N, A,
     M_NA, M_pro, M_anti, Mh_pro, Mh_anti, Tb, Ta, I10, P) = y

    # Neuron death rate (per-capita hazard), used to couple neuron loss
    # into the intracellular-to-extracellular transitions below.
    hazard = (p["d_FiN"] * _mm(Fi, p["K_Fi"])
              + p["d_TaN"] * _mm(Ta, p["K_Ta"]) * (p["K_I10"] / (p["K_I10"] + I10)))
    dN_death = hazard * N  # positive quantity = rate of neuron loss

    # --- Amyloid-beta ---
    dABi = (p["lam_ABi"] * (1 + p["delta_AP_ABi"] * AP)
            - p["d_ABi"] * ABi - dN_death * ABi / (N + 1e-9))

    dABm = (dN_death * ABi / (N + 1e-9)
            + p["lam_ABom"] * (1 + p["delta_AP_ABom"] * AP) * N
            + p["lam_AABom"] * (1 + p["delta_AP_ABom"] * AP) * A
            - p["kappa_ABom_ABoo"] * (1 + p["delta_AP_ABomo"] * AP) * ABm ** 2
            - p["d_ABom"] * ABm)

    dABo = (p["kappa_ABom_ABoo"] * (1 + p["delta_AP_ABomo"] * AP) * ABm ** 2
            - p["kappa_ABoo_ABop"] * ABo
            - p["d_ABoo"] * ABo)

    clearance_factor = 1.0 / (1 + p["delta_AP_dp"] * AP)
    dABp = (p["kappa_ABoo_ABop"] * ABo
            - clearance_factor * (p["d_Manti_ABop"] * M_anti + p["d_Mhanti_ABop"] * Mh_anti)
            * _mm(ABp, p["K_ABop"]))

    # --- GSK-3 / tau / NFTs ---
    Ins_t = insulin(t, p)
    dG = p["lam_InsG"] * (p["Ins0"] / max(Ins_t, 1e-9)) - p["d_G"] * G

    dtau = (p["lam_tau"] + p["lam_Gtau"] * max(G - p["G0"], 0.0)
            - p["kappa_tauFi"] * tau ** 2
            - dN_death * tau / (N + 1e-9)
            - p["d_tau"] * tau)

    dFi = (p["kappa_tauFi"] * tau ** 2 - p["d_Fi"] * Fi
           - dN_death * Fi / (N + 1e-9))

    dFo = (dN_death * Fi / (N + 1e-9)
           - p["kappa_MFo"] * M_anti * _mm(Fo, p["K_Manti_Fo"])
           - p["d_Fo"] * Fo)

    # --- Neurons ---
    dNdt = -hazard * N

    # --- Astrocytes ---
    dA = (p["kappa_TaA"] * Ta * (p["A_max"] - A)
          + p["kappa_ABopA"] * ABp * (p["A_max"] - A)
          - p["d_A"] * A)

    # --- Microglia ---
    M_activation = (p["kappa_FoM"] * _mm(Fo, p["K_Fo"])
                     + p["kappa_ABooM"] * _mm(ABo, p["K_ABoo"])) * M_NA

    eps_Ta = _mm(Ta, p["K_TaAct"])
    eps_I10 = _mm(I10, p["K_I10Act"])
    beta = eps_Ta / (eps_Ta + eps_I10 + 1e-9)

    dM_NA = -M_activation + p["d_Mpro"] * M_pro + p["d_Manti"] * M_anti

    conv_pro_to_anti = p["kappa_TbMpro"] * _mm(Tb, p["K_TbM"]) * M_pro
    conv_anti_to_pro = p["kappa_TaManti"] * _mm(Ta, p["K_TaM"]) * M_anti

    dM_pro = (beta * M_activation - conv_pro_to_anti + conv_anti_to_pro
              - p["d_Mpro"] * M_pro)
    dM_anti = ((1 - beta) * M_activation + conv_pro_to_anti - conv_anti_to_pro
               - p["d_Manti"] * M_anti)

    # --- Macrophages ---
    Mh_import = (p["kappa_PMhat"] * _mm(P, p["K_P"])
                 * max(p["Mhat_max"] - Mh_pro - Mh_anti, 0.0))

    conv_hpro_to_hanti = p["kappa_TbMhpro"] * _mm(Tb, p["K_TbMh"]) * Mh_pro
    conv_hanti_to_hpro = p["kappa_TaMhanti"] * _mm(Ta, p["K_TaMh"]) * Mh_anti

    dMh_pro = (beta * Mh_import - conv_hpro_to_hanti + conv_hanti_to_hpro
               - p["d_Mhpro"] * Mh_pro)
    dMh_anti = ((1 - beta) * Mh_import + conv_hpro_to_hanti - conv_hanti_to_hpro
                - p["d_Mhanti"] * Mh_anti)

    # --- Cytokines / chemokines ---
    dTb = (p["kappa_MantiTb"] * M_anti + p["kappa_MhantiTb"] * Mh_anti
           - p["d_Tb"] * Tb)
    dI10 = (p["kappa_MantiI10"] * M_anti + p["kappa_MhantiI10"] * Mh_anti
            - p["d_I10"] * I10)
    dTa = (p["kappa_MproTa"] * M_pro + p["kappa_MhproTa"] * Mh_pro
           - p["d_Ta"] * Ta)
    dP = (p["kappa_MproP"] * M_pro + p["kappa_MhproP"] * Mh_pro
          + p["kappa_AP_mcp"] * A - p["d_P"] * P)

    return np.array([
        dABi, dABm, dABo, dABp, dG, dtau, dFi, dFo, dNdt, dA,
        dM_NA, dM_pro, dM_anti, dMh_pro, dMh_anti, dTb, dTa, dI10, dP,
    ])


def default_initial_state():
    """
    A biologically-motivated starting point at age 30: healthy neuron and
    astrocyte density normalized to 1.0, all pathological/inflammatory
    variables near zero, small baseline cytokine tone.
    """
    y0 = np.zeros(N_STATES)
    y0[STATE_NAMES.index("N")] = 1.0       # neuron density, normalized
    y0[STATE_NAMES.index("A")] = 0.1       # small baseline astrocyte activation
    y0[STATE_NAMES.index("M_NA")] = 1.0    # microglia pool, normalized, resting
    y0[STATE_NAMES.index("Tb")] = 0.05
    y0[STATE_NAMES.index("I10")] = 0.05
    y0[STATE_NAMES.index("Ta")] = 0.02
    return y0

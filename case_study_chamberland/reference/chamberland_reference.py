"""
chamberland_reference.py

Clean-room reimplementation of the 19-state mechanistic ODE model from:

    Chamberland, E., Moravveji, S., Doyon, N., and Duchesne, S. (2024).
    "A computational model of Alzheimer's disease at the nano, micro,
    and macroscales." Frontiers in Neuroinformatics, 18:1348113.

Source of truth for this reimplementation: the authors' own public code,
released alongside the 2025 follow-up sensitivity-analysis paper
(Moravveji, Sadia, Doyon, Duchesne, Front. Neuroinform. 19:1590968),
at https://git.valeria.science/medics/models/sensitivity-analysis
(files parameters.py, InitialConditions.py, equations_SA.py, retrieved
2026-09-15). A copy of those files, unmodified, is kept alongside this
one under verification/authors_orig/ as the oracle this module is
checked against (see gate1_static_equivalence.py and
gate2_numerical_equivalence.py).

-------------------------------------------------------------------------
STATUS
-------------------------------------------------------------------------
This module supersedes models/chamberland_backbone.py, which was an
earlier, uncalibrated reconstruction built without access to the
authors' actual parameter values or code. That file is preserved
unmodified under models/legacy/ for provenance -- it is what the
original V2-V16 synthetic suite and the placeholder 22-node composite
model were built on, and should not be deleted. It should not be used
for anything going forward.

-------------------------------------------------------------------------
CANONICAL TIME UNIT: DAYS, NOT YEARS
-------------------------------------------------------------------------
This is the single most important correction versus the old
reconstruction. The authors' code takes t as age IN DAYS throughout
(confirmed directly in Parameters.Ins(t, S), Parameters.d_ABmo(t), and
every docstring in parameters.py). Rate constants are correspondingly
fast: d_Ta (TNF-alpha decay) ~= 54.84/day, d_Tb (TGF-beta decay)
~= 332.71/day. The old chamberland_backbone.py assumed t was age in
years with placeholder decay rates around 0.3/year -- roughly three
orders of magnitude off, in the wrong unit. This is a real, acute-ish
dynamical system, not a slow chronic one.

Simulate this model with t in days. A typical adult lifespan run spans
t = 30*365 to t = 80*365.

-------------------------------------------------------------------------
WHAT IS DELIBERATELY PRESERVED EXACTLY, EVEN WHERE IT LOOKS UNUSUAL
-------------------------------------------------------------------------
- The neuron-death hazard uses a SIGMOID for the F_i (intracellular NFT)
  term, with its own steepness parameter n=15, not a Michaelis-Menten
  term. The TNF-alpha term is Michaelis-Menten. These are genuinely
  different functional forms in the source model; do not "harmonize"
  them.
- The oligomer-to-plaque conversion term is (AB_o^o)**2, squared, not
  linear.
- The intracellular-to-extracellular transfer terms (for AB^i, G, tau,
  F_i) all multiply by abs(dydt[N]), the absolute value of the
  just-computed neuron death rate, not a separately-derived quantity.
  dydt[N] (neuron death) MUST be computed first in this function, before
  any of the terms that reference it.
- Ins(t, S) is called directly at each timestep (age-and-sex-dependent
  insulin concentration), not inverted through a placeholder Ins0 ratio.
- Several initial conditions are solved analytically from the model's
  own steady-state / quasi-equilibrium equations (quadratics for the
  amyloid and tau compartments, an explicit equilibrium for the plaque
  compartment), not zeros or round placeholder numbers. See
  default_initial_state() below.
- Several parameters differ by sex (N_0, A_0, G_0, lambda_InsG, K_Manti,
  Ins_0, and hence Ins(t, S)) and this dependency is preserved.
-------------------------------------------------------------------------

State vector (19 variables, in the authors' own order):
    0  ABi     intracellular amyloid-beta42 monomer      (g/mL)
    1  ABmo    extracellular amyloid-beta42 monomer       (g/mL)
    2  ABoo    extracellular amyloid-beta42 oligomer       (g/mL)
    3  ABpo    extracellular amyloid-beta42 plaque         (g/mL)
    4  G       GSK-3                                       (g/mL)
    5  tau     phosphorylated/hyperphosphorylated tau       (g/mL)
    6  Fi      intracellular neurofibrillary tangles (NFT)  (g/mL)
    7  Fo      extracellular NFT                            (g/mL)
    8  N       living neuron density                        (g/mL)
    9  A       activated astrocyte density                  (g/mL)
    10 M_NA    resting (non-activated) microglia            (g/mL)
    11 M_pro   pro-inflammatory microglia                   (g/mL)
    12 M_anti  anti-inflammatory microglia                  (g/mL)
    13 Mh_pro  pro-inflammatory macrophages                 (g/mL)
    14 Mh_anti anti-inflammatory macrophages                (g/mL)
    15 Tb      TGF-beta                                     (g/mL)
    16 I10     IL-10                                        (g/mL)
    17 Ta      TNF-alpha                                    (g/mL)
    18 P       MCP-1                                        (g/mL)

t is age in days throughout.
"""

import math
import numpy as np
from scipy.constants import Avogadro

STATE_NAMES = [
    "ABi", "ABmo", "ABoo", "ABpo", "G", "tau", "Fi", "Fo", "N", "A",
    "M_NA", "M_pro", "M_anti", "Mh_pro", "Mh_anti", "Tb", "I10", "Ta", "P",
]
N_STATES = len(STATE_NAMES)
IDX = {name: i for i, name in enumerate(STATE_NAMES)}


# -------------------------------------------------------------------------
# Age-and-sex-dependent forcing functions
# -------------------------------------------------------------------------

def d_ABmo(t):
    """
    Degradation rate of extracellular amyloid-beta42 monomer (/day).
    Half-life is linear with age: 3.8h at 30y to 9.4h at 80y.
    t: age in days.
    """
    halflife = (7.0 / 547500.0) * t + (11.0 / 600.0)
    return math.log(2) / halflife


def Ins(t, sex):
    """
    Brain insulin concentration (g/mL), age- and sex-dependent.
    Derived from peripheral insulin data (Bryhni et al. 2010), molar
    mass of insulin 5808 g/mol, assuming brain insulin is 10x lower
    than peripheral (Gray et al. 2014).
    t: age in days. sex: 0 = woman, 1 = man.
    """
    if sex == 0:
        return 0.1 * (-4.151e-15 * t + 3.460e-10)
    elif sex == 1:
        return 0.1 * (-4.257e-15 * t + 3.763e-10)
    raise ValueError("sex must be 0 (woman) or 1 (man)")


# -------------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------------

def default_params(sex, apoe4_status, xi=1.0):
    """
    Build the full parameter dict for a given sex and APOE4 status,
    matching the authors' Parameters class exactly.

    sex: 0 = woman, 1 = man.
    apoe4_status: 1 if APOE4 carrier, 0 otherwise.
    xi: 0 < xi <= 1, scales TotalMaxActivRateM (kappa_FoM, kappa_ABooM).
    """
    p = {}
    p["S"] = sex
    p["AP"] = apoe4_status
    p["xi"] = xi

    p["rho_cerveau"] = 1.03

    if sex == 0:
        p["N_0"] = 0.45
        p["A_0"] = 0.10
    else:
        p["N_0"] = 0.42
        p["A_0"] = 0.12

    M_ABm = 4514.0  # molar mass of AB42 monomer, g/mol
    m_Mhat = 4.990e-9  # mass of a macrophage/microglia, g

    # --- AB^i ---
    p["lambda_ABi"] = (3.63e-12 * 1e-3 * M_ABm * 86400) / 2
    p["delta_APi"] = (8373 - 2178) / (5631 - 783) - 1
    p["d_ABi"] = math.log(2) / (1.75 / 24)

    # --- AB_m^o ---
    p["lambda_ABmo"] = p["lambda_ABi"]
    p["delta_APm"] = p["delta_APi"]
    p["lambda_AABmo"] = (1.0 / 13.0) * p["lambda_ABmo"]
    kappa_ABmoABoo_min = 38 * 1000 * (1.0 / (2 * M_ABm)) * 86400
    p["kappa_ABmoABoo"] = kappa_ABmoABoo_min
    p["delta_APmo"] = 2.7 - 1

    # --- AB_o^o ---
    p["kappa_ABooABpo"] = (3.0 / 7.0) * 1e6 * 1000 / (2 * M_ABm)
    p["d_ABoo"] = 0.3e-3 * 86400

    # --- AB_p^o ---
    p["d_hatMantiABpo"] = math.log(2) / 3
    p["d_MantiABpo"] = math.log(2) / 0.85
    p["delta_APdp"] = (5.0 / 20.0) - 1
    p["K_ABpo"] = (1.11 + 0.53) / 527.4 / 1000

    # --- GSK-3 ---
    p["Ins_0"] = Ins(365 * 30, sex)
    p["d_G"] = math.log(2) / (41.0 / 24.0)
    if sex == 0:
        p["G_0"] = 1104e-12 * 47000 * p["rho_cerveau"]
    else:
        p["G_0"] = 310e-12 * 47000 * p["rho_cerveau"]
    p["lambda_InsG"] = p["d_G"] * p["G_0"]

    # --- tau ---
    p["lambda_tau"] = 26.3e-12
    p["lambda_Gtau"] = ((20.0 / 21.0) - (20.0 / 57.0)) * 1e-6 / 0.5 / 1000 / 1000 * 72500
    p["kappa_tauFi"] = (100.0 / 3.0) * 1e-6 / 19344 * 86400 * 1000
    p["d_tau"] = math.log(2) / 5.16

    # --- F_i ---
    p["d_Fi"] = 1e-2 * p["d_tau"]

    # --- F_o ---
    p["kappa_MFo"] = 0.4
    if sex == 0:
        p["K_Manti"] = (1.0 / 4.0) * 3.811e-2
    else:
        p["K_Manti"] = (1.0 / 4.0) * 3.193e-2
    p["d_Fo"] = (1.0 / 10.0) * p["d_tau"]

    # --- Neurons ---
    p["d_FiN"] = 1.0 / (2.51 * 365)
    p["K_Fi"] = 1.25e-10
    p["n"] = 15
    p["d_TaN"] = 7.26e-3 / 365 * 10
    p["K_Ta"] = 4.48e-12
    p["K_I10"] = 2.12e-12

    # --- Astrocytes ---
    p["A_max"] = p["A_0"]
    p["kappa_TaA"] = 0.92 / 100e-9
    p["kappa_ABpoA"] = (p["kappa_TaA"] * 2.24e-12) / (2 * p["K_ABpo"])
    p["d_A"] = 0.4

    # --- M_NA ---
    TotalMaxActivRateM = 0.20 * xi
    p["kappa_FoM"] = TotalMaxActivRateM * 2.0 / 3.0
    p["K_Fo"] = 11 * ((1000 * 72500) / Avogadro) * 1000
    p["kappa_ABooM"] = TotalMaxActivRateM * 1.0 / 3.0
    p["K_ABooM"] = 0.060 / 527.4 / 1000 * 1.5e2
    p["d_Mpro"] = 7.67e-4
    p["d_Manti"] = 7.67e-4

    # --- M_pro / M_anti polarization ---
    p["beta"] = 1.0
    p["K_TaAct"] = 2.24e-12
    p["K_I10Act"] = 2.12e-12
    p["kappa_TbMpro"] = 4.8
    p["K_TbM"] = 5.9e-11
    p["kappa_TaManti"] = 4.8
    p["K_TaM"] = 2.24e-12 * 2e2

    # --- Macrophages ---
    p["kappa_PMhat"] = 1.0 / 3.0 * 1e-2
    p["K_P"] = 6.23e-10 * 1e2
    p["Mhatmax"] = (830 * m_Mhat) / 2e-4
    p["kappa_TbMhatpro"] = 1.0 / (10.0 / 24.0)
    p["K_TbMhat"] = p["K_TbM"]
    p["kappa_TaMhatanti"] = 1.0 / (10.0 / 24.0)
    p["K_TaMhat"] = p["K_TaM"]
    p["d_Mhatpro"] = 7.67e-4
    p["d_Mhatanti"] = 7.67e-4

    # --- TGF-beta ---
    kappa_MhatantiTb_max = 10 * (47e-12 / 18 * 24) / (2e6 * m_Mhat)
    p["kappa_MhatantiTb"] = kappa_MhatantiTb_max
    p["kappa_MantiTb"] = p["kappa_MhatantiTb"]
    p["d_Tb"] = math.log(2) / (3.0 / 1440.0)

    # --- IL-10 ---
    p["kappa_MhatantiI10"] = 660e-12 / (2e5 * m_Mhat)
    p["kappa_MantiI10"] = p["kappa_MhatantiI10"]
    p["d_I10"] = math.log(2) / (3.556 / 24.0)

    # --- TNF-alpha ---
    p["kappa_MhatproTa"] = (1.5e-9 / 18 * 24) / (4e6 * m_Mhat)
    p["kappa_MproTa"] = p["kappa_MhatproTa"]
    p["d_Ta"] = math.log(2) / (18.2 / 1440.0)

    # --- MCP-1 ---
    p["kappa_MhatproP"] = 11e-9 / (2e6 * m_Mhat)
    p["kappa_MproP"] = p["kappa_MhatproP"]
    kappa_AP_min = (1.0 / 10.0) * p["kappa_MhatproP"]
    p["kappa_AP"] = kappa_AP_min
    p["d_P"] = math.log(2) / (3.0 / 24.0)

    return p


# -------------------------------------------------------------------------
# Initial conditions
# -------------------------------------------------------------------------

def default_initial_state(p, age_start_years=30):
    """
    Initial state vector at age_start_years, in g/mL.

    Several entries are solved analytically from the model's own
    quasi-steady-state relations (quadratics / equilibria), not chosen
    as arbitrary placeholders. This mirrors InitialConditions.py exactly.
    """
    age_start_days = age_start_years * 365
    y0 = np.zeros(N_STATES)

    # AB^i
    y0[IDX["ABi"]] = p["lambda_ABi"] * (1 + p["AP"] * p["delta_APi"]) / p["d_ABi"]

    # AB_m^o -- positive root of A*x^2 + B*x + C = 0
    A = p["kappa_ABmoABoo"] * (1 + p["AP"] * p["delta_APmo"])
    B = d_ABmo(age_start_days)
    C = -p["lambda_ABmo"] * (1 + p["AP"] * p["delta_APm"])
    y0[IDX["ABmo"]] = (-B + math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    # AB_o^o -- positive root
    A = p["kappa_ABooABpo"]
    B = p["d_ABoo"]
    C = -p["kappa_ABmoABoo"] * (1 + p["AP"] * p["delta_APmo"]) * (y0[IDX["ABmo"]] ** 2)
    y0[IDX["ABoo"]] = (-B + math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    # AB_p^o: computed later, at equilibrium, after microglia/macrophages

    # G
    y0[IDX["G"]] = p["G_0"]

    # tau -- positive root
    A = p["kappa_tauFi"]
    B = p["d_tau"]
    C = -(p["lambda_tau"] + p["lambda_Gtau"])
    y0[IDX["tau"]] = (-B + math.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    # F_i
    y0[IDX["Fi"]] = p["kappa_tauFi"] * (y0[IDX["tau"]] ** 2) / p["d_Fi"]

    # F_o
    y0[IDX["Fo"]] = 5e-17

    # N
    y0[IDX["N"]] = p["N_0"]

    # A (activated astrocytes) stays 0

    # M_pro, M_anti: fixed starting values
    y0[IDX["M_pro"]] = 1e-12
    y0[IDX["M_anti"]] = 1e-4

    # M_NA: resident pool minus the above two
    M_0 = 3.811e-2 if p["S"] == 0 else 3.193e-2
    y0[IDX["M_NA"]] = M_0 - (y0[IDX["M_pro"]] + y0[IDX["M_anti"]])

    # Mh_pro, Mh_anti stay 0

    # AB_p^o -- equilibrium, now that M_anti/Mh_anti are known
    Psi = p["kappa_ABooABpo"] * (y0[IDX["ABoo"]] ** 2)
    D = (p["d_MantiABpo"] * y0[IDX["M_anti"]] + p["d_hatMantiABpo"] * y0[IDX["Mh_anti"]]) * (1 + p["AP"] * p["delta_APdp"])
    y0[IDX["ABpo"]] = (Psi * p["K_ABpo"]) / (D - Psi)

    # Tb, I10, Ta, P: production/decay balance at t=0 given the above
    y0[IDX["Tb"]] = (p["kappa_MantiTb"] * y0[IDX["M_anti"]] + p["kappa_MhatantiTb"] * y0[IDX["Mh_anti"]]) / p["d_Tb"]
    y0[IDX["I10"]] = (p["kappa_MantiI10"] * y0[IDX["M_anti"]] + p["kappa_MhatantiI10"] * y0[IDX["Mh_anti"]]) / p["d_I10"]
    y0[IDX["Ta"]] = (p["kappa_MproTa"] * y0[IDX["M_pro"]] + p["kappa_MhatproTa"] * y0[IDX["Mh_pro"]]) / p["d_Ta"]
    y0[IDX["P"]] = (p["kappa_MproP"] * y0[IDX["M_pro"]] + p["kappa_MhatproP"] * y0[IDX["Mh_pro"]] + p["kappa_AP"] * y0[IDX["A"]]) / p["d_P"]

    return y0


# -------------------------------------------------------------------------
# Right-hand side
# -------------------------------------------------------------------------

def _mm(x, K):
    """Michaelis-Menten saturation term: x / (x + K)."""
    return x / (x + K)


def _sigmoid_death(Fi, K_Fi, n):
    """Sigmoid neuron-death term driven by intracellular NFTs."""
    return 1.0 / (1.0 + np.exp(-n * (Fi - K_Fi) / K_Fi))


def chamberland_rhs(t, y, p, ins_var=True):
    """
    Right-hand side of the 19-equation Chamberland backbone, reproducing
    equations_SA.py term for term. t is age in DAYS.

    ins_var: if True (default), insulin varies with age via Ins(t, S).
    If False, insulin is held at Ins_0 (matches the authors' InsVar flag).
    """
    (ABi, ABmo, ABoo, ABpo, G, tau, Fi, Fo, N, A,
     M_NA, M_pro, M_anti, Mh_pro, Mh_anti, Tb, I10, Ta, P) = y

    dydt = np.zeros(N_STATES)

    # Neuron death hazard -- computed FIRST; everything else references it.
    hazard = (p["d_FiN"] * _sigmoid_death(Fi, p["K_Fi"], p["n"])
              + p["d_TaN"] * _mm(Ta, p["K_Ta"]) * (1.0 / (1.0 + I10 / p["K_I10"])))
    dydt[IDX["N"]] = -hazard * N
    neuron_loss = abs(dydt[IDX["N"]])

    # AB^i
    dydt[IDX["ABi"]] = (p["lambda_ABi"] * (1 + p["AP"] * p["delta_APi"]) * (N / p["N_0"])
                         - p["d_ABi"] * ABi - (ABi / N) * neuron_loss)

    # AB_m^o
    dydt[IDX["ABmo"]] = (
        (ABi / N) * neuron_loss
        + p["lambda_ABmo"] * (1 + p["AP"] * p["delta_APm"]) * (N / p["N_0"])
        + p["lambda_AABmo"] * (A / p["A_0"])
        - p["kappa_ABmoABoo"] * (1 + p["AP"] * p["delta_APmo"]) * (ABmo ** 2)
        - d_ABmo(t) * ABmo
    )

    # AB_o^o
    dydt[IDX["ABoo"]] = (
        p["kappa_ABmoABoo"] * (1 + p["AP"] * p["delta_APmo"]) * (ABmo ** 2)
        - p["kappa_ABooABpo"] * (ABoo ** 2)
        - p["d_ABoo"] * ABoo
    )

    # AB_p^o
    dydt[IDX["ABpo"]] = (
        p["kappa_ABooABpo"] * (ABoo ** 2)
        - (p["d_MantiABpo"] * M_anti + p["d_hatMantiABpo"] * Mh_anti)
        * (1 + p["AP"] * p["delta_APdp"]) * _mm(ABpo, p["K_ABpo"])
    )

    # G (GSK-3)
    ins_t = Ins(t, p["S"]) if ins_var else p["Ins_0"]
    dydt[IDX["G"]] = (p["lambda_InsG"] * (p["Ins_0"] / ins_t) * (N / p["N_0"])
                       - p["d_G"] * G - (G / N) * neuron_loss)

    # tau
    dydt[IDX["tau"]] = (
        p["lambda_tau"] * (N / p["N_0"]) + p["lambda_Gtau"] * (G / p["G_0"])
        - p["kappa_tauFi"] * (tau ** 2) * (N / p["N_0"])
        - (tau / N) * neuron_loss - p["d_tau"] * tau
    )

    # F_i
    dydt[IDX["Fi"]] = (p["kappa_tauFi"] * (tau ** 2) * (N / p["N_0"])
                        - (Fi / N) * neuron_loss - p["d_Fi"] * Fi)

    # F_o
    dydt[IDX["Fo"]] = ((Fi / N) * neuron_loss
                        - p["kappa_MFo"] * _mm(M_anti, p["K_Manti"]) * Fo
                        - p["d_Fo"] * Fo)

    # A (astrocytes)
    dydt[IDX["A"]] = ((p["kappa_ABpoA"] * ABpo + p["kappa_TaA"] * Ta) * (p["A_max"] - A)
                       - p["d_A"] * A)

    # Microglia
    M_activ = (p["kappa_FoM"] * _mm(Fo, p["K_Fo"]) * M_NA
               + p["kappa_ABooM"] * _mm(ABoo, p["K_ABooM"]) * M_NA)

    dydt[IDX["M_NA"]] = p["d_Mpro"] * M_pro + p["d_Manti"] * M_anti - M_activ

    eps_Ta = _mm(Ta, p["K_TaAct"])
    eps_I10 = _mm(I10, p["K_I10Act"])
    denom_micro = p["beta"] * eps_Ta + eps_I10

    dydt[IDX["M_pro"]] = (
        (p["beta"] * eps_Ta / denom_micro) * M_activ
        - p["kappa_TbMpro"] * _mm(Tb, p["K_TbM"]) * M_pro
        + p["kappa_TaManti"] * _mm(Ta, p["K_TaM"]) * M_anti
        - p["d_Mpro"] * M_pro
    )
    dydt[IDX["M_anti"]] = (
        (eps_I10 / denom_micro) * M_activ
        + p["kappa_TbMpro"] * _mm(Tb, p["K_TbM"]) * M_pro
        - p["kappa_TaManti"] * _mm(Ta, p["K_TaM"]) * M_anti
        - p["d_Manti"] * M_anti
    )

    # Macrophages
    Mh_import = p["kappa_PMhat"] * _mm(P, p["K_P"]) * (p["Mhatmax"] - (Mh_pro + Mh_anti))

    dydt[IDX["Mh_pro"]] = (
        Mh_import * (p["beta"] * eps_Ta / denom_micro)
        - p["kappa_TbMhatpro"] * _mm(Tb, p["K_TbMhat"]) * Mh_pro
        + p["kappa_TaMhatanti"] * _mm(Ta, p["K_TaMhat"]) * Mh_anti
        - p["d_Mhatpro"] * Mh_pro
    )
    dydt[IDX["Mh_anti"]] = (
        Mh_import * (eps_I10 / denom_micro)
        + p["kappa_TbMhatpro"] * _mm(Tb, p["K_TbMhat"]) * Mh_pro
        - p["kappa_TaMhatanti"] * _mm(Ta, p["K_TaMhat"]) * Mh_anti
        - p["d_Mhatanti"] * Mh_anti
    )

    # Cytokines / chemokines
    dydt[IDX["Tb"]] = p["kappa_MantiTb"] * M_anti + p["kappa_MhatantiTb"] * Mh_anti - p["d_Tb"] * Tb
    dydt[IDX["I10"]] = p["kappa_MantiI10"] * M_anti + p["kappa_MhatantiI10"] * Mh_anti - p["d_I10"] * I10
    dydt[IDX["Ta"]] = p["kappa_MproTa"] * M_pro + p["kappa_MhatproTa"] * Mh_pro - p["d_Ta"] * Ta
    dydt[IDX["P"]] = (p["kappa_MproP"] * M_pro + p["kappa_MhatproP"] * Mh_pro
                       + p["kappa_AP"] * A - p["d_P"] * P)

    return dydt

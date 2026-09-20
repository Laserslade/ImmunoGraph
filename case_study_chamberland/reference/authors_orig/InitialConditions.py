"""File defining the function that creates the vector of initial conditions.

:Author: Éléonore Chamberland
:Creation date: September 9, 2022
:Last modified: August 27, 2023
"""

import math
import numpy as np


def InitialConditions(p, AgeStart=30):
    """Function that defines de vector of initial conditions of the model, in g/mL.

    :param p: Parameter class.
    :param AgeStart: Age at which the integration of the model starts.
    :return: y0 : A vector with the initial conditions.
    """
    y0 = np.zeros(19)

    """AB^i (Amyloid-beta monomer intracell.)"""
    y0[0] = p.lambda_ABi * (1 + p.AP * p.delta_APi) / p.d_ABi
    # APOE4+: 1.903101570888682e-10 ; APOE4-: 1.4893036990586492e-10

    """AB_m^o (Amyloid-beta monomer extracell.)"""
    A = p.kappa_ABmoABoo * (1 + p.AP * p.delta_APmo)
    B = p.d_ABmo(AgeStart * 365)
    C = -p.lambda_ABmo * (1 + p.AP * p.delta_APm)
    y0[1] = (-B + math.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
    # With AgeStart=30, et AP=1: = 4.13206616750243e-10; AP=0: 3.233829730236738e-10

    """AB_o^o (Amyloid-beta oligomers extracell.)"""
    A = p.kappa_ABooABpo
    B = p.d_ABoo
    C = - p.kappa_ABmoABoo * (1 + p.AP * p.delta_APmo) * (y0[1] ** 2)
    y0[2] = (-B + math.sqrt((B ** 2) - (4 * A * C))) / (2 * A)
    # With AgeStart=30, and AP=1 : = 6.468006453985709e-15; AP=0 : 1.4672579335221297e-15

    """AB_p^o (Amyloid-beta plaque extracell.)"""
    # y0[3]
    # Take equilibrium, hence defined after microglia and macrophages.

    """G (GSK3)"""
    y0[4] = p.G_0   # F: ~ 5.3445e-05 ; M: ~ 1.5007e-5

    """tau (tau proteins)"""
    A = p.kappa_tauFi
    B = p.d_tau
    C = -(p.lambda_tau + p.lambda_Gtau)
    y0[5] = (-B + math.sqrt((B**2) - (4 * A * C))) / (2 * A)
    # ~= 6.4947e-07

    """F_i (NFT inside the neurons)"""
    y0[6] = p.kappa_tauFi * (y0[5] ** 2) / p.d_Fi  # ~ 4.6751e-11

    """F_o (NFT outside the neurons)"""
    y0[7] = 5e-17

    """N (Living neurons)"""
    y0[8] = p.N_0

    """A (Activated astrocytes)"""
    # y0[9] = 0  # Keep the value = 0.

    # M_NA : Après M_pro et M_anti.

    """M_pro (Proinflammatory microglia)"""
    y0[11] = 1e-12

    """M_anti (Anti-inflammatory microglia)"""
    y0[12] = 1e-4

    """M_NA (Resting microglia)"""
    if p.S == 0:  # women
        M_0 = 3.811e-2
    else:  # men
        M_0 = 3.193e-2
    y0[10] = M_0 - (y0[11] + y0[12])

    """hat{M}_pro (Proinflammatory macrophages)"""
    # Keep the value = 0. (y0[13])

    """hat{M}_anti (Anti-inflammatory macrophages)"""
    # Keep the value = 0. (y0[14])

    """AB_p^o (Amyloid-beta plaque extracell.) - Équilibre!"""
    Psi = p.kappa_ABooABpo * (y0[2] ** 2)   # AP=1 : 1.985969404177517e-24 ; AP=0 : 1.0219851779307061e-25
    D = (p.d_MantiABpo * y0[12] + p.d_hatMantiABpo * y0[14]) * (1 + p.AP * p.delta_APdp)
    y0[3] = (Psi * p.K_ABpo) / (D - Psi)

    """T_{beta} (TGF-beta)"""
    y0[15] = (p.kappa_MantiTb * y0[12] + p.kappa_MhatantiTb * y0[14])/p.d_Tb  # = 1.887293111828693e-14

    """I_10 (IL-10 = Interleukin 10)"""
    y0[16] = (p.kappa_MantiI10 * y0[12] + p.kappa_MhatantiI10 * y0[14]) / p.d_I10
    # = 1.4136387580013201e-11, with Mia14

    """T_{alpha} (TNF-alpha)"""
    y0[17] = (p.kappa_MproTa * y0[11] + p.kappa_MhatproTa * y0[13]) / p.d_Ta
    # new kappas (Fadok98) min: = 1.827060352940543e-21

    """P (MCP-1)"""
    y0[18] = (p.kappa_MproP * y0[11] + p.kappa_MhatproP * y0[13] + p.kappa_AP * y0[9]) / p.d_P  # 1.987681043308942e-19

    return y0

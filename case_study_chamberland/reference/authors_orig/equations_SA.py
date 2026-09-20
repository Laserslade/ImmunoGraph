
"""This file contains the equations of the model.
It defines a function that generates the ODE system of the model.

:Author: Halima Sadia
:Creation date: oct 5, 2024
:Last modified:


"""

import numpy as np

InsVar = True

def ODEsystem_SA(t, y, p):

    dydt = np.zeros(19)

    # Living neurons (N)
    dydt[8] = -p.d_FiN * (1 / (1 + np.exp(- p.n * (y[6] - p.K_Fi) / p.K_Fi))) * y[8] \
              - p.d_TaN * (y[17] / (y[17] + p.K_Ta)) * (1 / (1 + (y[16] / p.K_I10))) * y[8]

    # Amyloid-beta monomer inside the neurons (AB^i)
    dydt[0] = p.lambda_ABi * (1 + p.AP * p.delta_APi) * (y[8] / p.N_0) - p.d_ABi * y[0] - (y[0] / y[8]) * abs(dydt[8])

    # Amyloid-beta monomer outside the neurons (AB_m^o)
    dydt[1] = (y[0] / y[8]) * abs(dydt[8]) + p.lambda_ABmo * (1 + p.AP * p.delta_APm) * (y[8] / p.N_0) \
              + p.lambda_AABmo * (y[9] / p.A_0) - p.kappa_ABmoABoo * (1 + p.AP * p.delta_APmo) * (y[1] ** 2) \
              - p.d_ABmo(t) * y[1]

    # Amyloid-beta oligomers outside (AB_o^o)
    dydt[2] = p.kappa_ABmoABoo * (1 + p.AP * p.delta_APmo) * (y[1] ** 2) - p.kappa_ABooABpo * (y[2] ** 2) \
              - p.d_ABoo * y[2]

    # Amyloid-beta plaque outside the neurons (AB_p^o)
    dydt[3] = p.kappa_ABooABpo * (y[2] ** 2) - ((p.d_MantiABpo * y[12] + p.d_hatMantiABpo * y[14])
                                                * (1 + p.AP * p.delta_APdp) * (y[3] / (y[3] + p.K_ABpo)))

    # Glycogen synthase kinase-type 3 (GSK-3) (G)
    if InsVar:
        dydt[4] = p.lambda_InsG * (p.Ins_0 / p.Ins(t, p.S)) * (y[8] / p.N_0) - p.d_G * y[4] \
                  - (y[4] / y[8]) * abs(dydt[8])
    else:  # Insuline cste
        dydt[4] = p.lambda_InsG * (y[8] / p.N_0) - p.d_G * y[4] - (y[4] / y[8]) * abs(dydt[8])

    # tau proteins (tau)
    dydt[5] = p.lambda_tau * (y[8] / p.N_0) + p.lambda_Gtau * (y[4] / p.G_0) \
              - p.kappa_tauFi * (y[5] ** 2) * (y[8] / p.N_0) - (y[5] / y[8]) * abs(dydt[8]) - p.d_tau * y[5]

    # NFT inside the neurons (F_i)
    dydt[6] = p.kappa_tauFi * (y[5] ** 2) * (y[8] / p.N_0) - (y[6] / y[8]) * abs(dydt[8]) - p.d_Fi * y[6]

    # NFT outside the neurons (F_o)
    dydt[7] = (y[6] / y[8]) * abs(dydt[8]) - p.kappa_MFo * (y[12] / (y[12] + p.K_Manti)) * y[7] - p.d_Fo * y[7]

    # Astrocytes (A)
    dydt[9] = (p.kappa_ABpoA * y[3] + p.kappa_TaA * y[17]) * (p.A_max - y[9]) - p.d_A * y[9]

    # Microglia (M)
    M_activ = p.kappa_FoM * (y[7] / (y[7] + p.K_Fo)) * y[10] + p.kappa_ABooM * (y[2] / (y[2] + p.K_ABooM)) * y[10]

    # Resting microglia (not activated) (M_NA)
    dydt[10] = p.d_Mpro * y[11] + p.d_Manti * y[12] - M_activ

    epsilon_Ta = y[17] / (y[17] + p.K_TaAct)
    epsilon_I10 = y[16] / (y[16] + p.K_I10Act)

    # Proinflammatory microglia (M_pro)
    dydt[11] = ((p.beta * epsilon_Ta) / (p.beta * epsilon_Ta + epsilon_I10)) * M_activ \
               - p.kappa_TbMpro * (y[15] / (y[15] + p.K_TbM)) * y[11] \
               + p.kappa_TaManti * (y[17] / (y[17] + p.K_TaM)) * y[12] \
               - p.d_Mpro * y[11]

    # Anti-inflammatory microglia (M_anti)
    dydt[12] = (epsilon_I10 / (p.beta * epsilon_Ta + epsilon_I10)) * M_activ \
               + p.kappa_TbMpro * (y[15] / (y[15] + p.K_TbM)) * y[11] \
               - p.kappa_TaManti * (y[17] / (y[17] + p.K_TaM)) * y[12] - p.d_Manti * y[12]

    # Proinflammatory macrophages (hat{M}_pro)
    dydt[13] = p.kappa_PMhat * (y[18] / (y[18] + p.K_P)) * (p.Mhatmax - (y[13] + y[14])) * \
               ((p.beta * epsilon_Ta) / (p.beta * epsilon_Ta + epsilon_I10)) \
               - p.kappa_TbMhatpro * (y[15] / (y[15] + p.K_TbMhat)) * y[13] \
               + p.kappa_TaMhatanti * (y[17] / (y[17] + p.K_TaMhat)) * y[14] - p.d_Mhatpro * y[13]

    # Anti-inflammatory macrophages (hat{M}_anti)
    dydt[14] = p.kappa_PMhat * (y[18] / (y[18] + p.K_P)) * (p.Mhatmax - (y[13] + y[14])) * \
               (epsilon_I10 / (p.beta * epsilon_Ta + epsilon_I10)) \
               + p.kappa_TbMhatpro * (y[15] / (y[15] + p.K_TbMhat)) * y[13] \
               - p.kappa_TaMhatanti * (y[17] / (y[17] + p.K_TaMhat)) * y[14] - p.d_Mhatanti * y[14]

    # TGF-Beta = Transforming growth factor beta (T_beta)
    dydt[15] = p.kappa_MantiTb * y[12] + p.kappa_MhatantiTb * y[14] - p.d_Tb * y[15]

    # IL-10 = Interleukin 10 (I_10)
    dydt[16] = p.kappa_MantiI10 * y[12] + p.kappa_MhatantiI10 * y[14] - p.d_I10 * y[16]

    # TNF-alpha = Tumor necrosis factor alpha (T_alpha)
    dydt[17] = p.kappa_MproTa * y[11] + p.kappa_MhatproTa * y[13] - p.d_Ta * y[17]

    # MCP-1 (P)
    dydt[18] = p.kappa_MproP * y[11] + p.kappa_MhatproP * y[13] + p.kappa_AP * y[9] - p.d_P * y[18]

    return dydt
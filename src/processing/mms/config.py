# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:01:45 2025

@author: richarj2
"""

from scipy.constants import atomic_mass, electron_mass

# Atomic masses (neutral atoms) from NIST in atomic mass units (u)
MASS_U = {'H': 1.00782503223, 'He': 4.00260325413, 'O': 15.99491461957}

ION_MASS_DICT = {
    'hplus':        MASS_U['H'] * atomic_mass - electron_mass,
    'heplus':       MASS_U['He'] * atomic_mass - electron_mass,
    'heplusplus':   MASS_U['He'] * atomic_mass - 2 * electron_mass,
    'oplus':        MASS_U['O'] * atomic_mass - electron_mass,
}

MMS_SPACECRAFT = ('mms1', 'mms2', 'mms3', 'mms4')

###---------- FGM Data ----------###

FGM_VARIABLES_TEMPLATE = { # 1 minute resolution
    # Vectors are 4 columns - magntiude and components
    # Field data is 8/16 Hz
    'epoch'     : 'Epoch',
    'B_gse'     : '{sc}_fgm_b_gse_srvy_l2',
    'B_gsm'     : '{sc}_fgm_b_gsm_srvy_l2',
    'B_flag'    : '{sc}_fgm_flag_srvy_l2', # quality
    # State data is every 30s
    'epoch_pos' : 'Epoch_state',
    'r_gse'     : '{sc}_fgm_r_gse_srvy_l2', # km
}

MMS_VARIABLES = {
    sc: {key: value.format(sc=sc) for key, value in FGM_VARIABLES_TEMPLATE.items()}
    for sc in MMS_SPACECRAFT
}

###---------- HPCA Data ----------###

HPCA_VARIABLES_TEMPLATE = {
    # Field data are 4 columns - magntiude and components
    # Spin data - roughly ever 10s
    'epoch'         : 'Epoch',
    'B_gse'         : '{sc}_hpca_B_GSE_spin_avg', # Use to calculate thermal pressure
    'B_gsm'         : '{sc}_hpca_B_GSM_spin_avg',
}

ION_SPECIES = ('hplus','heplus','heplusplus','oplus')

for ion in ION_SPECIES:
    # Plasma data are 3 columns or 3x3 tensors
    HPCA_VARIABLES_TEMPLATE[f'V_gse_{ion}'] = '{sc}_hpca_' + f'{ion}_ion_bulk_velocity'        # km/s
    HPCA_VARIABLES_TEMPLATE[f'V_gsm_{ion}'] = '{sc}_hpca_' + f'{ion}_ion_bulk_velocity_GSM'
    HPCA_VARIABLES_TEMPLATE[f'P_{ion}']     = '{sc}_hpca_' + f'{ion}_ion_pressure'             # Thermal pressure tensor, approximate p = 1/3 . Tr(P)
    HPCA_VARIABLES_TEMPLATE[f'N_{ion}']     = '{sc}_hpca_' + f'{ion}_number_density'           # n/cc
    HPCA_VARIABLES_TEMPLATE[f'T_{ion}']     = '{sc}_hpca_' + f'{ion}_scalar_temperature'       # eV

MMS_VARIABLES_HPCA = {
    sc: {key: value.format(sc=sc) for key, value in FGM_VARIABLES_TEMPLATE.items()}
    for sc in MMS_SPACECRAFT
}

###---------- FPI Data ----------###



FPI_VARIABLES_TEMPLATE = {
    'epoch'     : 'Epoch',                              # Nanoseconds since J2000
    'flag'      : '{sc}_dis_errorflags_fast',           # Vector of data-quality indicators at epoch start time
    'V_gse'     : '{sc}_dis_bulkv_gse_fast',            # Bulk-velocity vector in GSE     km/s
    'V_gse_unc' : '{sc}_dis_bulkv_err_fast',            # Bulk-velocity error vector      km/s
    'N_tot'     : '{sc}_dis_numberdensity_fast',        # Number density                  n/cc
    'N_tot_bg'  : '{sc}_dis_numberdensity_bg_fast',     # Background Number density       n/cc
    'N_tot_unc' : '{sc}_dis_numberdensity_err_fast',    # Number density error            n/cc
    'P_th_bg'   : '{sc}_dis_pres_bg_fast',              # Background pressure             nPa
    'P_th_tens' : '{sc}_dis_prestensor_gse_fast',       # Pressure tensor in GSE          nPa
    'P_th_unc'  : '{sc}_dis_prestensor_err_fast',       # Pressure tensor error           nPa
    'T_tens'    : '{sc}_dis_temptensor_gse_fast',       # Temperature tensor in GSE       eV
    'T_unc'     : '{sc}_dis_temptensor_err_fast',       # Temperature tensor error        eV
}

MMS_VARIABLES_FPI = {
    sc: {key: value.format(sc=sc) for key, value in FPI_VARIABLES_TEMPLATE.items()}
    for sc in MMS_SPACECRAFT
}



VARIABLES_DICT = {'fgm': MMS_VARIABLES, 'hpca': MMS_VARIABLES_HPCA, 'fpi': MMS_VARIABLES_FPI}




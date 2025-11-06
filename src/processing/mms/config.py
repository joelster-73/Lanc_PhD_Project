# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:01:45 2025

@author: richarj2
"""

MMS_VARIABLES = {
    # Vectors are 4 columns - magntiude and components
    # Field data is 8/16 Hz
    'epoch'     : 'Epoch',
    'B_gse'     : 'mms1_fgm_b_gse_srvy_l2',
    'B_gsm'     : 'mms1_fgm_b_gsm_srvy_l2',
    'B_flag'    : 'mms1_fgm_flag_srvy_l2', # quality
    # State data is every 30s
    'epoch_pos' : 'Epoch_state',
    'r_gse'     : 'mms1_fgm_r_gse_srvy_l2', # km
}

MMS_VARIABLES_HPCA = {
    # Field data are 4 columns - magntiude and components
    # Spin data - roughly ever 10s
    'epoch'         : 'Epoch',
    'B_gse'         : 'mms1_hpca_B_GSE_spin_avg', # Use to calculate thermal pressure
    'B_gsm'         : 'mms1_hpca_B_GSM_spin_avg',
}

ION_SPECIES = ('hplus','heplus','heplusplus','oplus')

for ion in ION_SPECIES:
    # Plasma data are 3 columns or 3x3 tensors
    MMS_VARIABLES_HPCA[f'V_gse_{ion}'] = f'mms1_hpca_{ion}_ion_bulk_velocity'        # km/s
    MMS_VARIABLES_HPCA[f'V_gsm_{ion}'] = f'mms1_hpca_{ion}_ion_bulk_velocity_GSM'
    MMS_VARIABLES_HPCA[f'P_{ion}']     = f'mms1_hpca_{ion}_ion_pressure'             # Thermal pressure tensor, approximate p = 1/3 . Tr(P)
    MMS_VARIABLES_HPCA[f'N_{ion}']     = f'mms1_hpca_{ion}_number_density'           # n/cc
    MMS_VARIABLES_HPCA[f'T_{ion}']     = f'mms1_hpca_{ion}_scalar_temperature'       # eV



from scipy.constants import atomic_mass, electron_mass

# Atomic masses (neutral atoms) from NIST in atomic mass units (u)
MASS_U = {'H': 1.00782503223, 'He': 4.00260325413, 'O': 15.99491461957}

ION_MASS_DICT = {
    'hplus':        MASS_U['H'] * atomic_mass - electron_mass,
    'heplus':       MASS_U['He'] * atomic_mass - electron_mass,
    'heplusplus':   MASS_U['He'] * atomic_mass - 2 * electron_mass,
    'oplus':        MASS_U['O'] * atomic_mass - electron_mass,
}


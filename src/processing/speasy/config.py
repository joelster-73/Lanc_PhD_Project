# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:24:20 2025

@author: richarj2
"""

from datetime import datetime
import speasy as spz

# Data trees

amda_tree = spz.inventories.tree.amda
cda_tree  = spz.inventories.tree.cda
ssc_tree  = spz.inventories.tree.ssc

OMNI_CDA_TREE = cda_tree.OMNI_Combined_1AU_IP_Data__Magnetic_and_Solar_Indices.OMNI_1AU_IP_Data.IMF_and_Plasma_data.OMNI_HRO2_1MIN

# To find product, type in the console "cda_tree." and then tab, and keep doing this to find product


sw_monitors = ('WIND','ACE','DSC','C1','C2','C3','C4','THA','THB','THC','GEO','IMP8')
cluster_sc = ('C1','C2','C3','C4')
themis_sc  = ('THA','THB','THC')


C1_dict = {
    'B_mag': 'c1_btot_5vps',    # 0.2s resolution
    'B_GSE': 'c1_b_5vps',       # 0.2s resolution
    'B_GSM': 'c1_bgsm_5vps',    # 0.2s resolution
    'R_GSE': 'c1_xyz_gse',      # 60s resolution
    'V_mag': 'c1_hia_vtot',     # Hot ion bulk velocity mag, spin resolution
    'V_GSE': 'c1_hia_v',        # Hot ion bulk velocity, spin resolution
    'P_dyn': 'c1_hia_press',    # Hot ion pressure, spin resolution
    'N_tot': 'c1_hia_dens',     # Hot ion number density, spin resolution
    'T_tot': 'c1_hia_t'         # (eV) Hot Ion temperature, spin resolution
}

C2_dict = {
    'B_mag': 'c2_btot_5vps',
    'B_GSE': 'c2_b_5vps',
    'B_GSM': 'c2_bgsm_5vps',
    'R_GSE': 'c2_xyz_gse',
    'V_mag': 'c2_hia_vtot',
    'V_GSE': 'c2_hia_v',
    'P_dyn': 'c2_hia_press',
    'N_tot': 'c2_hia_dens',
    'T_tot': 'c2_hia_t'
}

C3_dict = {
    'B_mag': 'c3_btot_5vps',
    'B_GSE': 'c3_b_5vps',
    'B_GSM': 'c3_bgsm_5vps',
    'R_GSE': 'c3_xyz_gse',
    'V_mag': 'c3_hia_vtot',
    'V_GSE': 'c3_hia_v',
    'P_dyn': 'c3_hia_press',
    'N_tot': 'c3_hia_dens',
    'T_tot': 'c3_hia_t'
}

C4_dict = {
    'B_mag': 'c4_btot_5vps',
    'B_GSE': 'c4_b_5vps',
    'B_GSM': 'c4_bgsm_5vps',
    'R_GSE': 'c4_xyz_gse',
    'V_mag': 'c4_hia_vtot',
    'V_GSE': 'c4_hia_v',
    'P_dyn': 'c4_hia_press',
    'N_tot': 'c4_hia_dens',
    'T_tot': 'c4_hia_t'
}

THA_dict = {
    'B_mag': 'tha_bl',          # only in low (0.25s) resolution
    'B_GSE': 'tha_bl',          # only in low (0.25s) resolution
    'B_GSM': 'tha_bl_gsm',      # only in low (0.25s) resolution
    'R_GSE': 'tha_xyz',         # 60s resolution
    'V_mag': None,
    'V_GSE': None,
    'P_dyn': None,
    'N_tot': None,
    'T_tot': None
}

THB_dict = {
    'B_mag': 'thb_bl',
    'B_GSE': 'thb_bl',
    'B_GSM': 'thb_bl_gsm',
    'R_GSE': 'thb_xyz',
    'V_mag': None,
    'V_GSE': None,
    'P_dyn': None,
    'N_tot': None,
    'T_tot': None
}

THC_dict = {
    'B_mag': 'thc_bl',
    'B_GSE': 'thc_bl',
    'B_GSM': 'thc_bl_gsm',
    'R_GSE': 'thc_xyz',
    'V_mag': None,
    'V_GSE': None,
    'P_dyn': None,
    'N_tot': None,
    'T_tot': None
}

THD_dict = {
    'B_mag': 'thd_bl',
    'B_GSE': 'thd_bl',
    'B_GSM': 'thd_bl_gsm',
    'R_GSE': 'thd_xyz',
    'V_mag': None,
    'V_GSE': None,
    'P_dyn': None,
    'N_tot': None,
    'T_tot': None
}

THE_dict = {
    'B_mag': 'the_bl',
    'B_GSE': 'the_bl',
    'B_GSM': 'the_bl_gsm',
    'R_GSE': 'the_xyz',
    'V_mag': None,
    'V_GSE': None,
    'P_dyn': None,
    'N_tot': None,
    'T_tot': None
}

ACE_dict = {
    'B_mag': 'imf_mag',         # 16s resolution
    'B_GSE': 'imf',             # 16s resolution
    'B_GSM': 'imf_gsm',         # 16s resolution
    'R_GSE': 'ace_xyz_gse',     # 12m resolution
    'V_mag': None,
    'V_GSE': 'sw_v_gse',        # 64s resolution
    'P_dyn': 'sw_pdyn',         # 64s resolution
    'N_tot': None,
    'T_tot': None
}

DSC_dict = {
    'B_mag': 'cda_tree.DSCOVR.MAG.DSCOVR_H0_MAG.B1F1',  # 1s resolution
    'B_GSE': 'dsc_b_gse',       # 1m resolution (1s res only for 2025+)
    'B_GSM': 'dsc_b_gsm',       # 1m resolution (1s res only for 2025+)
    'R_GSE': 'ssc_tree.Trajectories.dscovr',            # 12m resolution, in km
    'V_mag': None,
    'V_GSE': 'dsc_vpr_3s_gse',  # 3s resolution
    'P_dyn': 'dsc_pdyn_3s',     # 3s resolution
    'N_tot': None,
    'T_tot': None
}

GEO_dict = {
    'B_mag': 'gtl_bmag_edb',    # 3s resolution
    'B_GSE': 'gtl_b_edb',       # 3s resolution
    'B_GSM': 'gtl_b_edb_gsm',   # 3s resolution
    'R_GSE': 'gtl_xyz',         # 10m resolution
    'V_mag': None,
    'V_GSE': None,
    'P_dyn': None,
    'N_tot': None,
    'T_tot': None
}

IMP8_dict = {
    'B_mag': 'imp8_btot',       # 15s resolution
    'B_GSE': 'imp8_b',          # 15s resolution
    'B_GSM': 'imp8_b_gsm',      # 15s resolution
    'R_GSE': 'imp8_xyz',        # 12m resolution
    'V_mag': None,
    'V_GSE': None,
    'P_dyn': None,
    'N_tot': None,
    'T_tot': None
}

WIND_dict = {
    'B_mag': 'wnd_bmagh',       # 3s resolution
    'B_GSE': 'wnd_bh',          # 3s resolution
    'B_GSM': 'cda_tree.Wind.WIND.MFI.WI_H0_MFI.B3GSM',  # GSM data reference
    'R_GSE': 'wnd_xyz_gse',     # 10m resolution
    'V_mag': None,
    'V_GSE': 'wnd_swe_v',       # 90s resolution
    'P_dyn': 'wnd_swe_pdyn',    # 90s resolution
    'N_tot': None,
    'T_tot': None
}

OMNI_dict = { # 1m resolution
    'B_mag': 'omni_hro2_1min_b_amag',
    'B_GSE': 'omni_hro2_1min_b_gse',
    'B_GSM': [OMNI_CDA_TREE.BX_GSE, OMNI_CDA_TREE.BY_GSM, OMNI_CDA_TREE.BZ_GSM],  # GSM components
    'R_GSE': 'omni_hro2_1min_bsn_gse',  # BS nose location, Rt = - |Rt| * V/|V|
    'V_mag': 'omni_hro2_1min_sw_vb',    # Flow speed
    'V_GSE': 'omni_hro2_1min_sw_vgse',  # Bulk velocity
    'P_dyn': 'omni_hro2_1min_sw_p',     # (2*10**-6)*Np*Vp**2 nPa
    'N_tot': 'omni_hro2_1min_sw_n',     # Proton number density
    'T_tot': 'omni_hro2_1min_sw_t',     # (K) solar wind temp
    'sc'   : OMNI_CDA_TREE.IMF,         # 51=WIND, 52=WIND-V2, 50=IMP8, 60=GEO, 71=ACE
    'lag'  : OMNI_CDA_TREE.Timeshift,
    'MA'   : 'omni_hro2_1min_mach_num', # (V * Np**0.5) / (20 * B)
    'beta' : 'omni_hro2_1min_beta',     # [(T*4.16/10**5) + 5.34] * Np / B**2 (B in nT)
    'Ey'   : 'omni_hro2_1min_sw_e'      # -V(km/s) * Bz (nT; GSM) * 10**-3, mV/m
}


speasy_variables = {
    'C1': C1_dict,
    'C2': C2_dict,
    'C3': C3_dict,
    'C4': C4_dict,
    'THA': THA_dict,
    'THB': THB_dict,
    'THC': THC_dict,
    'THD': THD_dict,
    'THE': THE_dict,
    'ACE': ACE_dict,
    'DSC': DSC_dict,
    'GEO': GEO_dict,
    'IMP8': IMP8_dict,
    'WIND': WIND_dict,
    'OMNI': OMNI_dict
}


colour_dict = {
    'OMNI': 'orange',
    'C1':   'blue',
    'C2':   'cornflowerblue',
    'C3':   'lightskyblue',
    'C4':   'lightblue',
    'THA':  'forestgreen',
    'THB':  'seagreen',
    'THC':  'mediumseagreen',
    'THD':  'lightgreen',
    'THE':  'palegreen',
    'ACE':  'darkviolet',
    'DSC':  'deeppink',
    'GEO':  'cyan',
    'IMP8': 'crimson',
    'WIND': 'magenta'
}

database_colour_dict = {'CFA': 'b', 'Donki': 'r', 'Helsink': 'g'}


data_availability = {
    'C1':   (datetime(2000, 8, 22),          datetime(2024, 10, 3, 20, 35)),
    'C2':   (datetime(2000, 8, 22),          datetime(2024, 9, 8, 18, 45)),
    'C3':   (datetime(2000, 8, 22),          datetime(2024, 10, 1)),
    'C4':   (datetime(2000, 8, 22),          datetime(2024, 10, 6, 14, 7)),
    'THA':  (datetime(2007, 2, 17, 23, 36),  datetime(2025, 5, 17, 23, 59)),
    'THB':  (datetime(2007, 2, 17, 23, 36),  datetime(2025, 5, 10, 23, 59)),
    'THC':  (datetime(2007, 2, 17, 23, 36),  datetime(2025, 5, 9, 23, 59)),
    'THD':  (datetime(2007, 2, 17, 23, 36),  datetime(2025, 5, 17, 23, 59)),
    'THE':  (datetime(2007, 2, 17, 23, 36),  datetime(2025, 5, 16, 23, 59)),
    'ACE':  (datetime(1997, 9, 2),           datetime(2025, 5, 13)),
    'DSC':  (datetime(2015, 2, 11, 23, 48),  datetime(2025, 7, 1, 12)),
    'GEO':  (datetime(1992, 9, 18),          datetime(2006, 11, 22, 23, 59)),
    'IMP8': (datetime(1973, 10, 30),         datetime(2000, 6, 10, 23, 59)),
    'WIND': (datetime(1994, 11, 1),          datetime(2024, 9, 29)),
    'OMNI': (datetime(1995, 1, 1),           datetime(2025, 1, 9))
}

data_availability_mag = {
    'C1':   (datetime(2000, 12, 3, 0, 56),   datetime(2023, 12, 31, 23, 59)),
    'C2':   (datetime(2000, 12, 2, 14, 1),   datetime(2023, 12, 31, 23, 59)),
    'C3':   (datetime(2000, 12, 2, 15, 11),  datetime(2023, 12, 31, 23, 59)),
    'C4':   (datetime(2001, 1, 7),           datetime(2023, 12, 31, 23, 59)),
    'THA':  (datetime(2007, 2, 17, 23, 36),  datetime(2025, 5, 17, 23, 59)),
    'THB':  (datetime(2007, 2, 17, 23, 36),  datetime(2025, 5, 10, 23, 59)),
    'THC':  (datetime(2007, 2, 17, 23, 36),  datetime(2025, 5, 9, 23, 59)),
    'THD':  (datetime(2007, 2, 17, 23, 36),  datetime(2025, 5, 17, 23, 59)),
    'THE':  (datetime(2007, 2, 17, 23, 36),  datetime(2025, 5, 16, 23, 59)),
    'ACE':  (datetime(1997, 9, 2),           datetime(2025, 5, 13)),
    'DSC':  (datetime(2015, 6, 8),           datetime(2025, 5, 3)),
    'GEO':  (datetime(1992, 9, 18),          datetime(2006, 11, 22, 23, 59)),
    'IMP8': (datetime(1973, 10, 30),         datetime(2000, 6, 10, 23, 59)),
    'WIND': (datetime(1994, 11, 1),          datetime(2024, 9, 29)),
    'OMNI': (datetime(1995, 1, 1),           datetime(2025, 1, 9))
}
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:24:20 2025

@author: richarj2
"""

from datetime import datetime
import speasy as spz

# Data trees

amda_tree = spz.inventories.tree.amda
cda_tree = spz.inventories.tree.cda
ssc_tree = spz.inventories.tree.ssc

OMNI_CDA_TREE = cda_tree.OMNI_Combined_1AU_IP_Data__Magnetic_and_Solar_Indices.OMNI_1AU_IP_Data.IMF_and_Plasma_data.OMNI_HRO_1MIN

# To find product, type in the console "cda_tree." and then tab, and keep doing this to find product

all_spacecraft = {'L1': ('ACE','DSC','WIND'),
                  'Earth': ('OMNI','GEO','IMP8','C1','C2','C3','C4','THA','THB','THC','THD','THE')}

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


B_dict_gse = {
    'C1':   'c1_b_5vps',           # 0.2s resolution
    'C2':   'c2_b_5vps',
    'C3':   'c3_b_5vps',
    'C4':   'c4_b_5vps',
    'THA':  'tha_bl',              # only in low (0.25s) resolution
    'THB':  'thb_bl',
    'THC':  'thc_bl',
    'THD':  'thd_bl',
    'THE':  'the_bl',
    'ACE':  'imf',                 # 16s resolution
    'DSC':  'dsc_b_gse',           # 1m resolution (1s res only for 2025+)
    'GEO':  'gtl_b_edb',           # 3s resolution
    'IMP8': 'imp8_b',              # 15s resolution
    'WIND': 'wnd_bh',              # 3s resolution
    'OMNI': 'omni_hro2_1min_b_gse' # 1m resolution
}

B_dict_gsm = {
    'C1':   'c1_bgsm_5vps',  # 0.2s resolution
    'C2':   'c2_bgsm_5vps',
    'C3':   'c3_bgsm_5vps',
    'C4':   'c4_bgsm_5vps',
    'THA':  'tha_bl_gsm',    # only in low (0.25s) resolution
    'THB':  'thb_bl_gsm',
    'THC':  'thc_bl_gsm',
    'THD':  'thd_bl_gsm',
    'THE':  'the_bl_gsm',
    'ACE':  'imf_gsm',       # 16s resolution
    'DSC':  'dsc_b_gsm',     # 1m resolution (1s res only for 2025+)
    'GEO':  'gtl_b_edb_gsm', # 3s resolution
    'IMP8': 'imp8_b_gsm',    # 15s resolution
    'WIND': cda_tree.Wind.WIND.MFI.WI_H0_MFI.B3GSM,
    'OMNI': [OMNI_CDA_TREE.BX_GSE, OMNI_CDA_TREE.BY_GSM, OMNI_CDA_TREE.BZ_GSM]

}

B_dict_tot = {
    'C1':   'c1_btot_5vps',      # 0.2s resolution
    'C2':   'c2_btot_5vps',
    'C3':   'c3_btot_5vps',
    'C4':   'c4_btot_5vps',
    'THA':  'tha_bl_tot',        # only in low (0.25s) resolution
    'THB':  'thb_bl_tot',
    'THC':  'thc_bl_tot',
    'THD':  'thd_bl_tot',
    'THE':  'the_bl_tot',
    'ACE':  'imf_mag',           # 16s resolution
    'DSC':  cda_tree.DSCOVR.MAG.DSCOVR_H0_MAG.B1F1,  # 1s resolution
    #'DSC':  'dsc_b_mag',         # 1m resolution (1s res only for 2025+)
    'GEO':  'gtl_bmag_edb',      # 3s resolution
    'IMP8': 'imp8_btot',         # 15s resolution
    'WIND': 'wnd_bmagh',         # 3s resolution
    'OMNI': 'omni_hro2_1min_mag' # 1m resolution
}

R_dict_gse = { # Re
    'C1':   'c1_xyz_gse',            # 60s resolution
    'C2':   'c2_xyz_gse',
    'C3':   'c3_xyz_gse',
    'C4':   'c4_xyz_gse',
    'THA':  'tha_xyz',               # 60s resolution
    'THB':  'thb_xyz',
    'THC':  'thc_xyz',
    'THD':  'thd_xyz',
    'THE':  'the_xyz',
    'ACE':  'ace_xyz_gse',           # 12m resolution
    'DSC':  ssc_tree.Trajectories.dscovr, #12m resolution, in km not Re
    #'DSC':  'dsc_xyz_gse', # 1h resolution   - amda id is 'dsc_xyz_gse'
    'GEO':  'gtl_xyz',               # 10m resolution
    'IMP8': 'imp8_xyz',              # 12m resolution
    'WIND': 'wnd_xyz_gse',           # 10m resolution
    'OMNI': 'omni_hro2_1min_bsn_gse' # BS nose location
}

P_dict_dyn = { # Ram Pressure
    'C1':   None,                 # X resolution
    'C2':   None,
    'C3':   None,
    'C4':   None,
    'THA':  None,                 # X resolution
    'THB':  None,
    'THC':  None,
    'THD':  None,
    'THE':  None,
    'ACE':  'sw_pdyn',            # 64s resolution
    'DSC':  'dsc_pdyn_3s',        # 3s resolution
    'GEO':  None,
    'IMP8':  None,
    'WIND': 'wnd_swe_pdyn',       # 90s resolution
    'OMNI': 'omni_hro2_1min_sw_p' # 1m resolution
}

V_dict_gse = {
    'C1':   None,                    # X resolution
    'C2':   None,
    'C3':   None,
    'C4':   None,
    'THA':  None,                    # X resolution
    'THB':  None,
    'THC':  None,
    'THD':  None,
    'THE':  None,
    'ACE':  'sw_v_gse',              # 64s resolution
    'DSC':  'dsc_vpr_3s_gse',        # 3s resolution (proton velocity)
    'GEO':   None,
    'IMP8':  None,
    'WIND': 'wnd_swe_v',             # 90s resolution
    'OMNI': 'omni_hro2_1min_sw_vgse' # Bulk Velocitity
}


speasy_variables = {
    'B_GSE': B_dict_gse,
    'B_GSM': B_dict_gsm,
    'B_mag': B_dict_tot,
    'R_GSE': R_dict_gse,
    'P_dyn': P_dict_dyn,
    'V_GSE': V_dict_gse,
    'OMNI_sc': OMNI_CDA_TREE.IMF
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
    'GEO':  'teal',
    'IMP8': 'crimson',
    'WIND': 'magenta'
}
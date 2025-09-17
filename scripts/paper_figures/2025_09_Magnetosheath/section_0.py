# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""


# %%
import numpy as np
import pandas as pd

from src.config import PROC_CLUS_OMNI_MSH5
from src.processing.reading import import_processed_data

from src.plotting.comparing.parameter import compare_columns, plot_nguyen
from src.plotting.space_time import plot_orbit

from src.coordinates.boundaries import calc_msh_r_diff

column_names = {
    'r_x_name'  : 'r_x_GSE',
    'r_y_name'  : 'r_y_GSE',
    'r_z_name'  : 'r_z_GSE',
    'r_name'    : 'r_mag',
    'r_ax_name' : 'r_x_aGSE',
    'r_ay_name' : 'r_y_aGSE',
    'r_az_name' : 'r_z_aGSE',
    'v_x_name'  : 'V_x_GSE',
    'v_y_name'  : 'V_y_GSE',
    'v_z_name'  : 'V_z_GSE',
    'p_name'    : 'P_tot'
}

df_merged = import_processed_data(PROC_CLUS_OMNI_MSH5)

df_positions = calc_msh_r_diff(df_merged, 'BOTH', position_key='msh', data_key='sw', column_names=column_names)
df_merged = pd.concat([df_merged,df_positions[['r_MP','r_BS','r_phi','r_F']]],axis=1)

for param, ratio_limit in zip(('E_y_GSM','B_z_GSM'),(5,8)):

    df_merged.loc[np.abs(df_merged[f'{param}_msh'])>ratio_limit*np.abs(df_merged[f'{param}_sw']),f'{param}_msh'] = np.nan

### Look at relative strength in terms of E_Y = (-V x B)_Y
### Do histograms of parameters in raw databases and then in MSH times
### Consider plotting every interval sw and msh parameter

### HERE DO LOOP FOR INDEPENDENT/DEPENDENT RELTIOSN
### SO AE BUT ALSO SATURATION IN MSH BASED ON FACTOR E.G. CLOCK ANGLE
### THEN CREATE FIGURES USING ENTIRE OMNI/CLUSTER DATASETS

# Create similar plots looking at saturation against Electric field and how velocity and magnetic field affect


# %% Nguyen

plot_nguyen(df_merged, 'N_tot', 'B_avg')

plot_orbit(df_merged, sc_key='msh', plane='x-rho', bin_width=0.2, centre_Earth='quarter')


# %%
from src.coordinates.boundaries import mp_shue1998, bs_jelinek2012
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

f_step = 0.1
f_num = int(1/f_step)+1

theta_step = 3
theta_num = int(90/theta_step)+1

f_edges = np.linspace(0, 1, f_num)
theta_edges = np.linspace(0, np.pi/2, theta_num)

param_dict = {'beta': 'MA', 'V_x_GSM': 'V_x_GSE'}

for param in ('count','B_avg','V_mag','E_mag','N_tot','P_tot'):

    for filter_zone in ('all','low','high'):

        phi   = df_merged['r_phi']
        f     = df_merged['r_F']

        if filter_zone=='all':
            filter_mask = np.ones(len(phi),dtype=bool)
            title = 'All beta'
        elif filter_zone=='low':
            filter_mask = df_merged['beta_msh']<1
            title = r'$\beta<1$'
        elif filter_zone=='high':
            filter_mask = df_merged['beta_msh']>1
            title = r'$\beta>1$'
        elif filter_zone=='sub':
            filter_mask = df_merged['MA_sw']<1
            title = r'$M_A<1$'
        elif filter_zone=='sup':
            filter_mask = df_merged['MA_sw']>1
            title = r'$M_A>1$'
        elif filter_zone=='perp':
            filter_mask = (np.abs(df_merged['B_clock_sw']) <= np.pi/4) | (np.abs(df_merged['B_clock_sw']) >= 3*np.pi/4)
            title = 'Quasi-perp'
        elif filter_zone=='para':
            filter_mask = (np.abs(df_merged['B_clock_sw']) > np.pi/4) & (np.abs(df_merged['B_clock_sw']) < 3*np.pi/4)
            title = 'Quasi-para'

        if np.sum(filter_mask)==0:
            print(f'No data for {param} with {filter_zone} filter.')
            continue

        if param == 'count':
            counts, f_edges, theta_edges = np.histogram2d(f[filter_mask], phi[filter_mask], bins=[f_edges, theta_edges])
        else:
            param_sw = param_dict.get(param,param)
            v = df_merged[f'{param}_msh'] / df_merged[f'{param_sw}_sw']

            mask = filter_mask.copy()

            for series in (phi,f,v):
                mask &= ~np.isnan(series)

            mean_v, f_edges, theta_edges, _ = binned_statistic_2d(f[mask], phi[mask], v[mask], statistic='mean', bins=[f_edges, theta_edges])

        # meshgrid
        f_centres = 0.5 * (f_edges[:-1] + f_edges[1:])
        theta_centres = 0.5 * (theta_edges[:-1] + theta_edges[1:])

        f_centres = f_edges
        theta_centres = theta_edges

        F, P = np.meshgrid(f_centres, theta_centres, indexing='ij')

        # convert back to radius
        R = mp_shue1998(P) + F * (bs_jelinek2012(P) - mp_shue1998(P))

        X = R * np.cos(P)
        Y = R * np.sin(P)

        fig, ax = plt.subplots(figsize=(7,7), dpi=200)

        if param == 'count':
            #cm = ax.pcolormesh(X, Y, counts, cmap='hot', norm=mpl.colors.LogNorm())
            cm = ax.pcolormesh(X, Y, counts, cmap='hot')
            z_label = 'Minutes'
        else:
            z_min, z_max = None, None
            if param == 'beta':
                z_min = 0
                z_max = 2
            elif param == 'B_avg':
                z_min = 0
                z_max = 6
            elif param == 'E_mag':
                z_min = 0
                z_max = 4
            elif param == 'N_tot':
                z_min = 1
                z_max = 8
            elif param == 'V_mag':
                z_min = 0
                z_max = 0.8
            elif param == 'P_tot':
                z_min = 0
                z_max = 0.7

            cm = ax.pcolormesh(X, Y, mean_v, cmap='cool', vmin=z_min, vmax=z_max)
            z_label = f'{param} ratio'

        ax.plot(mp_shue1998(theta_edges)*np.cos(theta_edges),mp_shue1998(theta_edges)*np.sin(theta_edges),c='w',ls='--')
        ax.plot(bs_jelinek2012(theta_edges)*np.cos(theta_edges),bs_jelinek2012(theta_edges)*np.sin(theta_edges),c='w',ls='--')

        ax.set_xlabel('X [aGSE]')
        ax.set_ylabel(r'$\sqrt{Y^2+Z^2}$')
        ax.xaxis.set_inverted(True)

        ax.set_facecolor('k')
        ax.set_aspect('equal')

        cbar = fig.colorbar(cm, ax=ax)
        cbar.set_label(z_label)

        if param == 'count':
            ticks = cbar.get_ticks()
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{int(t*5):,}' for t in ticks])

        ax.set_title(title)

        plt.tight_layout()
        plt.show()


# %% Pulkkinen

rules = {'B_avg':   (2, (1, 5)),
         'B_z_GSM': (2, (1, 5)),
         'B_clock': (np.pi/5, np.pi/36),
         'E_y_GSM': (1, (0.5, 1)),
         'V_mag':   (50, 5),
         'N_tot':   (1, (1, 5)),
         'S_mag':   (10, (4, 100))
         }

# Also include beta MA plot
for param in ('B_avg','B_z_GSM','B_clock','E_y_GSM','V_mag','N_tot','S_mag'):

    if param!='B_z_GSM':
        continue

    sw_param  = f'{param}_sw'

    # Include sw error if B else ignore

    msh_param = f'{param}_msh'
    msh_err   = f'{param}_unc_msh'
    msh_count = f'{param}_count_msh'


    step, bin_width = rules.get(param,(1, (1, 5)))

    compare_columns(df_merged, sw_param, msh_param, display='heat', bin_width=bin_width)

    if '_GS' in param:
        field, _, coords = param.split('_')
        msh_count = f'{field}_{coords}_count_msh'

    compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='MA_sw', display='scatter', scatter_size=20, z_max=12, zero_lines=(param=='B_z_GSM'))

    if param=='B_z_GSM':
        compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='region_duration', display='scatter', scatter_size=20, z_max=60000, data3_name='region duration', zero_lines=(param=='B_z_GSM'))

    compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='beta_msh', display='scatter_binned', col2_counts=msh_count, scatter_size=20, bin_step=step, z_min=0, z_max=2)

# %% beta

# Different format
ma   = 'MA'
beta = 'beta'

sw_param  = f'{ma}_sw'

msh_param = f'{beta}_msh'
msh_err   = f'{beta}_unc_msh'
msh_count = f'{beta}_count_msh'


### NEED TO CLEAN UP THE MSH DATA FOR ANOMALOUS BETA THEN



compare_columns(df_merged[(df_merged['beta_msh']<50)&(df_merged['MA_sw']<60)], sw_param, msh_param, display='heat', bin_width=1)

compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='E_y_GSM_sw', display='scatter', scatter_size=20, z_min=0, z_max=8)

compare_columns(df_merged, sw_param, msh_param, col2_err=msh_err, col3='E_y_GSM_sw', display='scatter_binned', col2_counts=msh_count, scatter_size=20, bin_step=1, z_min=0, z_max=8)

# %% Correlation

for param in ('E_y_GSM','E_R'):
    sw_param  = f'{param}_sw'
    msh_param = f'{param}_msh'
    msh_err   = f'{param}_unc_msh'

    if '_GS' in param:
        field, _, coords = param.split('_')
        msh_count = f'{field}_{coords}_count_msh'

    step = 1

    mask = np.ones(len(df_merged),dtype=bool)

    if param=='E_y_GSM':
        mask &= (df_merged.loc[:, sw_param]>0) & (df_merged.loc[:, msh_param]>0) & (np.abs(df_merged.loc[:, msh_param])<1e6)

    compare_columns(df_merged.loc[mask], sw_param, 'AE_17m_pc', col3='MA_sw', display='scatter_binned', scatter_size=20, bin_step=step, z_max=8)

    compare_columns(df_merged.loc[mask], msh_param, 'AE_pc', col1_err=msh_err, col3='MA_sw', display='scatter_binned', scatter_size=20, bin_step=step)


# %%
for i in range(3):

    for couple in ('E_y_GSM_sw','B_z_GSM_sw'):

        mask = np.ones(len(df_merged),dtype=bool)
        if i>=1:
            if 'E_' in couple:
                mask &= df_merged[couple]>0
            elif 'B_' in couple:
                mask &= df_merged[couple]<0

        if i>=2:
            if 'E_' in couple:
                mask &= df_merged[couple]<16
            elif 'B_' in couple:
                mask &= df_merged[couple]>-20

        for z_param in ('MA_sw','MA_med','beta_msh'):


            if z_param=='V_mag':
                edges = [400,500,600,700,800]
            elif z_param=='MA_sw':
                edges = [5]
            elif z_param=='beta_msh':
                edges = [1]
            elif z_param=='MA_med':
                edges = [9.2]
                z_param = 'MA_sw'

            compare_columns(df_merged[mask], couple, 'AE_17m_pc', col3=z_param, display='scatter_binned_multiple', scatter_size=20, zs_edges=edges)


# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from uncertainties import ufloat, unumpy as unp
from ..calculations import vec_mag

from ...config import R_E
from ...processing.speasy.retrieval import get_shock_position

def find_outliers(shocks, threshold_time=40, coeff_lim=0.7):

    sc_labels = [col.split('_')[0] for col in shocks if '_coeff' in col]

    only_Earth = True

    indices = []
    for index, shock in shocks.iterrows():

        detector = shock['spacecraft']

        BS_time     = shock['OMNI_time']
        if pd.isnull(BS_time):
            continue
        BS_coeff = shock['OMNI_coeff']
        if np.isnan(BS_coeff) or BS_coeff<coeff_lim or BS_coeff>1:
            #1.1 indicates exact matches
            continue

        add_shock = False
        for sc in sc_labels:
            if sc in ('OMNI',detector):
                continue

            elif (only_Earth) and sc in ('WIND','ACE','DSC'):
                continue

            corr_coeff = shock[f'{sc}_coeff']
            if isinstance(corr_coeff, (pd.Series, pd.DataFrame)) and len(corr_coeff) > 1:
                corr_coeff = corr_coeff.iloc[0]  # Get the first value
            else:
                corr_coeff = corr_coeff

            if np.isnan(corr_coeff) or corr_coeff<coeff_lim or corr_coeff>1:
                #1.1 indicates exact matches
                continue

            sc_time = shock[f'{sc}_time']
            if pd.isnull(sc_time):
                continue
            time_diff     = (shock[f'{sc}_time'] - BS_time).total_seconds()
            if np.abs(time_diff)>=(40*60): # positive or engative
                add_shock = True

        if add_shock:
            indices.append(index)

    return indices


# make into separate functions

def accuracy_against_earth_sun():

    # if 'L1' in x_axis:
    #     detector = shock['detectors'].split(',')[0]
    #     detector_pos = shock[[f'{detector}_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy()
    #     detector_pos_unc = shock[[f'{detector}_r_{comp}_GSE_unc' for comp in ('x','y','z')]].to_numpy()
    #     if np.isnan(detector_pos[0]):
    #         continue
    #     elif np.isnan(detector_pos_unc[0]):
    #         detector_pos_unc = np.zeros(len(BS_pos))
    #     L1_pos_u = unp.uarray(detector_pos,detector_pos_unc)

    return





def get_time_dist_differences(shocks, normals=None):

    # For quantifying the error on OMNI
    # Can adapt for the different methods (i.e. two spacecraft, omni and its spacecraft, omni and any other spacecraft)


    df = pd.DataFrame(columns=['detector','interceptor','time','time_unc','delta_x','delta_x_unc','delta_r','delta_r_unc','delta_n','delta_n_unc','delta_t','delta_t_unc'])

    sc_labels = [col.split('_')[0] for col in shocks if '_time_unc_s' in col]

    for index, shock in shocks.iterrows():

        # Position of bow shock and time when OMNI predicts shock intercepts it
        BS_time     = shock['OMNI_time']
        BS_time_unc = shock['OMNI_time_unc_s']
        if pd.isnull(BS_time):
            continue

        BS_pos_u = get_shock_position(shock, 'OMNI')
        if BS_pos_u is None:
            continue

        normal = None
        if normals is not None:
            detectors = shock['detectors']
            if ',' in detectors:
                detectors = detectors.split(',')
            else:
                detectors = (detectors,)

            # Needs to be changed
            det = detectors[0]
            normal_row = normals[(normals.index==shock[f'{det}_time'])&(normals['spacecraft']==det)]

            if len(normal_row)>0:
                normal_row = normal_row.iloc[0]
            normal = normal_row[['Nx','Nx_unc','Ny','Ny_unc','Nz','Nz_unc','v_sh','v_sh_unc']]

        # Distance and times of spacecraft from BSN
        for sc in sc_labels:

            new_row = []

            # Want to compare spacecraft only not used by OMNI
            if  sc=='OMNI' or sc==shock['OMNI_sc']:
                continue

            sc_time = shock[f'{sc}_time']
            sc_time_unc = shock[f'{sc}_time_unc_s']

            if pd.isnull(sc_time):
                continue

            sc_pos_u = get_shock_position(shock, sc)
            if sc_pos_u is None:
                continue

            new_row += [sc, 'OMNI']

            time_diff     = (sc_time - BS_time).total_seconds()
            time_diff_unc = ufloat(time_diff,sc_time_unc) - ufloat(0,BS_time_unc)
            new_row += [time_diff, time_diff_unc.s]

            for method in ('x_comp', 'dist', 'normal', 'normal time'):

                diff = calc_sc_dist_diff(sc_pos_u, BS_pos_u, shock=normal, method=method)

                if diff is None:
                    new_row += [np.nan, np.nan]
                else:
                    new_row += [np.float64(diff[0]), np.float64(diff[1])]

            df.loc[len(df)] = new_row

    df.attrs = {}
    df_units = {key: 'STRING' for key in ('detector', 'interceptor')}
    df_units |= {key: 's' for key in ('time', 'time_unc', 'delta_t', 'delta_t_unc')}
    df_units |= {key: r'$\mathrm{R_E}$' for key in ('delta_x', 'delta_x_unc', 'delta_r', 'delta_r_unc', 'delta_n', 'delta_n_unc')}
    df.attrs['units'] = df_units

    return df


def calc_sc_dist_diff(sc1_pos, sc2_pos, shock=None, method='normal'):

    if method=='x_comp':
        dist_diff = sc1_pos[0] - sc2_pos[0]

        return (dist_diff.n, dist_diff.s)

    elif method=='dist':
        dist_diff    = vec_mag(sc1_pos-sc2_pos)
        if isinstance(dist_diff, np.ndarray):  # flatten if needed
            dist_diff = dist_diff[0]
        distance     = unp.nominal_values(dist_diff)
        distance_unc = unp.std_devs(dist_diff)

        return (distance, distance_unc)

    elif 'normal' in method and shock is not None:
        try:
            n_sh = unp.uarray(shock[['Nx','Ny','Ny']].to_numpy(),
                              shock[['Nx_unc','Ny_unc','Nz_unc']].to_numpy())
            nx = n_sh[0]
            if nx.s>abs(nx.n):
                return None

            diff = np.dot(n_sh, (sc1_pos - sc2_pos))

            if 'time' in method:
                v_sh = ufloat(shock['v_sh'], shock['v_sh_unc']) / R_E # km/s -> RE/s
                if v_sh.n < 0 or v_sh.s>abs(v_sh.n):
                    return None

                diff /= v_sh

            return (diff.n, diff.s)

        except:
            return None

    return None
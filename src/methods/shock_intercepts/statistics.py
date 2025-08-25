# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import itertools as it

from uncertainties import ufloat, unumpy as unp
from ...analysing.calculations import vec_mag

from ...config import R_E
from ...processing.speasy.retrieval import get_shock_position, retrieve_omni_value
from ...processing.shocks.helsinki import get_list_of_events_all, get_list_of_events_helsinki

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


def get_shock_propagations(shocks, normals=None):


    monitors = ('ACE','WIND','DSC','C1','C3','C4','OMNI')


    df = pd.DataFrame(columns=['detector','interceptor','time','time_unc','delta_x','delta_x_unc','delta_r','delta_r_unc','delta_n','delta_n_unc','delta_t','delta_t_unc'])

    for index, shock in shocks.iterrows():

        for upstream, downstream in it.combinations(monitors, 2):

            if downstream=='OMNI':
                if pd.isnull(shock['OMNI_time']):
                    continue
                omni_sc = retrieve_omni_value(shock['OMNI_time'], 'OMNI_sc')
                if omni_sc is not None:
                    omni_sc = omni_sc.upper()
                if upstream!=omni_sc:
                    continue

            ###----- Upstream spacecraft time and position -----###
            up_time = shock[f'{upstream}_time']
            up_unc  = shock[f'{upstream}_time_unc_s']
            if pd.isnull(up_time):
                continue
            if pd.isnull(up_unc):
                up_unc = np.zeros(3)

            up_pos  = shock[[f'{upstream}_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy(dtype=float)
            up_pos_unc = shock[[f'{upstream}_r_{comp}_GSE_unc' for comp in ('x','y','z')]].to_numpy(dtype=float)
            if not np.all(np.isfinite(up_pos_unc)):
                up_pos_unc = np.zeros(3)

            up_pos_u = unp.uarray(up_pos,up_pos_unc)

            ###----- Downstream spacecraft time and position -----###
            dw_time = shock[f'{downstream}_time']
            dw_unc  = shock[f'{downstream}_time_unc_s']
            if pd.isnull(dw_time):
                continue
            if pd.isnull(dw_unc):
                dw_unc = np.zeros(3)

            dw_pos  = shock[[f'{downstream}_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy(dtype=float)
            dw_pos_unc = shock[[f'{downstream}_r_{comp}_GSE_unc' for comp in ('x','y','z')]].to_numpy(dtype=float)
            if not np.all(np.isfinite(dw_pos_unc)):
                dw_pos_unc = np.zeros(3)

            dw_pos_u = unp.uarray(dw_pos,dw_pos_unc)

            if downstream=='OMNI' and np.sum(np.abs(dw_pos)>=9999)>1: # Bad data flag
                continue

            ###----- Normal vector of the upstream spacecraft -----###
            normal_vec = None
            if normals is not None:
                normal_row = normals[(normals.index==shock[f'{upstream}_time'])&(normals['spacecraft']==upstream)]
                if len(normal_row)>0:
                    normal_row = normal_row.iloc[0]
                normal_vec = normal_row[['Nx','Nx_unc','Ny','Ny_unc','Nz','Nz_unc','v_sh','v_sh_unc']]

            new_row = []

            new_row += [upstream, downstream]

            time_diff     = (up_time - dw_time).total_seconds()
            time_diff_unc = ufloat(time_diff,up_unc) - ufloat(0,dw_unc)
            new_row += [time_diff, time_diff_unc.s]

            for method in ('x_comp', 'dist', 'normal', 'normal time'):

                diff = calc_sc_dist_diff(up_pos_u, dw_pos_u, shock=normal_vec, method=method)

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


def get_diffs_with_OMNI(shocks, normals=None):


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

        normal_vecs = {}
        if normals is not None:
            detectors = shock['detectors']
            for det in detectors:
                normal_row = normals[(normals.index==shock[f'{det}_time'])&(normals['spacecraft']==det)]
                if len(normal_row)>0:
                    normal_row = normal_row.iloc[0]
                normal_vecs[det] = normal_row[['Nx','Nx_unc','Ny','Ny_unc','Nz','Nz_unc','v_sh','v_sh_unc']]

        # Distance and times of spacecraft from BSN
        for sc in sc_labels:

            new_row = []

            # Want to compare only spacecraft not used by OMNI
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

            detector = shock[f'{sc}_sc']
            normal = normal_vecs.get(detector,None)

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
            if nx.s>abs(nx.n): # direction of shock completely uncertain
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



def shock_compressions(shocks):

    try:
        event_list = get_list_of_events_all(shocks)
    except:
        event_list = get_list_of_events_helsinki(shocks)

    df = pd.DataFrame(columns=['eventID','sc_up','sc_dw','comp_up','comp_dw','change_rel','change_abs','time'])

    df.attrs = {'units': {}}
    df.attrs['units']['eventID'] = ''
    df.attrs['units']['time'] = 's'
    df.attrs['units'].update({key: 'STRING' for key in ('sc_up', 'sc_dw')})
    df.attrs['units'].update({key: '1' for key in ('comp_up','comp_dw','change_rel','change_abs')})


    for eventNum, event in enumerate(event_list):

        # Need more shocks for comparison
        if len(event)<=1:
            continue

        for i in range(2):

            new_row = [eventNum]

            if i==0: # Earliest and latest
                event_copy = event.copy()
                if 'OMNI' in event_copy:
                    del event_copy['OMNI']
                upstream   = min(event_copy, key=lambda k: event[k][0])
                downstream = max(event_copy, key=lambda k: event[k][0])
                if upstream==downstream:
                    continue

            else: # OMNI
                if 'OMNI' not in event:
                    continue
                omni_time = event['OMNI'][0]
                omni_sc = retrieve_omni_value(omni_time, 'OMNI_sc')
                if omni_sc is None:
                    continue
                omni_sc = omni_sc.upper()

                if omni_sc=='WIND-V2':
                    upstream = 'WIND'
                else:
                    upstream = omni_sc
                downstream = 'OMNI'

                if upstream not in event:
                    continue

            new_row += [upstream, downstream]

            up_time, up_unc = event.get(upstream)
            up_comp = shocks.loc[up_time, ['B_ratio','B_ratio_unc']]
            if isinstance(up_comp, pd.DataFrame):
                up_comp = up_comp.iloc[0].to_numpy()
            else:
                up_comp = up_comp.to_numpy()
            up_comp = ufloat(up_comp[0],up_comp[1])

            dw_time_u = event.get(downstream)
            dw_time, dw_unc = dw_time_u

            dw_comp = shocks.loc[dw_time, ['B_ratio','B_ratio_unc']]
            if isinstance(dw_comp, pd.DataFrame):
                dw_comp = dw_comp.iloc[0].to_numpy()
            else:
                dw_comp = dw_comp.to_numpy()
            dw_comp = ufloat(dw_comp[0],dw_comp[1])

            new_row += [up_comp, dw_comp, dw_comp/up_comp, dw_comp-up_comp]

            dt = (dw_time-up_time).total_seconds()
            dt_unc = ufloat(dt,dw_unc)-ufloat(0,up_unc)

            new_row += [ufloat(dt,dt_unc.s)]

            df.loc[len(df)] = new_row

    return df

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from uncertainties import ufloat, unumpy as unp
from ..calculations import get_position_u, vec_mag

from src.processing.speasy.retrieval import retrieve_position_unc
from src.processing.speasy.config import speasy_variables

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


def get_time_dist_differences(shocks, **kwargs):

    # For quantifying the error on OMNI

    selection     = kwargs.get('selection','all')
    x_axis        = kwargs.get('x_axis','dist')
    colouring     = kwargs.get('colouring','spacecraft')

    distances     = []
    distances_unc = []
    times         = []
    times_unc     = []
    z_values      = []

    sc_labels = [col.split('_')[0] for col in shocks if '_time_unc_s' in col]

    for index, shock in shocks.iterrows():

        BS_time     = shock['OMNI_time']
        BS_time_unc = shock['OMNI_time_unc_s']
        if pd.isnull(BS_time):
            continue

        if x_axis!='earth_sun' or 'L1' not in x_axis:
            try:
                BS_pos = shock[[f'OMNI_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy()
                BS_pos_unc = shock[[f'OMNI_r_{comp}_GSE_unc' for comp in ('x','y','z')]].to_numpy()
            except:
                BS_pos, BS_pos_unc = retrieve_position_unc('OMNI', speasy_variables, BS_time, BS_time_unc)

            if BS_pos is None:
                continue
            elif np.isnan(BS_pos[0]):
                continue
            elif np.isnan(BS_pos_unc[0]):
                BS_pos_unc = np.zeros(len(BS_pos))

            BS_pos_u = unp.uarray(BS_pos,BS_pos_unc)

        if 'L1' in x_axis:
            detector = shock['detectors'].split(',')[0]
            detector_pos = shock[[f'{detector}_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy()
            detector_pos_unc = shock[[f'{detector}_r_{comp}_GSE_unc' for comp in ('x','y','z')]].to_numpy()
            if np.isnan(detector_pos[0]):
                continue
            elif np.isnan(detector_pos_unc[0]):
                detector_pos_unc = np.zeros(len(BS_pos))
            L1_pos_u = unp.uarray(detector_pos,detector_pos_unc)

        all_distances = {}
        all_dist_uncs = {}
        for sc in sc_labels:
            sc_time = shock[f'{sc}_time']
            sc_time_unc = shock[f'{sc}_time_unc_s']
            if  sc=='OMNI':
                continue
            elif pd.isnull(sc_time):
                continue

            try:
                sc_pos = shock[[f'{sc}_r_{comp}_GSE' for comp in ('x','y','z')]].to_numpy()
                sc_pos_unc = shock[[f'{sc}_r_{comp}_GSE_unc' for comp in ('x','y','z')]].to_numpy()
            except:
                sc_pos, sc_pos_unc = retrieve_position_unc(sc, speasy_variables, sc_time, sc_time_unc)

            if sc_pos is None:
                continue
            elif np.isnan(sc_pos[0]):
                continue
            elif np.isnan(sc_pos_unc[0]):
                sc_pos_unc = np.zeros(len(sc_pos))

            sc_pos_u = unp.uarray(sc_pos,sc_pos_unc)
            if x_axis=='earth_sun':
                sc_rho = unp.sqrt(sc_pos_u[1]**2+sc_pos_u[2]**2)

                distance     = sc_rho.n
                distance_unc = sc_rho.s

            elif x_axis=='x_comp':
                sc_x = sc_pos_u[0]
                if np.isnan(sc_x.n):
                    continue
                bs_x = BS_pos_u[0]

                distance     = (sc_x-bs_x).n
                distance_unc = (sc_x-bs_x).s

            elif x_axis=='dist':
                dist_diff    = vec_mag(sc_pos_u-BS_pos_u)
                distance     = unp.nominal_values(dist_diff)
                distance_unc = unp.std_devs(dist_diff)

            elif x_axis=='signed_dist':
                sign         = np.sign(sc_pos_u[0].n)
                dist_diff    = vec_mag(sc_pos_u-BS_pos_u)
                distance     = sign*unp.nominal_values(dist_diff)
                distance_unc = unp.std_devs(dist_diff)

            elif x_axis=='L1_x':

                distance     = detector_pos[0]
                distance_unc = detector_pos_unc[0]

            elif x_axis=='L1_rho':

                L1_rho = unp.sqrt(L1_pos_u[1]**2+L1_pos_u[2]**2)
                distance     = unp.nominal_values(L1_rho)
                distance_unc = unp.std_devs(L1_rho)

            else:
                raise Exception(f'{x_axis} not valid choice of "x_axis".')

            all_distances[sc] = distance
            all_dist_uncs[sc] = distance_unc

        if selection=='closest':
            closest_sc = min(all_distances, key=all_distances.get)

        for sc in all_distances:
            if (selection=='closest') and sc!=closest_sc:
                continue

            if colouring=='detector':
                z_value = shock['detectors'][0].upper()

            elif colouring=='spacecraft':
                z_value = sc

            elif colouring=='none':
                z_value = 0

            z_values.append(z_value)

            distances.append(all_distances[sc])
            distances_unc.append(all_dist_uncs[sc])

            time_diff     = (shock[f'{sc}_time'] - BS_time).total_seconds()
            time_diff_unc = ufloat(time_diff,shock[f'{sc}_time_unc_s']) - ufloat(0,BS_time_unc)
            times.append(time_diff)
            times_unc.append(time_diff_unc.s)


    distances     = np.array(distances)
    times         = np.array(times)
    distances_unc = np.array(distances_unc)
    times_unc     = np.array(times_unc)
    z_values      = np.array(z_values)

    return distances, times, distances_unc, times_unc, z_values
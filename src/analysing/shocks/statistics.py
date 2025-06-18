# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from uncertainties import ufloat, unumpy as unp
from ..calculations import get_position_u, vec_mag

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

    coeff_lim     = kwargs.get('coeff_lim',0.7)
    selection     = kwargs.get('selection','all')
    x_axis        = kwargs.get('x_axis','dist')
    colouring     = kwargs.get('colouring','spacecraft')
    cfa_shocks    = kwargs.get('cfa_shocks',None)

    distances     = []
    distances_unc = []
    times         = []
    times_unc     = []
    z_values      = []


    sc_labels = [col.split('_')[0] for col in shocks if '_coeff' in col]

    for index, shock in shocks.iterrows():
        detector = shock['spacecraft']

        BS_time     = shock['OMNI_time']
        if pd.isnull(BS_time):
            continue


        BS_pos = get_position_u(shock,'OMNI')
        if BS_pos is None and x_axis!='earth_sun':
            continue

        for sc in sc_labels:
            if (selection=='closest' and sc!=shock['closest']) or sc in ('OMNI',detector):
                continue
            elif (selection=='earth') and sc in ('WIND','ACE','DSC'):
                continue

            corr_coeff = shock[f'{sc}_coeff']
            if isinstance(corr_coeff, (pd.Series, pd.DataFrame)) and len(corr_coeff) > 1:
                corr_coeff = corr_coeff.iloc[0]  # Get the first value
            else:
                corr_coeff = corr_coeff

            if np.isnan(corr_coeff) or corr_coeff<coeff_lim or corr_coeff>1:
                #1.1 indicates exact matches
                continue

            if x_axis=='earth_sun':
                L1_pos = get_position_u(shock,detector)
                if L1_pos is None:
                    continue
                L1_rho = unp.sqrt(L1_pos[1]**2+L1_pos[2]**2)

                distance     = L1_rho.n
                distance_unc = L1_rho.s

            elif x_axis=='x_comp':
                sc_x = ufloat(shock[f'{sc}_r_x_GSE'],shock[f'{sc}_r_x_GSE_unc'])
                if np.isnan(sc_x.n):
                    continue
                bs_x = ufloat(shock['OMNI_r_x_GSE'] ,shock['OMNI_r_x_GSE_unc'])

                distance     = (sc_x-bs_x).n
                distance_unc = (sc_x-bs_x).s


            elif x_axis=='dist':
                sc_pos = get_position_u(shock,sc)
                if sc_pos is None:
                    continue
                dist_diff = vec_mag(sc_pos-BS_pos)
                distance     = unp.nominal_values(dist_diff)
                distance_unc = unp.std_devs(dist_diff)

            elif x_axis=='signed_dist':
                sc_pos = get_position_u(shock,sc)
                if sc_pos is None:
                    continue
                sign         = np.sign(shock[f'{sc}_r_x_GSE'])
                dist_diff    = vec_mag(sc_pos-BS_pos)
                distance     = sign*unp.nominal_values(dist_diff)
                distance_unc = unp.std_devs(dist_diff)
            else:
                raise Exception(f'{x_axis} not valid choice of "x_axis".')

            if colouring=='coeff':
                z_value = corr_coeff

            elif colouring=='angle':
                try:
                    normal = cfa_shocks.loc[index,['Nx','Ny','Nz']]
                    normal_mag = np.linalg.norm(normal)
                    normal /= normal_mag # to ensure normalised
                except:
                    continue
                basis_vector = np.array([-1,0,0])
                z_value = np.arccos(np.dot(normal,basis_vector))

            elif colouring=='sun_earth':
                sc_pos = get_position_u(shock,sc)
                if sc_pos is None:
                    continue
                sc_vec = unp.nominal_values(sc_pos)
                z_value = np.sqrt(sc_vec[1]**2+sc_vec[2]**2)

            elif colouring=='detector':
                z_value = detector.upper()

            elif colouring=='spacecraft':
                z_value = sc

            elif colouring=='database':
                z_value = 'CFA' if shock['source']=='C' else 'Donki'


            if np.isnan(z_value):
                continue
            z_values.append(z_value)


            distances.append(distance)
            distances_unc.append(distance_unc)

            time_diff     = (shock[f'{sc}_time'] - BS_time).total_seconds()
            time_diff_unc = ufloat(time_diff,shock[f'{sc}_time_unc_s']) - ufloat(0,shock['OMNI_time_unc_s'])
            times.append(time_diff)
            times_unc.append(time_diff_unc.s)


    distances     = np.array(distances)
    times         = np.array(times)
    distances_unc = np.array(distances_unc)
    times_unc     = np.array(times_unc)
    z_values      = np.array(z_values)

    return distances, times, distances_unc, times_unc, z_values
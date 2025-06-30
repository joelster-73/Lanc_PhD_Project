# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""

from src.config import PROC_SHOCKS_DIR
from src.processing.reading import import_processed_data

all_processed_shocks = import_processed_data(PROC_SHOCKS_DIR)

# %%
from src.analysing.shocks.intercepts import find_propagation_time
from src.processing.speasy.retrieval import retrieve_modal_omni_sc, retrieve_datum
from src.analysing.shocks.in_sw import in_solar_wind
from src.processing.speasy.config import speasy_variables


from datetime import timedelta
from uncertainties import ufloat
from src.config import R_E

position_var = 'R_GSE'
coeff_lim = 0.7

#upstream = ('WIND','ACE','DSC')
#downstream = ('C1','C2','C3','C4','THB','THC','GEO')
sw_monitors = ('WIND','ACE','DSC','C1','C2','C3','C4','THB','THC','GEO','IMP8')

possible_events = []

for eventID, event in all_processed_shocks.groupby(lambda x: int(all_processed_shocks.loc[x, 'eventNum'])):
    times = event.index.tolist()
    detectors = event['spacecraft'].tolist()


    start, end = min(times), max(times)+timedelta(minutes=90)
    sc_info = retrieve_modal_omni_sc(speasy_variables, start, end, return_counts=True)
    if sc_info is None:
        print(f'No OMNI info for event #{eventID}')
        continue
    modal_sc, counts_dict = sc_info
    if modal_sc == 'Bad Data':
        print(f'No good OMNI data for event #{eventID}')
        continue
    total_counts = sum(list(counts_dict.values()))
    omni_spacecraft = [key.upper() for key, value in counts_dict.items() if value/total_counts>0.2 and key!='Bad Data']

    intercept_dict = {}
    for interceptor in sw_monitors:
        # Don't want spacecraft that are used by OMNI - looking for those to compare with
        if interceptor in omni_spacecraft:
            continue
        elif 'WIND-V2' in omni_spacecraft and interceptor=='WIND':
            continue

        # Initial estimate
        intercept_pos_estimate, _ = retrieve_datum(position_var, interceptor, speasy_variables, times[0], add_omni_sc=False)
        if intercept_pos_estimate is None:
            continue

        intercept_times = []
        intercept_uncs  = []
        for i, row in event.iterrows():
            detector = row['spacecraft']
            if detector == interceptor: # Looking for when shock intercepts a different spacecraft
                continue

            detector_pos = row[['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
            detector_time = row.name

            # Approximate downstream location
            approx_time = detector_time + timedelta(seconds=((intercept_pos_estimate[0]-detector_pos[0])*R_E/500))
            if not in_solar_wind(interceptor, approx_time, speasy_variables):
                continue
            intercept_pos, _ = retrieve_datum(position_var, interceptor, speasy_variables, approx_time, add_omni_sc=False)

            time_lag = find_propagation_time(detector_time, detector, interceptor, 'B_mag', detector_pos, intercept_pos=intercept_pos)
            if time_lag is None:
                continue

            delay, coeff = time_lag
            if coeff<=coeff_lim:
                continue

            lagged_unc = delay - ufloat(0,row['time_unc'])
            intercept_times.append(detector_time + timedelta(seconds=delay.n))
            intercept_uncs.append(lagged_unc.s)

        if len(intercept_times)==0:
            continue
        elif len(intercept_times)==1:
            intercept_dict[interceptor] = (intercept_times[0],intercept_uncs[0])

        min_time = min(intercept_times)
        times_u = np.array([ufloat((time-min_time).total_seconds(), unc) for time, unc in zip(intercept_times,intercept_uncs)])
        avg_time = np.mean(times_u)
        pred_time = min_time + timedelta(seconds=avg_time.n)
        pred_unc = avg_time.s

        intercept_dict[interceptor] = (pred_time,pred_unc)

    if len(intercept_dict)==0:
        print(f'No downstream monitors for event #{eventID}')
        continue

    # MAKE FUNCTION TO DO THE SUBTRACT MIN TIME ETC

    print(f'Will try to find OMNI time for event #{eventID}')
    print(f'Detectors: {detectors}\nOMNI sc: {omni_spacecraft}\nInterceptors:{list(intercept_dict.keys())}\n')
    possible_events.append(eventID)
    continue

    for sc in spacecraft:
        interceptor = sc if sc!='WIND-V2' else 'WIND'
        sc_row = event[event['spacecraft'] == interceptor]
        if row.empty:
            predicted_times = []
            time_uncertainties = []
            for i, row in event.iterrows():
                detector = row['spacecraft']

                detect_pos, _ = retrieve_datum(position_var, source, speasy_variables, row.name, add_omni_sc=False)
                time_lag = find_propagation_time(row.name, detector, interceptor, 'B_mag', row[['r_x_GSE','r_y_GSE','r_z_GSE']], intercept_pos=detect_pos)
                if time_lag is None:
                    continue
                delay, coeff = time_lag
                if coeff<=coeff_lim:
                    continue
                lagged_unc = delay - ufloat(0,row['time_unc'])
                predicted_times.append(row.name+timedelta(seconds=delay.n))
                time_uncertainties.append(lagged_unc)
            min_time = min(predicted_times)
            times_u = np.array([ufloat(time-min_time, unc) for time, unc in zip(predicted_times,time_uncertainties)])
            avg_time = np.mean(time_u)
            pred_time = min_time + avg_time.n
            pred_unc = avg_time.s
            print(pred_time,pred_unc)
        else:
            print(row)

print(possible_events)
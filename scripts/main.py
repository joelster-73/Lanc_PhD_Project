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

event_IDs = []
monitors = []


for eventID, event in all_processed_shocks.groupby(lambda x: int(all_processed_shocks.loc[x, 'eventNum'])):

    times = event.index.tolist()
    uncs = event['time_unc'].tolist()
    detectors = event['spacecraft'].tolist()
    detect_dict = dict(zip(detectors,list(zip(times,uncs))))

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
        elif interceptor in detectors:
            intercept_dict[interceptor] = detect_dict[interceptor]

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

    omni_detectors = {}
    for omni_sc in omni_spacecraft:
        if omni_sc=='WIND-V2' and 'WIND' in detectors:
            omni_detectors['WIND'] = detect_dict['WIND']
        elif omni_sc in detectors:
            omni_detectors[omni_sc] = detect_dict[omni_sc]

    if len(omni_detectors)==0:
        print(f'Need to interpolate to find shock in OMNI detector for event #{eventID}')
        continue


        for omni_sc in omni_spacecraft:

            # Initial estimate
            omni_pos_estimate, _ = retrieve_datum(position_var, 'OMNI', speasy_variables, times[0], add_omni_sc=False)
            if omni_pos_estimate is None:
                continue

            intercept_times = []
            intercept_uncs  = []
            for i, row in event.iterrows():
                detector = row['spacecraft']

                detector_pos = row[['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
                detector_time = row.name

                # Approximate BSN location
                approx_time = detector_time + timedelta(seconds=((omni_pos_estimate[0]-detector_pos[0])*R_E/500))
                omni_pos, _ = retrieve_datum(position_var, 'OMNI', speasy_variables, approx_time, add_omni_sc=False)

                time_lag = find_propagation_time(detector_time, detector, 'OMNI', 'B_mag', detector_pos, intercept_pos=omni_pos)
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
                omni_detectors[omni_sc] = (intercept_times[0],intercept_uncs[0])

            min_time = min(intercept_times)
            times_u = np.array([ufloat((time-min_time).total_seconds(), unc) for time, unc in zip(intercept_times,intercept_uncs)])
            avg_time = np.mean(times_u)
            pred_time = min_time + timedelta(seconds=avg_time.n)
            pred_unc = avg_time.s

            omni_detectors[omni_sc] = (pred_time,pred_unc)


    # Initial estimate
    omni_pos_estimate, _ = retrieve_datum(position_var, 'OMNI', speasy_variables, times[0], add_omni_sc=False)
    if omni_pos_estimate is None:
        continue

    omni_times = []
    omni_uncs  = []
    for detector, (detect_time, detect_unc) in omni_detectors.items():
        if detector in detectors:
            detector_pos = event[event['spacecraft']==detector][['r_x_GSE','r_y_GSE','r_z_GSE']].iloc[0].to_numpy()
        else:
            detector_pos, _ = retrieve_datum(position_var, detector, speasy_variables, detect_time, add_omni_sc=False)

        # Approximate BSN location
        approx_time = detector_time + timedelta(seconds=((omni_pos_estimate[0]-detector_pos[0])*R_E/500))
        omni_pos, _ = retrieve_datum(position_var, 'OMNI', speasy_variables, approx_time, add_omni_sc=False)

        time_lag = find_propagation_time(detect_time, detector, 'OMNI', 'B_mag', detector_pos, intercept_pos=omni_pos)
        if time_lag is None:
            continue

        delay, coeff = time_lag
        if coeff<=coeff_lim:
            continue

        lagged_unc = delay - ufloat(0,detect_unc)
        omni_times.append(detect_time + timedelta(seconds=delay.n))
        omni_uncs.append(lagged_unc.s)

    if len(omni_times)==0:
        print(f'Could not find shock in OMNI for event #{eventID}')
        continue
    elif len(intercept_times)==1:
        omni_time = (omni_times[0],omni_uncs[0])

    min_time = min(omni_times)
    times_u = np.array([ufloat((time-min_time).total_seconds(), unc) for time, unc in zip(omni_times,omni_uncs)])
    avg_time = np.mean(times_u)
    pred_time = min_time + timedelta(seconds=avg_time.n)
    pred_unc = avg_time.s

    omni_time = (pred_time,pred_unc)

    print(f'Found OMNI time for event #{eventID}')
    intercept_dict['OMNI'] = omni_time
    monitors.append(intercept_dict)

# %%
print(len(monitors))
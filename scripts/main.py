# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""


from src.processing.reading import import_processed_data

all_processed_shocks = import_processed_data(PROC_SHOCKS_DIR)

# %%
from src.analysing.shocks.intercepts import find_propagation_time
from src.processing.speasy.retrieval import retrieve_modal_omni_sc
from src.analysing.shocks.in_sw import in_solar_wind
from datetime import timedelta
from uncertainties import ufloat

coeff_lim = 0.7

upstream = ('WIND','ACE','DSC')
downstream = ('C1','C2','C3','C4','THB','THC','GEO')

possible_events = []

for eventID, event in all_processed_shocks.groupby(lambda x: int(all_processed_shocks.loc[x, 'eventNum'])):
    times = event.index.tolist()


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
    spacecraft = [key.upper() for key, value in counts_dict.items() if value/total_counts>0.2 and key!='Bad Data']

    sc_dw_dict = {}
    for sc_dw in downstream:

        # Approximate downstream location
        approx_time = times[0]+timedelta(minutes=60)
        pos_dw, _ = retrieve_datum(position_var, sc_dw, speasy_variables, approx_time, add_omni_sc=False)
        if pos_dw is None:
            continue
        elif not in_solar_wind(sc_dw, approx_time, speasy_variables):
            continue

        # MAKE FUNCTION TO DO THE SUBTRACT MIN TIME ETC

        sc_dw_times = []
        sc_dw_uncs  = []
        for i, row in event.iterrows():
            sc_up = row['spacecraft']
            detector = sc_up if sc_up!='WIND-V2' else 'WIND'
            sc_up_time = row.name

            time_lag = find_propagation_time(sc_up_time, detector, sc_dw, 'B_mag', row[['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy(), intercept_pos=pos_dw)
            if time_lag is None:
                continue

            delay, coeff = time_lag
            if coeff<=coeff_lim:
                continue

            lagged_unc = delay - ufloat(0,row['time_unc'])
            sc_dw_times.append(sc_up_time + timedelta(seconds=delay.n))
            sc_dw_uncs.append(lagged_unc.s)

        if len(sc_dw_times)==0:
            continue
        elif len(sc_dw_times)==1:
            sc_dw_dict[sc_dw] = (sc_dw_times[0],sc_dw_uncs[0])

        min_time = min(sc_dw_times)
        times_u = np.array([ufloat((time-min_time).total_seconds(), unc) for time, unc in zip(sc_dw_times,sc_dw_uncs)])
        avg_time = np.mean(times_u)
        pred_time = min_time + timedelta(seconds=avg_time.n)
        pred_unc = avg_time.s

        sc_dw_dict[sc_dw] = (pred_time,pred_unc)

    if len(sc_dw_dict)==0:
        print(f'No downstream monitors for event #{eventID}')
        continue


    print(f'Will try to find OMNI time for event #{eventID}')
    print(sc_dw_dict)
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
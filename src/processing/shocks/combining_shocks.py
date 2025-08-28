# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 09:34:24 2025

@author: richarj2
"""

from src.config import HELSINKI_DIR, PROC_CFA_DIR
from src.processing.reading import import_processed_data
from src.processing.shocks.helsinki import process_helsinki_shocks, get_list_of_events_all
from src.processing.shocks.donki import get_donki_shocks
from src.processing.speasy.retrieval import retrieve_datum

import numpy as np
import pandas as pd

# %%

helsinki_shocks = process_helsinki_shocks(HELSINKI_DIR, 'Helsinki_database.dat')
helsinki_shocks = helsinki_shocks[['spacecraft','r_x_GSE','r_y_GSE','r_z_GSE','res_B','res_p']]
helsinki_shocks['time_unc'] = helsinki_shocks.apply(lambda row: 0.5*max(row['res_B'], row['res_p']), axis=1)
helsinki_shocks.drop(columns=['res_B','res_p'],inplace=True)
helsinki_shocks['source'] = 'H'

cfa_shocks = import_processed_data(PROC_CFA_DIR)
cfa_shocks = cfa_shocks[['time_s_unc','spacecraft','r_x_GSE','r_y_GSE','r_z_GSE']]
cfa_shocks.rename(columns={'time_s_unc':'time_unc'},inplace=True)
cfa_shocks['spacecraft'] = cfa_shocks['spacecraft'].str.upper()
cfa_shocks = cfa_shocks.loc[
    ~(cfa_shocks['spacecraft'] == cfa_shocks['spacecraft'].shift()) |
    (cfa_shocks.index.to_series().diff() > pd.Timedelta(minutes=5))
]
cfa_shocks['source'] = 'C'

donki_shocks = get_donki_shocks()
donki_shocks.rename(columns={'time_s_unc':'time_unc'},inplace=True)
donki_shocks['spacecraft'] = donki_shocks['spacecraft'].str.upper()
donki_shocks['source'] = 'D'

for col in ('r_x_GSE','r_y_GSE','r_z_GSE'):
    donki_shocks[col] = np.nan

# %%
from collections import Counter

shocks = pd.concat([helsinki_shocks,cfa_shocks,donki_shocks]).sort_index()
shocks = shocks[shocks['spacecraft'].isin(('WIND','ACE','DSC','C1','C2','C3','C4'))]
event_list = get_list_of_events_all(shocks)
print(sum([len(d) for d in event_list]))

key_counts = Counter()
for d in event_list:
    key_counts.update(d.keys())

print(key_counts)

# %%
columns = ['epoch','eventNum','time_unc','spacecraft','r_x_GSE','r_y_GSE','r_z_GSE','source']
all_shocks = pd.DataFrame(columns=columns)

position_var = 'R_GSE'
source_dict = {'C': cfa_shocks, 'D': donki_shocks, 'H': helsinki_shocks}

for i, event_dict in enumerate(event_list):
    for sc, info in event_dict.items():
        time = info[0]
        df = source_dict.get(info[2])
        position = df.loc[time,['r_x_GSE','r_y_GSE','r_z_GSE']].to_numpy()
        rad_dist = np.linalg.norm(position)
        if np.isnan(rad_dist) or (np.linalg.norm(rad_dist)<=20 and sc in ('WIND','ACE','DSC')):
            position, _ = retrieve_datum(position_var, sc, time, add_omni_sc=False)
            if position is None:
                continue
        all_shocks.loc[len(all_shocks)] = [time, str(i+1), info[1], sc, position[0], position[1], position[2], info[2]]

all_shocks.set_index('epoch',inplace=True)


# %%
new_attrs = cfa_shocks.attrs.copy()
for col in list(cfa_shocks.attrs['units']):
    if col not in all_shocks:
        del new_attrs['units'][col]
new_attrs['units']['eventNum'] = 'STRING'
new_attrs['units']['source'] = 'STRING'
new_attrs['units']['time_unc'] = 's'

all_shocks.attrs = new_attrs

# %%
from src.processing.writing import write_to_cdf
from src.config import PROC_SHOCKS_DIR
import os

output_file = os.path.join(PROC_SHOCKS_DIR, 'all_shocks.cdf')

write_to_cdf(all_shocks, output_file, new_attrs, reset_index=True)

# %%



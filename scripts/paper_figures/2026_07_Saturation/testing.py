vars # -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:29:26 2025

@author: richarj2
"""
from src.methods.saturation.plotting.general import plot_driver_response

dep_vars    = ('SME', 'SMU', 'THL', 'RES')
ind_vars    = ('E_R', 'E_yz', 'B_z_GSM', 'S_x_GSE')

for dep_var in dep_vars:

    for ind_var in ind_vars:

        for region in ('sw','msh','msh20'):

            try:
                _ = plot_driver_response(ind_var, dep_var, spacecraft='combined', region=region, fit_type='saturation', iterations=2500, condition='<0', cond_col='B_z_GSM')
            except:
                continue




# %%

from src.processing.mag.indices import import_processed_index

index = 'SME'
df_pc = import_processed_index(index, resolution='1min', return_series=False, return_full=True)
df_pc[index].plot()


import matplotlib.pyplot as plt


s = df_pc[index]
# First difference in nT
ds = s.diff()

# Local baseline and anomaly
baseline = s.rolling(
    "11min", center=True, min_periods=5
).median()
anomaly = s - baseline

fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

ax[0].plot(s.index, s, lw=0.8)
ax[0].set_ylabel("Index (nT)")

ax[1].plot(ds.index, ds, lw=0.8)
ax[1].axhline(0, color="grey", lw=0.5)
ax[1].set_ylabel("Δ index (nT/min)")

ax[2].plot(anomaly.index, anomaly, lw=0.8)
ax[2].axhline(0, color="grey", lw=0.5)
ax[2].set_ylabel("Deviation (nT)")

plt.tight_layout()
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 17:22:14 2026

@author: richarj2

Geomagentic response data.
"""


# %% distributions

from src.methods.saturation.plotting.quality import plot_avg_distributions


# Don't include figure, just to discuss
# Probably can't use this will have to creata another
# Create one column for entire dataset then have a column for times when have sw/msh/msh20 etc. data to see sampling bias

plot_avg_distributions('combined', 'responses', restrict=True)



# %% uncertainties

from src.methods.saturation.plotting.uncertainties import table_response_uncertainties

# Create a figure like the driver uncertainties one, to aid with description

# Include as one combined table

_, df_sw = table_response_uncertainties(('B_z_GSM', 'E_R', 'E_yz', 'S_x_GSE'), ('SME', 'SMU', 'THL', 'RES'), region='sw')
_, df_msh = table_response_uncertainties(('B_z_GSM', 'E_R', 'E_yz', 'S_x_GSE'), ('SME', 'SMU', 'THL', 'RES'), region='msh')


# %% save

def sort_drivers(df):
    return df.sort_index(
        key=lambda idx: idx.str.lstrip('-')
    )

import pandas as pd

column = 'max_ratio'

# Reshape each DataFrame
sw_table  = sort_drivers(df_sw.pivot(index='driver', columns='response', values=column))
msh_table = sort_drivers(df_msh.pivot(index='driver', columns='response', values=column))

# Stack the two tables vertically
result = pd.concat([sw_table, msh_table], keys=['sw', 'msh'], names=['region', 'driver'])

# Save to a text file
result.to_csv(f'response_uncertainties_{column}.txt', sep='\t', float_format='%.5f')

print(result)

latex_table = result.to_latex(
    float_format='%.5f',
    escape=True,
    multirow=False,
    caption=None,
    label=None,
    column_format='llllll'
)

driver_latex = {
    '-B_z_GSM': r'-B\textsubscript{z,GSM}',
    'E_R':      r'E\textsubscript{R}',
    'E_yz':     r'E\textsubscript{yz}',
    '-S_x_GSE': r'-S\textsubscript{x,GSE}',
}

for driver, latex in driver_latex.items():
    latex_table = latex_table.replace(
        driver.replace('_', r'\_'),
        latex
    )

with open(f'response_uncertainties_{column}.tex', 'w') as f:
    f.write(latex_table)

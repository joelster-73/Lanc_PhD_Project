# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 17:53:12 2026

@author: richarj2
"""

import matplotlib.pyplot as plt

from ....plotting.utils import save_figure
from ....plotting.formatting import create_label, format_comma_integers
from ....plotting.config import blue, black

def plot_calibration(df, ind_col, good_col, total_col):

    independent     = df[ind_col]
    good_percentage = df[good_col]
    total_counts    = df[total_col]

    fig, ax = plt.subplots(figsize=(12,8), dpi=300)

    ax.plot(independent, total_counts, c=black)
    ax.set_xlabel(create_label(ind_col, units=df.attrs.get('units',{})))
    ax.set_ylabel('Total Data')
    format_comma_integers(ax, 'y')

    ax2 = ax.twinx()

    ax2.plot(independent, good_percentage, c=blue, ls='--')
    ax2.set_ylabel('Good Data [%]', color=blue)


    file_name = f'Varying_{ind_col}'

    plt.tight_layout()
    save_figure(fig, file_name=file_name, sub_directory='Calibration', overwrite=True)
    plt.show()
    plt.close()

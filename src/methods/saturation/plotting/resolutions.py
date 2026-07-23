# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 08:55:41 2026

@author: richarj2
"""

import matplotlib.pyplot as plt

from .utils import minimum_counts, def_param_names, get_variable_range, mask_df

from ..sc_delay_time import merge_with_lag

from ....plotting.utils import save_figure
from ....plotting.formatting import create_label, add_legend
from ....plotting.comparing.parameter import compare_dataframes

from ....processing.reading import import_processed_data
from ....processing.mag.indices import import_processed_index


def plot_resolutions_saturation(ind_var, dep_var, resolutions, spacecraft='omni', region='sw', lag=17, bounds=None, restrict=True, skip_zero=False, **kwargs):
    """
    Plots driver-response on one set of axes for many lag times
    To see if a particular lag time shows stronger saturation than others
    """
    kwargs['min_count'] = kwargs.get('min_count',minimum_counts['counts'])
    kwargs['display']   = kwargs.get('display','rolling')
    if kwargs['display']=='rolling':
        kwargs['region'] = ''

    cmap = plt.get_cmap('autumn_r')
    norm = plt.Normalize(vmin=0, vmax=len(resolutions)-1)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)

    for i, resolution in enumerate(resolutions):

        # Imports
        if spacecraft=='omni':
            df_sw = import_processed_data('omni', resolution=resolution)
        else:
            df_sw = import_processed_data(region, dtype='plasma', resolution=resolution, file_name=f'{region}_times_{spacecraft}')

        df_pc = import_processed_index(dep_var, resolution=resolution, return_series=False)

        ind_err, ind_count = def_param_names(df_sw, ind_var)
        bin_width, limits, invert = get_variable_range(ind_var, region, restrict=restrict, bounds=bounds)

        # Masks and slicing
        df_ind = mask_df(df_sw, ind_var, limits)
        df_dep = mask_df(df_pc, dep_var)

        df_ind, df_dep = merge_with_lag(df_ind, df_dep, lag, resolution)

        # Config
        colour = cmap(norm(i))
        kwargs['data_colour']  = colour
        kwargs['error_colour'] = colour
        kwargs['window_width'] = bin_width

        _ = compare_dataframes(df_ind, df_dep, ind_var, dep_var, col1_err=ind_err, col1_counts=ind_count, fig=fig, ax=ax, return_objs=True, **kwargs)

        ax.plot([], [], ls='-', color=colour, label=resolution)

        if invert:
            ax.invert_xaxis()

    ax.set_xlabel(create_label(ind_var,units=df_sw.attrs['units']))
    ax.set_ylabel(create_label(dep_var,units=df_pc.attrs['units']))

    add_legend(fig, ax)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()
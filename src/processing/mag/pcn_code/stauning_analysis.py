# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import os
import numpy as np

from config import DIRECTORIES
from stauning_imports import import_data, import_true_pcn
from stauning_plots import print_coeffs_monthly_ut, plot_coeff, plot_pcn_uncertainty

def coefficients_overview(source):

    print(f'----------{source}----------\n')

    source_dir = DIRECTORIES.get('analysis')
    source_dir = os.path.join(source_dir, source)
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(os.path.join(source_dir,'years'), exist_ok=True)

    df = import_data('coeff', '', source=source)

    for column in df:

        if column.endswith('covar'):
            continue

        print(f'-----{column}-----')

        df_column = df[[column]]

        if column.endswith('_var'):

            df[column.replace('_var','_unc')] = np.sqrt(df_column)

            df_column = df[[column.replace('_var','_unc')]]

        df_2d = print_coeffs_monthly_ut(df_column)
        df_2d.to_csv(os.path.join(source_dir,f'{column}.txt'), sep='\t', index=True, float_format='%.3g')

        fig = plot_coeff(coeff_data=df_column)
        fig.savefig(os.path.join(source_dir,f'{column}.png'), dpi=300, bbox_inches='tight')


    source2=None
    if source=='original':
        return
        source2 = 'true'
        pcn = import_true_pcn()

    for year in range(1997,2022):
        year = str(year)

        df_pcn = import_data('pcn', year, source)

        if source=='original':
            df_original = pcn.loc[pcn.index.year==int(year)]
        else:
            df_original = import_data('pcn', year, 'original')

        fig, _ = plot_pcn_uncertainty(df_pcn, df_original, source=source, source2=source2)
        fig.savefig(os.path.join(source_dir,'years',f'{year}.png'), dpi=300, bbox_inches='tight')


if __name__ == '__main__':


    coefficients_overview('original')
    coefficients_overview('staun_proj')
    coefficients_overview('staun_phi')
    coefficients_overview('recreated_phi')



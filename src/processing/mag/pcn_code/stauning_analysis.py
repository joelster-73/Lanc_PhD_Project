# -*- coding: utf-8 -*-
'''
Created on Sat Feb 28 14:22:13 2026

@author: richarj2
'''

import os
import numpy as np


from src.processing.mag.pcn_code.config import DIRECTORIES
from src.processing.mag.pcn_code.stauning_imports import import_data, import_true_pcn
from src.processing.mag.pcn_code.stauning_plots import plot_coeff, plot_pcn_uncertainty, plot_yearly_uncertainty, plot_unc_vs_pcn
from src.processing.mag.pcn_code.stauning_compares import print_coeffs_monthly_ut, compare_coeff

# %%
def coefficients_overview(source):

    print(f'----------{source}----------\n')

    source_dir = DIRECTORIES.get('analysis')
    source_dir = os.path.join(source_dir, source)
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(os.path.join(source_dir,'years'), exist_ok=True)

    df = import_data('coeff', '', source=source)

    for column in df:

        print(f'-----{column}-----')

        df_column = df[[column]]

        if column.endswith('_var'):

            df[column.replace('_var','_unc')] = np.sqrt(df_column)

            df_column = df[[column.replace('_var','_unc')]]

        df_2d = print_coeffs_monthly_ut(df_column)
        df_2d.to_csv(os.path.join(source_dir,f'{column}.txt'), sep='\t', index=True, float_format='%.3g')

        fig = plot_coeff(coeff_data=df_column)
        fig.savefig(os.path.join(source_dir,f'{column}.png'), dpi=600, bbox_inches='tight')

    # All coefficients
    coeffs = ['phi','a','b']
    fig = plot_coeff(coeff_data=df[coeffs])
    fig.savefig(os.path.join(source_dir,'coeff.png'), dpi=600, bbox_inches='tight')

    coeffs = ['phi_unc','a_unc','b_unc','covar']

    if source not in ('original','staun_proj'):
        fig = plot_coeff(coeff_data=df[coeffs])
        fig.savefig(os.path.join(source_dir,'coeff_var.png'), dpi=600, bbox_inches='tight')

        scale_data = {
            'phi_unc': 1,
            'a_unc':   df['a'].to_numpy(),
            'b_unc':   df['b'].to_numpy(),
            'covar':   df['a'].to_numpy() * df['b'].to_numpy()
        }
        fig = plot_coeff(coeff_data=df[coeffs], scale_data=scale_data)
        fig.savefig(os.path.join(source_dir,'coeff_var_scaled.png'), dpi=300, bbox_inches='tight')


    if source not in ('original','staun_proj'):
        compare_coeff(source)

        plot_yearly_uncertainty(source)

        plot_unc_vs_pcn(source, deg=2)



    source2='Original'
    if source=='original':
        return
        source2 = 'true'
        pcn = import_true_pcn()

    for year in range(1997,2024):
        # need to increase to include 2024 and 2025
        # 2025 not available to download
        year = str(year)

        df_pcn = import_data('pcn', year, source)

        if source=='original':
            df_original = pcn.loc[pcn.index.year==int(year)]
        else:
            df_original = import_data('pcn', year, 'original')

        fig, _ = plot_pcn_uncertainty(df_pcn, df_original, source=source, source2=source2)
        fig.savefig(os.path.join(source_dir,'years',f'{year}.png'), dpi=600, bbox_inches='tight')


# %% main

if __name__ == '__main__':

    if True:
        coefficients_overview('original')
        coefficients_overview('staun_proj')

    coefficients_overview('staun_phi')
    coefficients_overview('recreated_phi')
    coefficients_overview('updated_phi')


plot_yearly_uncertainty('staun_phi')
plot_yearly_uncertainty('recreated_phi')
plot_yearly_uncertainty('updated_phi')

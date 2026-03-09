# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:44:28 2026

@author: richarj2
"""

import os
import numpy as np

import scipy.io

from config import DIRECTORIES
from stauning_imports import import_ab

def compare_phi(year, source='staun_omni'):

    if source.lower() == 'updated_omni':
        raise ValueError ('Cannot compare npz as it fixes bug.')
    in_dir = DIRECTORIES.get(source)
    phi_np = np.load(os.path.join(in_dir, 'phis', f'Phi_{year}.npz'))['Phi'].astype(np.float32)

    phi_mat = scipy.io.loadmat(os.path.join(DIRECTORIES.get('phi'), 'phis', f'F_{year}.mat'))[f'F_{year}'].astype(np.float32)

    diff     = np.abs(phi_mat - phi_np)
    rel_diff = (diff / (np.abs(phi_mat) + 1e-10)) * 100  # avoid div by zero

    print(f'Year {year} vs MATLAB:\n'
          f'  mean abs diff = {diff.mean():.4f}°\n'
          f'  max abs diff = {diff.max():.4f}°\n'
          f'  mean rel diff = {rel_diff.mean():.4f}%\n'
          f'  within 0.1° = {(diff < 0.1).mean()*100:.1f}%\n'
          f'  [numerical precision only — LOWESS interpolation differences]\n')



def compare_ab(year, source='staun_proj'):
    ab_mat = import_ab(year)
    ab_np  = import_ab(year, source)
    for key, unit in zip(('a','b'),('mV/m/nT','nT')):
        key_mat = ab_mat[key]
        key_np  = ab_np[key]
        diff     = np.abs(key_mat - key_np)
        print(f'Year {year} vs MATLAB for {key}:')
        print(f'  mean | max abs diff = {diff.mean():.4f} | {diff.max():.4f} {unit}')
        print(f'  within 0.1 {unit:<9} = {(diff < 0.1).mean()*100:.1f}%')
        print(f'  within 0.5 {unit:<9} = {(diff < 0.5).mean()*100:.1f}%')
        print(f'  > 0.5 count        = {(diff > 0.5).sum()} / {diff.size}')
        if (diff > 0.5).any():
            idx = np.argwhere(diff > 0.5)
            print('  Large diff locations (month, time):')
            for i, j in idx[:5]:
                print(f'    [{i:02d},{j:03d}] MATLAB={key_mat[i,j]:.4f}  Python={key_np[i,j]:.4f}  diff={diff[i,j]:.4f}')
        print()



def compare_hproj(year, source='staun_phi'):
    hproj_mat = scipy.io.loadmat(os.path.join(DIRECTORIES['data'], f'Hproj_{year}.mat'))[f'Hproj_{year}'].astype(np.float32).ravel()

    compare_dir = DIRECTORIES.get(source)
    hproj_np  = np.load(os.path.join(compare_dir, 'hprojs', f'Hproj_{year}.npz'))[f'Hproj_{year}'].astype(np.float32).ravel()

    diff     = np.abs(hproj_mat - hproj_np)
    rel_diff = (diff / (np.abs(hproj_mat) + 1e-10)) * 100

    print(f'Year {year} vs MATLAB:\n'
          f'  mean abs diff = {diff.mean():.4f} nT\n'
          f'  max abs diff = {diff.max():.4f} nT\n'
          f'  mean rel diff = {rel_diff.mean():.4f}%\n'
          f'  within 1 nT = {(diff < 1).mean()*100:.1f}%\n')
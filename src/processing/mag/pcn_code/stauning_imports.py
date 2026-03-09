# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:44:21 2026

@author: richarj2
"""

import os
import numpy as np

import scipy.io

from config import DIRECTORIES

def import_phi(var, source='original'):
    """
    Imports direction angles 'phi'
    If source = original, loads in the matlab file, else laods in the pickle
    Same file to import yearly file or averaged 2d/year file
    """

    if source == 'original':
        if isinstance(var,int):
            path = os.path.join(DIRECTORIES.get('phi'), 'phis', f'F_{var}.mat')
            var = f'F_{var}'
        else:
            path = os.path.join(DIRECTORIES.get('phi'), f'Fi_{var}.mat')
            var = f'Fi_{var}'

        mat = scipy.io.loadmat(path)
        struc = mat[var]

    else:
        if source=='staun_proj':
            source = 'staun_phi'

        in_dir = DIRECTORIES.get(source)

        if isinstance(var,int):
            path = os.path.join(in_dir, 'phis', f'Phi_{var}.npz')
        else:
            path = os.path.join(in_dir, f'Phi_{var}.npz')

        data = np.load(path)
        var = 'Phi'
        struc = data[var]

    return struc


def import_ab(var, source='original'):
    """
    Imports regression coefficients a/b
    If source = original, loads in the matlab file, else laods in the pickle
    Same file to import yearly file or averaged 2d/year file
    """

    if source == 'original':
        if isinstance(var,int):
            path = os.path.join(DIRECTORIES.get('ab'), 'abs', f'ab_{var}.mat')
        else:
            path = os.path.join(DIRECTORIES.get('ab'), f'ab_{var}.mat')

        mat = scipy.io.loadmat(path)
        var = f'ab_{var}'

        struc = mat[var]
        struc = {'a': struc[0][0][0].squeeze(), 'b': struc[0][0][1].squeeze()}

    else:
        if isinstance(var,int):
            path = os.path.join(DIRECTORIES.get(source), 'abs', f'ab_{var}.npz')
        else:
            path = os.path.join(DIRECTORIES.get(source), f'ab_{var}.npz')

        data = np.load(path)

        struc = {'a': data['a'], 'b': data['b']}
        for key in ('a_var','b_var','covar'):
            if key in data:
                struc[key] = data[key]

    return struc

def import_coeff(source='original'):
    """
    Imports average coefficients phi/a/b
    If source = original, loads in the matlab file, else laods in the pickle
    Same file to import yearly file or averaged year file
    """

    if source == 'original':
        path = os.path.join(DIRECTORIES['coeff'], 'coeff.mat')
        mat = scipy.io.loadmat(path)
        struc = mat['coeff']

        struc = {'phi': struc[0][0][0].squeeze(), 'a': struc[0][0][1].squeeze(), 'b': struc[0][0][2].squeeze()}

    else:
        path = os.path.join(DIRECTORIES.get(source), 'coeff.npz')
        data = np.load(path)

        struc = {'phi': data.get('phi'), 'a': data['a'], 'b': data['b']}
        for key in ('phi_var','a_var','b_var','covar'):
            if key in data:
                struc[key] = data[key]

    return struc
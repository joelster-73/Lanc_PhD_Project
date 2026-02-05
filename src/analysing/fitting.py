# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:07:51 2025

@author: richarj2
"""
import numpy as np

from scipy.stats import lognorm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.signal import find_peaks

from uncertainties import ufloat, umath


# %% Wrapper


def fit_series(df1, col1, col2, df2=None, col1_err=None, col2_err=None, col1_counts=None, col2_counts=None, **kwargs):

    if df2 is None:
        df2 = df1

    series1 = df1.loc[:,col1]
    series2 = df2.loc[:,col2]

    kwargs['xunit'] = series1.attrs.get('units',{}).get(col1,'')
    kwargs['yunit'] = series2.attrs.get('units',{}).get(col2,'')

    if col1_err is not None:
        kwargs['xs_unc'] = df1.loc[:,col1_err]

    if col2_err is not None:
        kwargs['ys_unc'] = df2.loc[:,col2_err]

    if col1_counts is not None:
        kwargs['xs_counts'] = df1.loc[:,col1_counts]

    if col2_counts is not None:
        kwargs['ys_counts'] = df2.loc[:,col2_counts]

    return fit_function(series1, series2, **kwargs)

# %% Routines

def fit_function(xs, ys, **kwargs):

    print_text = kwargs.get('print_text',False)

    if len(xs)==0:
        print('xs is empty')
        return {}
    if len(ys)==0:
        print('ys is empty')
        return {}

    mask = (~np.isnan(xs)) & (~np.isnan(ys))
    if kwargs.get('ys_unc',None) is not None:
        mask &= ~np.isnan(kwargs['ys_unc'])
        mask &= kwargs['ys_unc']>0

    xs = xs[mask]
    ys = ys[mask]
    if kwargs.get('ys_unc',None) is not None:
        kwargs['ys_unc'] = kwargs['ys_unc'][mask]

    fit_type = kwargs.get('fit_type','straight')

    fit_dict = {}

    FIT_MAP = {
        'straight':       {'fit_func': straight_best_fit,    'func': straight_line},
        'saturation':     {'fit_func': saturation_fit,       'func': saturation},
        'gaussian':       {'fit_func': gaussian_fit,         'func': gaussian},
        'bimodal':        {'fit_func': bimodal_fit,          'func': bimodal},
        'bimodal_offset': {'fit_func': bimodal_fit_offset,   'func': bimodal_offset},
        'lognormal':      {'fit_func': lognormal_fit,        'func': lognormal},
        'linear_flat':    {'fit_func': linear_flat_fit,      'func': linear_flat},
    }

    try:
        fit_func = FIT_MAP[fit_type]['fit_func']
        func     = FIT_MAP[fit_type]['func']
    except KeyError:
        raise Exception(f'{fit_type} not valid fit type.')


    data_units = {}
    data_units['xunit'] = kwargs.get('xunit','')
    data_units['yunit'] = kwargs.get('yunit','')

    # popt, perr, ('m', 'c'), (r'{yunit} {xunit}$^{-1}$', '{yunit}')
    popt, perr, plab, punits = fit_func(xs, ys, **kwargs)
    params = {}
    units = {}

    for lab, p, err, unit in zip(plab, popt, perr, punits):
        params[lab] = ufloat(p, err)
        units[lab]  = unit.format(**data_units)

    fit_dict['params'] = params
    fit_dict['units'] = units
    fit_dict['func'] = func

    y_fit = func(xs, *popt)
    fit_dict['R2'] = r2_score(ys, y_fit)
    fit_dict['chi2'] = reduced_chi2(ys, y_fit, len(popt), kwargs.get('ys_unc',None))

    if fit_type == 'gaussian':
        peak_x = params['mu']
        peak_y = func(peak_x.n, *popt)
        fit_dict['peaks'] = [(peak_x,peak_y)]

    elif fit_type == 'lognormal':
        peak_x = umath.exp(params['mu']-(params['sigma'])**2)
        peak_y = func(peak_x.n, *popt)
        fit_dict['peaks'] = [(peak_x,peak_y)]

    elif fit_type in ('bimodal','bimodal_offset'):
        peaks = []
        for mu in ('mu1','mu2'):
            peak_x = params[mu]
            peak_y = func(peak_x.n, *popt)
            peaks.append((peak_x,peak_y))
        fit_dict['peaks'] = peaks

    # Add save_text behaviour but put into function
    if print_text:
        print(f'Best fit params for {fit_type} fit:')
        for key, value in params.items():
            unit = units[key]
            print(f'{key}: {value:P} [{unit}]')
        print(f'R²: {fit_dict["R2"]:.3g}')
        print(f'χ²ν: {fit_dict["chi2"]:.5g}')
        print()

    return fit_dict

# %% Functions

def reduced_chi2(y_obs, y_model, n_params, y_err=None):

    residuals = y_obs - y_model
    dof = len(y_obs) - n_params

    if y_err is None:
        chi2_red = np.sum(residuals**2) / dof
    else:
        chi2_red = np.sum((residuals / y_err)**2) / dof

    return chi2_red

def fit_with_errors(model_func, x, y, p0=None, bounds=(-np.inf, np.inf), yerr=None):

    try:
        popt, pcov = curve_fit(
            model_func, x, y,
            p0=p0,
            bounds=bounds,
            sigma=yerr,
            absolute_sigma=(yerr is not None)
        )

        if yerr is None:
            # Scale covariance by reduced chi-square
            chi2_red = reduced_chi2(y, model_func(x, *popt), len(popt))
            perr = np.sqrt(np.diag(pcov) * chi2_red)
        else:
            perr = np.sqrt(np.diag(pcov))

        return popt, perr

    except RuntimeError:
        print(f'Fit for {model_func.__name__} failed!')
        if p0 is None:
            return None, None
        return p0, np.zeros(len(p0))



def histogram_perc(mids, counts, perc=50):

    cum_counts = np.cumsum(counts)
    total = cum_counts[-1]
    perc_idx = np.searchsorted(cum_counts, total*(perc/100))
    percentile = mids[perc_idx]
    return percentile

def histogram_params(mids, counts):

    mean = np.average(mids, weights=counts)
    sigma = np.sqrt(np.average((mids - mean)**2, weights=counts))

    median = histogram_perc(mids, counts, 50)

    N = np.sum(counts)
    bin_width = np.median(np.diff(mids))
    amp = N * bin_width / (sigma * np.sqrt(2*np.pi))

    return mean, sigma, median, amp

# %% Straight
def straight_line(x, *params):
    if len(params)==1:
        m = params
        return m*x

    m, c = params
    return m*x + c

def straight_best_fit(x, y, **kwargs):

    yerr   = kwargs.get('ys_unc',None)
    origin = kwargs.get('origin',False)

    if origin:
        def func(xt, m):
            return m * xt
    else:
        def func(xt, m, c):
            return m * xt + c

    if origin:
        p0 = (1.0,)
    else:
        m0, c0 = np.polyfit(x, y, 1)
        p0 = (m0, c0)

    popt, perr = fit_with_errors(func, x, y, p0=p0, yerr=yerr)
    if not origin:
        return popt, perr, ('m', 'c'), (r'{yunit} ({xunit})$^{{-1}}$', '{yunit}')

    return popt, perr, ('m',), (r'{yunit} ({xunit})$^{{-1}}$',)


# %% Saturation
def saturation(x, *params):
    """
    Michaelis–Menten saturation curve:
        y = V_max * x / (K + x)
    """
    V_max, K = params
    return V_max * x / (K + x)

def saturation_fit(x, y, **kwargs):

    yerr = kwargs.get('ys_unc', None)

    def func(xt, V_max, K):
        return V_max * xt / (K + xt)

    # Initial guesses
    V_max0 = np.max(y)
    K0     = np.median(x)

    p0 = (V_max0, K0)

    popt, perr = fit_with_errors(func, x, y, p0=p0, yerr=yerr)
    return popt, perr, ('V_max', 'K'), ('{yunit}', '{xunit}')


def linear_flat(x, *params):
    """
    Linear up to a breakpoint, then flat:
        y = (y_b / x_b) * x          for x <= x_b
        y = y_b                      for x >  x_b
    """
    y_b, x_b = params
    slope = y_b / x_b
    return np.where(x <= x_b, slope * x, y_b)


def linear_flat_fit(x, y, **kwargs):

    yerr = kwargs.get('ys_unc', None)

    def func(xt, y_b, x_b):
        slope = y_b / x_b
        return np.where(xt <= x_b, slope * xt, y_b)

    # Initial guesses
    x_b0 = np.median(x)
    y_b0 = np.interp(x_b0, x, y)

    p0 = (y_b0, x_b0)

    popt, perr = fit_with_errors(func, x, y, p0=p0, yerr=yerr)
    return popt, perr, ('y_b', 'x_b'), ('{yunit}', '{xunit}')


# %% Gaussian
def gaussian(x, *params):
    A, mu, sig = params
    return A * np.exp(-((x - mu) ** 2) / (2 * sig ** 2))

def gaussian_fit(x, y, **kwargs):

    yerr          = kwargs.get('ys_unc',None)
    simple_bounds = kwargs.get('simple_bounds',False)

    # Initial guesses
    mean_guess, std_guess, median, amplitude_guess = histogram_params(x, y)

    initial_guess = (amplitude_guess, mean_guess, std_guess)
    # stops negative As and sigs
    if simple_bounds:
        bounds = ([0, np.min(x), 0],
                  [np.inf, np.max(x), np.inf])
    else:
        bounds = ([amplitude_guess*0.75, mean_guess-std_guess, 0],
                  [amplitude_guess*1.25, mean_guess+std_guess, 2*std_guess])

    # Perform the curve fitting
    popt, perr = fit_with_errors(gaussian, x, y, p0=initial_guess, bounds=bounds, yerr=yerr)
    return popt, perr, ('A','mu','sigma'), ('{yunit}', '{xunit}', '{xunit}')


# %% Bimodal

def rough_peaks(x,y,npeaks=2):

    prom_perc = 0.1
    for i in range(10):
        mask = (y!=0) & (y>0.0005*np.max(y))
        peaks, _ = find_peaks(y[mask],prominence=prom_perc*np.max(y))
        if len(peaks)<npeaks:
            prom_perc *= 0.9
        elif len(peaks)>npeaks:
            prom_perc *= 1.1
        else:
            return x[mask][peaks]

    return (histogram_perc(x,y,40),histogram_perc(x,y,60))

def bimodal(x, *params):
    A1, mu1, sig1, A2, mu2, sig2 = params
    return gaussian(x, A1, mu1, sig1) + gaussian(x, A2, mu2, sig2)

def bimodal_fit(x, y, **kwargs):

    yerr          = kwargs.get('ys_unc',None)
    simple_bounds = kwargs.get('simple_bounds',False)

    # Initial guesses
    x1_guess, x2_guess = rough_peaks(x,y)
    mean, sigma, median, amplitude_guess = histogram_params(x, y)

    std_guess = np.sqrt(sigma**2 - np.abs(x1_guess-x2_guess)*2 / 4)

    initial_guess = (amplitude_guess, x1_guess, std_guess, amplitude_guess, x2_guess, std_guess)

    # stops negative As and sigs
    if simple_bounds:
        bounds = ([0, np.min(x), 0, 0, median, 0],
                  [np.inf, median, np.inf, np.inf, np.max(x), np.inf])

    else:
        bounds = ([0.5*amplitude_guess, x1_guess-std_guess, 0.4*sigma, 0.5*amplitude_guess, median, 0.4*sigma],
                  [2.0*amplitude_guess, median, sigma, 2.0*amplitude_guess, x2_guess+std_guess,    sigma])

    # Perform the curve fitting
    popt, perr = fit_with_errors(bimodal, x, y, p0=initial_guess, bounds=bounds, yerr=yerr)
    return popt, perr, ('A1', 'mu1', 'sigma1', 'A2', 'mu2', 'sigma2'), ('{yunit}', '{xunit}', '{xunit}', '{yunit}', '{xunit}', '{xunit}')



def bimodal_offset(x, *params):
    A1, mu1, sig1, A2, mu2, sig2, c = params
    return gaussian(x, A1, mu1, sig1) + gaussian(x, A2, mu2, sig2) + c


def bimodal_fit_offset(x, y, **kwargs):

    yerr          = kwargs.get('ys_unc',None)
    simple_bounds = kwargs.get('simple_bounds',True)

    # Initial guesses
    x1_guess, x2_guess = rough_peaks(x,y)

    mean, sigma, median, amplitude = histogram_params(x, y)

    std_guess = np.sqrt(sigma**2 - np.abs(x1_guess-x2_guess)*2 / 4)
    offset_guess = np.min(y)
    amplitude_guess = amplitude - offset_guess

    initial_guess = (amplitude_guess, x1_guess, std_guess, amplitude_guess, x2_guess, std_guess, offset_guess)

    # stops negative As and sigs
    if simple_bounds:
        bounds = ([0, np.min(x), 0, 0, median, 0, 0],
                  [np.inf, median, np.inf, np.inf, np.max(x), np.inf, 1.1*np.min(y)])

    else:
        bounds = ([0.5*amplitude_guess, np.min(x), 0.5*sigma, 0.5*amplitude_guess, median, 0.5*sigma, 0.8*np.min(y)],
                  [2.0*amplitude_guess, median, sigma, 2.0*amplitude_guess, np.max(x), sigma, 1.1*np.min(y)])

    # Perform the curve fitting
    popt, perr = fit_with_errors(bimodal_offset, x, y, p0=initial_guess, bounds=bounds, yerr=yerr)
    return popt, perr, ('A1', 'mu1', 'sigma1', 'A2', 'mu2', 'sigma2', 'c'), ('{yunit}', '{xunit}', '{xunit}', '{yunit}', '{xunit}', '{xunit}', '{yunit}')


# %% Lognormal
def lognormal(x, *params):
    A, mu, sigma = params
    scale = np.exp(mu)
    return A * lognorm.pdf(x, sigma, scale=scale)

def lognormal_fit(x, y, **kwargs):

    yerr          = kwargs.get('ys_unc',None)
    # Initial guesses
    _, sigma_guess, median, amplitude_guess = histogram_params(x, y)

    #amplitude_guess = np.max(y)
    mu_guess = np.log(median)  # Use median for scale as an initial guess
    initial_guess = (amplitude_guess, mu_guess, sigma_guess)

    # Perform the curve fitting
    popt, perr = fit_with_errors(lognormal, x, y, p0=initial_guess, bounds=(0, np.inf), yerr=yerr)
    return popt, perr, ('A', 'mu', 'sigma'), ('{yunit}', '{xunit}', '{xunit}')




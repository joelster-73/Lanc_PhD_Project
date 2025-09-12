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


def fit_function(xs, ys, print_text=False, **kwargs):


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

    if fit_type=='straight':
        fit_func = straight_best_fit
        func = straight_line

    elif fit_type=='gaussian':
        fit_func = gaussian_fit
        func = gaussian

    elif fit_type=='bimodal':
        fit_func = bimodal_fit
        func = bimodal

    elif fit_type=='bimodal_offset':
        fit_func = bimodal_fit_offset
        func = bimodal_offset

    elif fit_type=='lognormal':
        fit_func = lognormal_fit
        func = lognormal
    else:
        raise Exception(f'{fit_type} not valid fit type.')

    popt, perr, plab = fit_func(xs, ys, **kwargs)
    params = {}
    for lab, p, err in zip(plab, popt, perr):
        params[lab] = ufloat(p, err)

    fit_dict['params'] = params
    fit_dict['func'] = func

    y_fit = func(xs, *popt)
    r_squared = r2_score(ys, y_fit)
    fit_dict['R2'] = r_squared

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
        print('Best fit params for {fit_type} fit:')
        for key, value in params.items:
            print(f'{key}: ${value:L}$')
        print(f'$R^2$; {r_squared:.3g}')

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
            residuals = y - model_func(x, *popt)
            dof = len(y) - len(popt)
            chi2_red = np.sum(residuals**2) / dof
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
    return popt, perr, ('m', 'c') if not origin else ('m',)


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
    return popt, perr, ('A','mu','sigma')


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
    return popt, perr, ('A1', 'mu1', 'sigma1', 'A2', 'mu2', 'sigma2')



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
    return popt, perr, ('A1', 'mu1', 'sigma1', 'A2', 'mu2', 'sigma2', 'c')


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
    return popt, perr, ('A', 'mu', 'sigma')



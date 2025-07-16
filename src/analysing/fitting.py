# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:07:51 2025

@author: richarj2
"""
import numpy as np

from scipy.stats import norm, lognorm
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from uncertainties import ufloat, umath

from .utils import save_statistics


def straight_best_fit(x, y, yerr=None, name='', detailed=False, print_text=False, origin=False):
    """
    Performs a straight-line (linear) best fit to the provided data points,
    returning the slope and intercept along with their uncertainties.

    Parameters
    ----------
    x : numpy.array
        The independent variable data points (x-values) to which the straight-line
        fit is applied.

    y : numpy.array
        The dependent variable data points (y-values) corresponding to the x-values.

    yerr : numpy.array, optional
        The uncertainties in the y-values. If provided, these are used to weight the
        fitting process, giving more importance to data points with smaller errors.
        Defaults to None, which assumes equal weighting for all data points.

    Returns
    -------
    a : float
        The slope of the best-fit line.

    b : float
        The intercept of the best-fit line.
    """
    if name != '':
        name += ' '
    if origin:
        def func(xt,m):
            return m*xt
        popt, pcov = curve_fit(func,x,y,sigma=yerr,absolute_sigma=True)
        a = popt[0]
        b = 0
        a_unc = ufloat(popt[0],pcov[0,0])
        b_unc = ufloat(0,0)
        y_fit = func(x,a)
    else:
        if yerr is not None:
            coeffs, cov = np.polyfit(x, y, 1, w=1/yerr, cov=True) # Weighted
        else:
            coeffs, cov = np.polyfit(x, y, 1, cov=True)

        a, b = coeffs
        a_unc = ufloat(a,np.sqrt(cov[0,0]))
        b_unc = ufloat(b,np.sqrt(cov[1,1]))
        y_fit = np.poly1d(coeffs)(x)

    r_squared = r2_score(y,y_fit)
    minx, maxx = np.min(x), np.max(x)
    text = (f'{name}\n'
            f'Best Fit: y = ({a_unc:P})x + ({b_unc:P})\n'
            f'First: ({minx:.3g}, {(a_unc*minx+b_unc).n:.3g})\n'
            f'Last: ({maxx:.3g}, {(a_unc*maxx+b_unc).n:.3g})\n'
            f'R²: {r_squared:.3g}\n')
    if print_text:
        print(text)
    #save_statistics(text)

    if detailed:
        return a_unc, b_unc, r_squared

    return a, b, r_squared


def gaussian(x, A, mu, sig):
    return A * np.exp(-((x - mu) ** 2) / (2 * sig ** 2))

def gaussian_fit(x, y, normal=False, name='', detailed=False, print_text=False):
    """
    Fits a Gaussian distribution to the data.

    Parameters
    ----------
    x : numpy.array
        The independent variable data.
    y : numpy.array
        The dependent variable data.
    norm : bool, optional
        If True, fits a normal distribution (fixed amplitude).
        If False, includes amplitude as a fitting parameter.

    Returns
    -------
    tuple
        If norm=True: (mean, stddev)
        If norm=False: (amplitude, mean, stddev)
    """
    if name != '':
        name += ' '
    if normal:
        # Fit a normal distribution to the data
        mean, stddev = norm.fit(x)
        return mean, stddev
    else:

        # Initial guesses
        non_zero = x[y!=0]
        mean_guess = np.mean(non_zero)
        std_guess = np.std(non_zero)

        amplitude_guess = np.max(y) / np.exp(-(1 / (2 * std_guess ** 2)))

        q1 = np.percentile(non_zero,25)
        q3 = np.percentile(non_zero,75)
        iqr = q3 - q1

        initial_guess = (amplitude_guess, mean_guess, std_guess)
        bounds = ([amplitude_guess*0.75, q1, 0],  # stops negative As and sigs
                  [amplitude_guess*1.25, q3, 2*iqr])

        try:
            # Perform the curve fitting
            popt, pcov = curve_fit(gaussian, x, y, p0=initial_guess, bounds=bounds)
            A, mu, sigma = popt

            perr = np.sqrt(np.diag(pcov))
            A_unc = ufloat(A,perr[0])
            mu_unc = ufloat(mu,perr[1])
            sig_unc = ufloat(sigma,perr[2])

            x_median = np.median(x)
            text = (f'{name}\n'
                    f'Best Fit for "A ⋅ exp(-((x - μ)²) / (2σ²))":\n'
                    f'A:   {A_unc:P}\n'
                    f'mu:  {mu_unc:P}\n'
                    f'sig: {sig_unc:P}\n'
                    f'med: {x_median:.3g}\n')
            if print_text:
                print(text)
            if detailed:
                return A_unc, mu_unc, sig_unc
            return A, mu, abs(sigma)  # Ensure non-negative sigma
        except RuntimeError:
            print('Gaussian fit failed!')
            print(initial_guess)
            return initial_guess


def bimodal(x, A1, mu1, sig1, A2, mu2, sig2):
    return gaussian(x, A1, mu1, sig1) + gaussian(x, A2, mu2, sig2)

def bimodal_fit(x, y, symmetric=False, name='', detailed=False, print_text=False, simple_bounds=False):
    """
    Fits a Gaussian distribution to the data.

    Parameters
    ----------
    x : numpy.array
        The independent variable data.
    y : numpy.array
        The dependent variable data.
    norm : bool, optional
        If True, fits a normal distribution (fixed amplitude).
        If False, includes amplitude as a fitting parameter.

    Returns
    -------
    tuple
        If norm=True: (mean, stddev)
        If norm=False: (amplitude, mean, stddev)
    """
    if name != '':
        name += ' '

    # Initial guesses
    non_zero = x[y!=0]
    std_guess = np.std(non_zero) / 4
    x1_guess = - std_guess
    x2_guess = std_guess
    amplitude_guess = np.max(y) / np.exp(-(1 / (2 * std_guess ** 2)))

    initial_guess = (amplitude_guess, x1_guess, std_guess, amplitude_guess, x2_guess, std_guess)

    if not simple_bounds:

        bounds = ([amplitude_guess*0.75, np.percentile(non_zero,25), 0, amplitude_guess*0.75, np.percentile(non_zero,25), 0],  # stops negative As and sigs
              [amplitude_guess*1.25, np.percentile(non_zero,75), np.inf, amplitude_guess*1.25, np.percentile(non_zero,75), np.inf])
    else:
        bounds = ([0, np.percentile(non_zero,0), 0, 0, np.percentile(non_zero,0), 0],  # stops negative As and sigs
                  [np.inf, np.percentile(non_zero,100), np.inf, np.inf, np.percentile(non_zero,100), np.inf])


    try:
        # Perform the curve fitting
        popt, pcov = curve_fit(bimodal, x, y, p0=initial_guess, bounds=bounds)
        A1, mu1, sigma1, A2, mu2, sigma2 = popt
        perr = np.sqrt(np.diag(pcov))
        A1_unc = ufloat(A1,perr[0])
        mu1_unc = ufloat(mu1,perr[1])
        sig1_unc = ufloat(sigma1,perr[2])
        A2_unc = ufloat(A2,perr[3])
        mu2_unc = ufloat(mu2,perr[4])
        sig2_unc = ufloat(sigma2,perr[5])

        text = (f'{name}\n'
                f'Best Fit for "A1 ⋅ exp(-((x - μ1)²) / (2σ1²)) + A2 ⋅ exp(-((x - μ2)²) / (2σ2²))":\n'
                f'A1:   {A1_unc:P}\n'
                f'mu1:  {mu1_unc:P}\n'
                f'sig1: {sig1_unc:P}\n'
                f'A2:   {A2_unc:P}\n'
                f'mu2:  {mu2_unc:P}\n'
                f'sig2: {sig2_unc:P}\n')
        if print_text:
            print(text)
        save_statistics(text)
        if detailed:
            return A1_unc, mu1_unc, sig1_unc, A2_unc, mu2_unc, sig2_unc
        return A1, mu1, sigma1, A2, mu2, sigma2  # Ensure non-negative sigma
    except RuntimeError:
        print('Gaussian fit failed!')
        print(initial_guess)
        if detailed:
            return [ufloat(x,0) for x in initial_guess]
        return initial_guess

def bimodal_offset(x, A1, mu1, sig1, A2, mu2, sig2, c):
    return gaussian(x, A1, mu1, sig1) + gaussian(x, A2, mu2, sig2) + c

def bimodal_fit_offset(x, y, symmetric=False, name='', detailed=False, print_text=False):
    """
    Fits a Gaussian distribution to the data.

    Parameters
    ----------
    x : numpy.array
        The independent variable data.
    y : numpy.array
        The dependent variable data.
    norm : bool, optional
        If True, fits a normal distribution (fixed amplitude).
        If False, includes amplitude as a fitting parameter.

    Returns
    -------
    tuple
        If norm=True: (mean, stddev)
        If norm=False: (amplitude, mean, stddev)
    """
    if name != '':
        name += ' '

    # Initial guesses
    amplitude_guess = max(y) - min(y)  # Approximate peak of y
    offset_guess = min(y)

    std_guess = np.std(x)
    x1_guess = np.mean(x) - std_guess / 2
    x2_guess = np.mean(x) + std_guess /2

    initial_guess = (amplitude_guess, x1_guess, std_guess, amplitude_guess, x2_guess, std_guess, offset_guess)
    bounds = ([0, -np.inf, 0, 0, -np.inf, 0, 0],  # stops negative As and sigs
          [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])

    try:
        # Perform the curve fitting
        popt, pcov = curve_fit(bimodal_offset, x, y, p0=initial_guess, bounds=bounds)
        A1, mu1, sigma1, A2, mu2, sigma2, c = popt
        perr = np.sqrt(np.diag(pcov))

        A1_unc = ufloat(A1,perr[0])
        mu1_unc = ufloat(mu1,perr[1])
        sig1_unc = ufloat(sigma1,perr[2])
        A2_unc = ufloat(A2,perr[3])
        mu2_unc = ufloat(mu2,perr[4])
        sig2_unc = ufloat(sigma2,perr[5])
        c_unc = ufloat(c,perr[6])

        text = (f'{name}\n'
                f'Best Fit for "A1 ⋅ exp(-((x - μ1)²) / (2σ1²)) + A2 ⋅ exp(-((x - μ2)²) / (2σ2²))":\n'
                f'A1:   {A1_unc:P}\n'
                f'mu1:  {mu1_unc:P}\n'
                f'sig1: {sig1_unc:P}\n'
                f'A2:   {A2_unc:P}\n'
                f'mu2:  {mu2_unc:P}\n'
                f'sig2: {sig2_unc:P}\n'
                f'c:    {c_unc:P}\n')

        if print_text:
            print(text)
        save_statistics(text)
        if detailed:
            return A1_unc, mu1_unc, sig1_unc, A2_unc, mu2_unc, sig2_unc, c_unc
        return A1, mu1, sigma1, A2, mu2, sigma2, c  # Ensure non-negative sigma
    except RuntimeError:
        print('Gaussian fit failed!')
        if detailed:
            return [ufloat(x,0) for x in initial_guess]
        return initial_guess




def lognormal(x, A, mu, sigma):
    scale = np.exp(mu)
    s = sigma
    return A * lognorm.pdf(x, s, scale=scale)

def lognormal_fit(x, y, name='', detailed=False, print_text=False):
    if name != '':
        name += ' '

    # Initial guesses
    amplitude_guess = max(y)
    mu_guess = np.log(np.median(x))  # Use median for scale as an initial guess
    sigma_guess = 1  # Typical initial guess for shape (s)
    initial_guess = (amplitude_guess, mu_guess, sigma_guess)

    try:
        # Perform the curve fitting
        popt, pcov = curve_fit(lognormal, x, y, p0=initial_guess, bounds=(0, np.inf))
        A, mu, sigma = popt
        perr = np.sqrt(np.diag(pcov))

        A_unc     = ufloat(A, perr[0])
        mu_unc    = ufloat(mu, perr[1])
        sig_unc   = ufloat(sigma, perr[2])
        mode_unc  = umath.exp(mu_unc-sig_unc**2)
        text = (f'{name}\n'
                f'Best Fit for Log-normal:\n'
                f'A:   {A_unc:P}\n'
                f'mu:  {mu_unc:P}\n'
                f'sig: {sig_unc:P}\n')
        if print_text:
            print(text)
        save_statistics(text)

        if detailed:
            return A_unc, mu_unc, sig_unc, mode_unc
        return A, mu, sigma, np.exp(mu-sigma**2)
    except RuntimeError:
        print('Log-normal fit failed!')
        return initial_guess

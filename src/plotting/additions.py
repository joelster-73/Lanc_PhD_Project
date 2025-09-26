# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:37:12 2025

@author: richarj2
"""
import numpy as np
from datetime import timedelta
from pandas import Timedelta

from matplotlib.patches import Wedge, Circle
from .utils import segment_dataframe

def plot_segments(ax, data, colour, label, name, lw=0.6, fmt='-', marker=None, delta=Timedelta(minutes=1)):
    data = segment_dataframe(data, delta)
    label_shown = False
    if fmt == 'o':
        marker = 'o'
        lw = 0.2
        fmt = ':'
    for _, segment in data.groupby('segment'):
        if not label_shown:
            ax.plot(segment[name], c=colour, lw=lw, ls=fmt, marker=marker, label=label)
            label_shown = True
        else:
            ax.plot(segment[name], c=colour, lw=lw, ls=fmt, marker=marker)

def create_circle(ax, centre=(0,0), radius=1, colour='black'):
    circle = Circle(centre, radius, facecolor=colour, edgecolor='grey')
    ax.add_patch(circle)


def create_quarter_circle_marker(ax, centre=(0,0), radius=1, angle_start=90):

    # Create one semi-circle
    white_half = Wedge(centre, radius, angle_start - 90, angle_start, facecolor='w', edgecolor='k', clip_on=False)

    ax.add_patch(white_half)


def create_half_circle_marker(ax, centre=(0,0), radius=1, angle_start=90, full=True):
    """
    Creates a half-circle Earth marker (half white, half black).
    """
    # Create two semi-circles
    if full:
        black_half = Wedge(centre, radius, angle_start, angle_start + 180, facecolor='k', edgecolor='k', clip_on=False)
        white_half = Wedge(centre, radius, angle_start + 180, angle_start + 360, facecolor='w', edgecolor='k', clip_on=False)
        ax.add_patch(black_half)
    else:
        white_half = Wedge(centre, radius, angle_start, angle_start+180, facecolor='w', edgecolor='k', clip_on=False)

    ax.add_patch(white_half)


def plot_error_region(ax, xs, ys, y_errs, c='k', alpha=0.1, marker='x', label=None, step=None):

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    y_errs = np.array(y_errs, dtype=float)

    ax.fill_between(xs, ys-y_errs, ys+y_errs, color=c, alpha=alpha, step=step)
    ax.plot(xs, ys, marker=marker, c=c, label=label)


def plot_vertical_line_unc(ax, time, uncertainty, label_info=None, colour='k', linestyle='--', uncertainty_tuple=None, return_label=False):

    minutes, seconds = divmod(int(uncertainty), 60) # uncertainty in seconds

    formatted_uncertainty = f'{minutes:02}:{seconds:02}'
    line_label = (
        time.strftime('%H:%M:%S') + r' $\pm$ ' + formatted_uncertainty
    )

    if label_info is not None:
        line_label = f'{label_info}: ' + line_label

    dt = timedelta(seconds=uncertainty)
    if uncertainty_tuple is not None:
        dt_left  = timedelta(seconds=uncertainty_tuple[0])
        dt_right = timedelta(seconds=uncertainty_tuple[1])
    else:
        dt_left, dt_right = dt, dt

    ax.axvspan(time-dt_left, time+dt_right, color=colour, alpha=0.08)

    if return_label:
        ax.axvline(x=time, c=colour, ls=linestyle, lw=0.5)
        return line_label

    ax.axvline(x=time, c=colour, ls=linestyle, lw=0.5, label=line_label)
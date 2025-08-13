# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:37:12 2025

@author: richarj2
"""
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

def create_circle(ax, centre, radius, colour='black'):
    circle = Circle(centre, radius, facecolor=colour, edgecolor='grey')
    ax.add_patch(circle)


def create_half_circle_marker(ax, centre, radius, angle_start=90, full=True):
    """
    Creates a half-circle Earth marker (half white, half black).
    """
    # Create two semi-circles
    if full:
        black_half = Wedge(centre, radius, angle_start, angle_start + 180, facecolor='k', edgecolor='k', clip_on=False)
        white_half = Wedge(centre, radius, angle_start + 180, angle_start + 360, facecolor='w', edgecolor='k', clip_on=False)
    else:
        black_half = Wedge(centre, radius, angle_start, angle_start + 90, facecolor='k', edgecolor='k', clip_on=False)
        white_half = Wedge(centre, radius, angle_start - 90, angle_start, facecolor='w', edgecolor='k', clip_on=False)

    ax.add_patch(black_half)
    ax.add_patch(white_half)

def plot_error_region(ax, xs, ys, y_errs, c='k', alpha=0.1, marker='x', label=None, step=None):

    ax.fill_between(xs, ys-y_errs, ys+y_errs, color=c, alpha=alpha, step=step)
    ax.plot(xs, ys, marker=marker, c=c, label=label)
# -*- coding: utf-8 -*-
"""
Created on Fri May 16 11:24:05 2025

@author: richarj2
"""

import re
import matplotlib.dates as mdates


def check_labels(df, *labels):
    for label in labels:
        if label not in df.keys():
            raise ValueError(f'Field data "{label}" not found in data.')


def dark_mode_fig(fig,black,white,heat=False):
    fig.patch.set_facecolor(white)
    for ax in fig.get_axes():
        if heat:
            ax.set_facecolor('k')
        ax.tick_params(axis='x', which='both', labelbottom=True)
        ax.tick_params(axis='both', labelcolor=black, colors=black)
        for spine in ax.spines.values():
            spine.set_edgecolor(black)

def add_figure_title(fig,black='k',title=None,x_label=None,y_label=None,ax=None,name_latex=False):

    if title is None:
        if x_label is not None and y_label is not None:
            if name_latex:
                title = f'${y_label}$ against ${x_label}$'
            else:
                title = f'{y_label} against {x_label}'
        else:
            title = 'Figure'
    if ax is not None:
        ax.set_title(title, color=black, wrap=True)
    else:
        fig.suptitle(title, color=black, wrap=True)

def add_legend(fig, ax, legend_on=True, loc='upper left', anchor=None, cols=1, heat=False, edge_col='k', frame_on=True, title=None, rows=None):
    num_labels = len(ax.get_legend_handles_labels()[0])
    if legend_on and num_labels>0:
        if cols==-1:
            cols = num_labels
        if rows is not None:
            cols = max(num_labels // rows,1)
        label_colour = 'k'
        face_colour = 'w'
        if heat:
            label_colour = 'w'
            face_colour = 'k'

        if loc == 'split':
            handles, labels = ax.get_legend_handles_labels()
            half = len(handles)//2
            legend1 = ax.legend(handles[:half],labels[:half],
                                loc='upper left', bbox_to_anchor=anchor, ncols=cols, frameon=frame_on,
                                labelcolor=label_colour, facecolor=face_colour, edgecolor=edge_col, handlelength=0, title=title)
            legend1.set_zorder(5)
            ax.add_artist(legend1)

            legend2 = ax.legend(handles[half:],labels[half:],
                                loc='upper right', bbox_to_anchor=anchor, ncols=cols, frameon=frame_on,
                                labelcolor=label_colour, facecolor=face_colour, edgecolor=edge_col, title=title)
            legend2.set_zorder(5)

        else:
            legend = ax.legend(loc=loc, bbox_to_anchor=anchor, ncols=cols, frameon=frame_on,
                           labelcolor=label_colour, facecolor=face_colour, edgecolor=edge_col, title=title)
            legend.set_zorder(5)


def format_string(s):
    # make so instead of first character, everything before 'mag' is put in ||
    # do something similar for avg and put it under a bar

    if 'mag' in s or 'avg' in s:
        s = f'|{s[0]}|' + s[5:]

    #s = s.replace('r_rho', r'\\rho')
    s = s.replace('r_rho', r'\sqrt{Y^2+Z^2}')
    s = s.replace('r_x', 'X')
    s = s.replace('r_y', 'Y')
    s = s.replace('r_z', 'Z')

    # List of common Greek letters
    greek_letters = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta',
                     'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi',
                     'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi', 'omega', 'Delta',
                     'parallel', 'perp') # added to regex

    # Create a regex pattern to match Greek letters anywhere
    greek_pattern = r'(' + '|'.join(greek_letters) + r')'

    s = re.sub(greek_pattern, r'\\\1', s)

    # Split the string on the first underscore
    parts = s.split('_', 1)

    if len(parts) > 1:
        # Replace underscores with commas in the part after the first part
        subscript = parts[1].replace('_', ',')

        # Use regex to find and replace Greek letters with their LaTeX format
        #subscript = re.sub(greek_pattern, r'\\\1', subscript)

        return f'{parts[0]}_{{{subscript}}}'
    else:
        return s  # Return as is if there's no underscore

def data_string(input_string):
    """
    Converts string with multiple 'parts'
    """
    # Split the input string by spaces and process each part with data_string
    parts = input_string.split()
    processed_parts = [format_string(part) for part in parts]

    # Join the processed parts with spaces
    return ' '.join(processed_parts)

def create_label(column, unit=None, data_name=None, name_latex=False, units=None):
    if data_name is not None and name_latex:
        label = f'${data_string(data_name)}$'
    elif data_name is not None:
        label = f'{data_name}'
    else:
        label = f'${data_string(column)}$'

    if unit is None and units is not None:
        unit = units.get(column,None)
    if unit is not None:
        if unit == 'Re':
            label += r' [$\mathrm{R_E}$]'
        elif unit not in ('','1','STRING','LIST','NUM'):
            label += f' [{unit}]'

    return label

def ordinal_suffix(n):
    if 10 <= n % 100 <= 20:  # Special case for 11th, 12th, 13th, etc.
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def common_prefix(x, y):
    # Determine the minimum length to avoid index errors
    min_length = min(len(x), len(y))

    # Iterate and compare characters
    for i in range(min_length):
        if x[i] != y[i]:
            return x[:i]  # Return the common prefix up to the mismatch

    return x[:min_length]  # Entire shorter string is the common prefix


def array_to_string(arr, fmt='.1f'):
    return f'[{", ".join(f"{value:{fmt}}" for value in arr)}]'

def custom_date_formatter(x, pos):
    timestamp = mdates.num2date(x)
    if pos == 0 or (timestamp.hour == 0 and timestamp.minute == 0 and timestamp.second == 0):
        return timestamp.strftime('%H:%M\n%d %b')
        #return timestamp.strftime('%H:%M\n%Y-%m-%d')
    return timestamp.strftime('%H:%M')


# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:42:37 2025

@author: richarj2
"""
import numpy as np
import pandas as pd
from pandas import Timedelta

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import ticker as mticker

from .config import black, white, blue

from .additions import create_circle, create_half_circle_marker, create_quarter_circle_marker, segment_dataframe, plot_segments
from .formatting import add_legend, add_figure_title, create_label, check_labels, dark_mode_fig, data_string
from .utils import save_figure, calculate_bins

from ..coordinates.boundaries import msh_boundaries

# %% Time_distributions

def plot_dataset_years_months(df, **kwargs):

    fig, axs = plt.subplots(1, 2, figsize=(14,5))

    _ = plot_time_distribution(df, timing='year', want_legend=False, fig=fig, ax=axs[0], return_objs=True, **kwargs)

    _ = plot_time_distribution(df, timing='month', column_text='perc', fig=fig, ax=axs[1], return_objs=True, **kwargs)

    axs[1].set_ylabel(None)
    add_legend(fig, axs[1], loc='upper right',edge_col=white)

    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()


def plot_time_distribution(series, **kwargs):

    colour      = kwargs.get('colour','k')
    timing      = kwargs.get('timing','year')
    brief_title = kwargs.get('brief_title',f'{timing.capitalize()}s')
    print_data  = kwargs.get('print_data',False)
    column_text = kwargs.get('column_text',None)
    want_legend = kwargs.get('want_legend',True)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    if timing=='year':
        data = series.index.year
    elif timing=='month':
        data = series.index.month
    series = data.to_numpy()

    series = series[~np.isnan(series)]
    bin_edges = range(np.min(series),np.max(series)+2,1)

    ###-------------------Plotting-------------------##

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    counts, bins, _ = ax.hist(series, bins=bin_edges, alpha=1.0, color=colour, edgecolor='grey')

    #ax.set_xlabel(timing.capitalize(),c=black)

    if timing=='month':
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        ax.set_xticks(ticks=np.arange(1, 13))
        label_positions = np.arange(1, 13) + 0.5
        for pos, label in zip(label_positions, month_names):
            ax.text(pos, -0.025, label, ha='center', va='top', transform=ax.get_xaxis_transform())
        ax.set_xticklabels([])

    ax.set_ylabel('Counts',c=black)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    ax.plot([],[],' ',label=f'{len(series):,} mins')

    if print_data:
        ordered_counts = data.value_counts().sort_index()
        total_rows = ordered_counts.sum()

        summary_df = pd.DataFrame({
            timing: ordered_counts.index.astype(str),
            'row_count': ordered_counts.values,
            'proportion': ordered_counts.values / total_rows * 100
        })

        total_row = pd.DataFrame({
            timing: ['Total'],
            'row_count': [total_rows],
            'proportion': [100]
        }, index=[100])
        summary_df = pd.concat([summary_df, total_row])

        print(summary_df)


    if column_text is not None:
        def format_label(v, d_type='perc'):
            if v==0:
                return ''

            if d_type=='perc':
                if v < 0.05:
                    return f'{v:.2f}%'
                elif v < 10:
                    return f'{v:.1f}%'
                else:
                    return f'{v:.2g}%'

            return f'{v:,}'

        column_data = counts.copy()
        if column_text=='perc':
            column_data /= (np.sum(counts) / 100)
        column_labels = np.array([format_label(datum, d_type=column_text) for datum in column_data])

        mids = (bins[:-1] + bins[1:]) / 2

        for centre, count, label in zip(mids, counts, column_labels):
            ax.text(centre, count, label, ha='center', va='bottom', fontsize=10)

    add_figure_title(fig, black, brief_title, ax=ax)
    add_legend(fig, ax, loc='upper right', edge_col='w', legend_on=want_legend)
    dark_mode_fig(fig,black,white)
    if return_objs:
        return fig, ax

    plt.tight_layout()

    save_figure(fig)
    plt.show()
    plt.close()

# %% Orbits
def plot_orbit(df, plane='yz', coords='GSE', **kwargs):

    df = df.copy()
    display       = kwargs.get('display','Scatter')
    bin_width     = kwargs.get('bin_width',None)
    x_name        = kwargs.get('x_name',None)
    y_name        = kwargs.get('y_name',None)
    show_key      = kwargs.get('show_key',False)

    sc_key        = kwargs.get('sc_key', None)
    models        = kwargs.get('models', 'None')
    df_omni       = kwargs.get('df_omni',None)

    regions       = kwargs.get('region_nums',None)
    region_labels = kwargs.get('region_labels',None)

    equal_axes    = kwargs.get('equal_axes',True)
    signed_rho    = kwargs.get('signed_rho',None)
    centre_Earth  = kwargs.get('centre_Earth','full')
    nose_text     = kwargs.get('nose_text',None)

    brief_title   = kwargs.get('brief_title',None)
    want_legend   = kwargs.get('want_legend',False)

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    is_heat = False
    if display == 'Heat':
        is_heat = True

    if brief_title is None:
        start_year = df.index[0].strftime('%Y')
        end_year = df.index[-1].strftime('%Y')
        brief_title = f'Orbit from {start_year} to {end_year} in {plane} plane.'

    plane = plane[::-1] if plane in ('zy','yx','zx') else plane
    if plane=='yz' and models!='None':
        print('MSH models not plotted in yz plane.')
        models = 'None'
    if plane in ('yz','xz'):
        equal_axes = False

    ###-------------------VALIDATE INPUTS AND CONFIGURE PLOT KEYS-------------------###

    if plane == 'x-rho':
        x_comp = 'x'
        y_comp = 'rho'
        x_label = f'r_x_{coords}'
        y_label = 'r_rho'

        if sc_key is not None:
            x_label = f'{x_label}_{sc_key}'
            y_label = f'{y_label}_{sc_key}'

        if y_label not in df:
            y_coord = f'r_y_{coords}'
            z_coord = f'r_z_{coords}'
            if sc_key is not None:
                y_coord = f'r_y_{coords}_{sc_key}'
                z_coord = f'r_z_{coords}_{sc_key}'

            check_labels(df, y_coord, z_coord)
            df[y_label] = np.sqrt(df[y_coord]**2 + df[z_coord]**2)

        check_labels(df, x_label, y_label)

        if signed_rho=='y':
            df[y_label] *= np.sign(df[y_coord])
        elif signed_rho=='z':
            df[y_label] *= np.sign(df[z_coord])
    else:
        components = ('x','y','z')
        x_comp = plane[0]
        y_comp = plane[1]
        if x_comp not in components or y_comp not in components or x_comp == y_comp:
            raise ValueError('Plane "{plane}" not a valid choice.')

        x_label = f'r_{x_comp}_{coords}'
        y_label = f'r_{y_comp}_{coords}'

        if sc_key is not None:
            x_label = f'{x_label}_{sc_key}'
            y_label = f'{y_label}_{sc_key}'

        check_labels(df, x_label, y_label)

    unit = df.attrs.get('units', {}).get(x_label, None)

    ###-------------------PLOTTING POSITION-------------------###
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    if display == 'heat':
        n_bins = (calculate_bins(df[x_label],bin_width), calculate_bins(df[y_label],bin_width))
        h = ax.hist2d(df[x_label], df[y_label],
                      bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')

        cbar = fig.colorbar(h[3], ax=ax)
        cbar.ax.tick_params(colors=black)
        cbar.set_label('Number of Points', color=black)
        cbar.outline.set_edgecolor(black)
        ax.set_facecolor('k')

    elif display == 'scatter':
        ax.scatter(df[x_label], df[y_label], c='b', s=0.3)

    elif display == 'scatter_regions':
        cmap = plt.get_cmap('tab20c')
        if regions is None:
            regions = df['GRMB_region'].unique()
        num_regions = len(regions)
        colours = cmap(np.linspace(0, 1, num_regions))

        for region, colour in zip(regions, colours):
            region_data = df[df['GRMB_region'] == region]
            percent = len(region_data)/len(df)*100
            if region_labels is not None:
                region_label = region_labels.get(region,region)
            plt.scatter(region_data[x_label],region_data[y_label],
                c=[colour],label=f'{region_label}: {percent:.2g}%',s=0.3)

    else:
        raise ValueError(f'Invalid display option: {display}')

    ###-------------------ADD MSH MODELS-------------------###
    if 'BS' in models or 'Both' in models:

        # Bow shock
        phi = 0 if y_comp=='z' else np.pi/2
        bs_kwargs = {}
        bs_kwargs['phi'] = phi
        if 'Median' in models:
            if df_omni is None:
                print('Need OMNI data for Median solar wind conditions; used typical.')
            else:
                df_plasma        = df_omni.loc[df_omni.index.isin(df.index)]
                bs_kwargs['Pd']     = np.nanmedian(df_plasma['p_flow'])
                bs_kwargs['v_sw_x'] = np.nanmedian(df_plasma['v_x_GSE'])
                bs_kwargs['v_sw_y'] = np.nanmedian(df_plasma['v_y_GSE'])
                bs_kwargs['v_sw_z'] = np.nanmedian(df_plasma['v_z_GSE'])

        bs_jel = msh_boundaries('jelinek', 'BS', **bs_kwargs)
        bs_x_coords = bs_jel.get(x_comp)
        bs_y_coords = bs_jel.get(y_comp)

        if plane=='x-rho':
            mask = bs_jel.get('y')<=0
            bs_x_coords = bs_x_coords[mask]
            bs_y_coords = bs_y_coords[mask]

        bs_nose = list(bs_jel.get('nose'))
        bs_nose.append(np.sqrt(bs_nose[1]**2+bs_nose[2]**2))

        bs_nose_dict = dict(zip(('x','y','z','rho'), bs_nose))
        bs_stand_off_x = bs_nose_dict.get(x_comp)
        bs_stand_off_y = bs_nose_dict.get(y_comp)

        # Bow shock
        ax.plot(bs_x_coords, bs_y_coords, color='lime', lw=3, label='Bow shock')
        ax.scatter(bs_stand_off_x, bs_stand_off_y, c='lime')
        if nose_text=='BS':
            ax.plot([0,bs_stand_off_x],[0,bs_stand_off_y],c='w',ls=':',lw=2,zorder=1)
            ax.text(bs_stand_off_x-0.5, bs_stand_off_y+0.5, f'$R_0$ = {bs_jel.get("R0"):.1f} $\\mathrm{{R_E}}$, {np.degrees(bs_jel.get("alpha_tot")):.1f}$^\\circ$', fontsize=10, color='lime')

    if 'MP' in models or 'Both' in models:

        # Magnetopause
        phi = 0 if y_comp=='z' else np.pi/2
        mp_kwargs = {}
        mp_kwargs['phi'] = phi
        if 'Median' in models:
            if df_omni is None:
                print('Need OMNI data for Median solar wind conditions; used typical.')
            else:
                df_plasma        = df_omni.loc[df_omni.index.isin(df.index)]
                mp_kwargs['Pd']     = np.nanmedian(df_plasma['p_flow'])
                mp_kwargs['v_sw_x'] = np.nanmedian(df_plasma['v_x_GSE'])
                mp_kwargs['v_sw_y'] = np.nanmedian(df_plasma['v_y_GSE'])
                mp_kwargs['v_sw_z'] = np.nanmedian(df_plasma['v_z_GSE'])

        mp_jel = msh_boundaries('shue', 'MP', **mp_kwargs)
        mp_x_coords = mp_jel.get(x_comp)
        mp_y_coords = mp_jel.get(y_comp)

        if plane=='x-rho':
            mask = mp_jel.get('y')<=0
            mp_x_coords = mp_x_coords[mask]
            mp_y_coords = mp_y_coords[mask]

        mp_nose = list(mp_jel.get('nose'))
        mp_nose.append(np.sqrt(mp_nose[1]**2+mp_nose[2]**2))

        mp_nose_dict = dict(zip(('x','y','z','rho'), mp_nose))
        mp_stand_off_x = mp_nose_dict.get(x_comp)
        mp_stand_off_y = mp_nose_dict.get(y_comp)

        if 'Both' not in models:
            ax.plot([0,mp_stand_off_x],[0,mp_stand_off_y],c='w',ls=':',lw=2,zorder=1)
            if nose_text=='MP':
                ax.text(mp_stand_off_x-0.5, mp_stand_off_y+0.5, f'$R_0$ = {mp_jel.get("R0"):.1f} $\\mathrm{{R_E}}$, {np.degrees(mp_jel.get("alpha_tot")):.1f}$^\\circ$', fontsize=10, color='m')


        # Magnetopause
        ax.plot(mp_x_coords, mp_y_coords, color='m', lw=3, label='Magnetopause')
        ax.scatter(mp_stand_off_x, mp_stand_off_y, c='m')


    ###-------------------ADD TITLES AND AXIS LABELS-------------------###

    if equal_axes:
        ax.set_aspect('equal', adjustable='box')

    # Plot Earth at the origin
    if plane=='yz':
        create_circle(ax, centre=(0,0), radius=1, colour='black')
    elif centre_Earth=='quarter':
        create_quarter_circle_marker(ax, centre=(0, 0), radius=1)
    elif centre_Earth=='half':
        create_half_circle_marker(ax, centre=(0, 0), radius=1, full=False)
    elif centre_Earth=='full':
        create_half_circle_marker(ax, centre=(0, 0), radius=1, full=True)


    if plane in ('xy',):
        ax.invert_xaxis()
        ax.invert_yaxis()
    elif plane in ('xz','yz'):
        ax.invert_xaxis()
    elif plane in ('x-rho'):
        if centre_Earth != 'none':
            ax.set_xlim(0)
            ax.set_ylim(0)
        ax.invert_xaxis()
    elif plane in ('yx',):
        ax.invert_yaxis()

    if not show_key:
        x_label = x_label.replace(f'_{sc_key}', '')
        y_label = y_label.replace(f'_{sc_key}', '')

    x_axis_label = create_label(x_label, data_name=x_name, unit=unit)
    y_axis_label = create_label(y_label, data_name=y_name, unit=unit)
    ax.set_xlabel(x_axis_label, c=black)
    ax.set_ylabel(y_axis_label, c=black)

    ###-------------------ADJUST LAYOUT AND DISPLAY PLOT-------------------##
    if want_legend:
        add_legend(fig, ax, heat=is_heat)
    dark_mode_fig(fig, black, white, is_heat)
    add_figure_title(fig, black, brief_title, x_axis_label, y_axis_label, ax)

    if return_objs:
        return fig, ax, cbar

    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()

def plot_apogee_perigee(df, **kwargs):

    axis_labels = kwargs.get('axis_labels',True)
    colour      = kwargs.get('colour','r')
    show_text   = kwargs.get('show_text',False)
    window_size = kwargs.get('window_size',30) # days

    fig         = kwargs.get('fig',None)
    ax          = kwargs.get('ax',None)
    return_objs = kwargs.get('return_objs',False)

    r_mag = np.sqrt(df['r_x_GSE']**2+df['r_y_GSE']**2+df['r_z_GSE']**2)

    window_size *= 24 * 60

    # Compute the rolling maximum
    apogees = r_mag.rolling(window=window_size, center=True).max()
    perigees = r_mag.rolling(window=window_size, center=True).min()

    apo_max = np.max(apogees)
    apo_min = np.min(apogees)

    apo_max_time = apogees.idxmax()
    apo_min_time = apogees.idxmin()

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    r_mag.plot(color='grey', lw=0.5)
    apogees.plot(color=colour, lw=2.25)
    perigees.plot(color=colour, lw=2.25)

    # Max and min apogees
    ax.scatter(apo_max_time, apo_max, c=colour, s=60, zorder=5)
    ax.scatter(apo_min_time, apo_min, c=colour, s=60, zorder=5)

    ax.text(apo_max_time, apo_max+0.2, f'{apo_max:.2f}', color='r', ha='left', va='bottom')
    ax.text(apo_min_time, apo_min+0.2, f'{apo_min:.2f}', color='r', ha='left', va='bottom')

    if show_text:
        # Times
        ax.axvline(apo_max_time, 0, 0.01, color=colour, lw=0.5)
        ax.axvline(apo_min_time, 0, 0.01, color=colour, lw=0.5)

        ax.text(apo_max_time, -0.01, f'{apo_max_time.year}', color='r', ha='center', va='top', transform=ax.get_xaxis_transform())
        ax.text(apo_min_time, -0.01, f'{apo_min_time.year}', color='r', ha='center', va='top', transform=ax.get_xaxis_transform())

    if axis_labels:
        ax.set_xlabel('Time', c=black)
        ax.set_ylabel(r'Radial Distance [$\mathrm{R_E}$]', c=black)
        add_figure_title(fig, 'Cluster\'s orbit showing Apogee and Perigee')

    dark_mode_fig(fig,black,white)

    if return_objs:
        return fig, ax

    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()

# %% Value_over_time

def plot_scalar_field(df, data_col, **kwargs):

    delta  = kwargs.get('delta',Timedelta(minutes=1))
    title  = kwargs.get('title',None)

    df = df.copy()
    ###-------------------VALIDATE INPUTS AND CONFIGURE PLOT KEYS-------------------###
    y_label = data_col
    y_str = data_string(data_col)
    check_labels(df, y_label)

    start_date_str = df.index[0].strftime('%Y-%m-%d')
    end_date_str = df.index[-1].strftime('%Y-%m-%d')

    ###-------------------INITIALISE FIGURE-------------------###
    fig, ax = plt.subplots()

    ###-------------------SEGMENT DATA AND PLOT SCALAR FIELD-------------------###
    plot_segments(ax, df, y_label, blue, f'${y_str}$', delta=delta)

    ###-------------------ADD TITLES AND AXIS LABELS-------------------###
    plot_title_info = f' {len(df):,} minutes of data are in the figure.'
    fig.suptitle(f'{title} from {start_date_str} to {end_date_str}.{plot_title_info}', c=black, wrap=True)
    ax.set_xlabel('Time', c=black)

    """
    # Format the x-axis to display detailed time and date
    if df.index[-1] - df.index[0] <= pd.Timedelta(weeks=1):
        date_format = mdates.DateFormatter('%H:%M:%S\n%Y-%m-%d')  # Show time and date
    else:
        date_format = mdates.DateFormatter('%Y-%m-%d')  # Show only the date
    ax.xaxis.set_major_formatter(date_format)
    """
    # Add y-axis label with units
    unit = df.attrs.get('units', {}).get(y_label, '')
    if unit != '':
        unit = f' [{unit}]'
    ax.set_ylabel(f'${data_string(y_str)}${unit}', c=black)

    ###-------------------ADJUST LAYOUT AND DISPLAY PLOT-------------------###
    dark_mode_fig(fig,black,white)
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()


def plot_vector_field(df, field, source=None, keys=None, delta=Timedelta(minutes=1), coords='GSE'):

    ###-------------------VALIDATE INPUTS AND CONFIGURE PLOT KEYS-------------------###
    if source is not None and keys is not None:
        y_key = keys.get(source)
        if not y_key:
            raise KeyError(f'Source "{source}" not found in keys dictionary.')

        y_title =f'{source} {field}'
        y_mag_label = f'{field}_mag_{y_key}'
        y_x_label = f'{field}_x_{coords}_{y_key}'
        y_y_label = f'{field}_y_{coords}_{y_key}'
        y_z_label = f'{field}_z_{coords}_{y_key}'
    else:
        y_title =f'{field}'
        y_mag_label = f'{field}_mag'
        y_x_label = f'{field}_x_{coords}'
        y_y_label = f'{field}_y_{coords}'
        y_z_label = f'{field}_z_{coords}'

    check_labels(df, y_x_label, y_y_label, y_z_label)

    start_date_str = df.index[0].strftime('%Y-%m-%d')
    end_date_str = df.index[-1].strftime('%Y-%m-%d')

    ###-------------------INITIALISE FIGURE AND SUBPLOTS-------------------###
    fig, axs = plt.subplots(
        4,  # Number of plots
        figsize=(12, 12),
        sharex=True,  # Share the same x-axis range
        sharey=False,  # Do not share y-axis ranges
        gridspec_kw={'hspace': 0} # Adjust vertical spacing
    )

    ###-------------------SEGMENT DATA AND PLOT VECTOR COMPONENTS-------------------###
    segment_dataframe(df, delta=delta)
    for _, segment in df.groupby('segment'):
        axs[0].plot(segment[y_x_label], c='b', lw=0.5)
        axs[1].plot(segment[y_y_label], c='b', lw=0.5)
        axs[2].plot(segment[y_z_label], c='b', lw=0.5)
        axs[3].plot(segment[y_mag_label], c='r', lw=0.5)
    df.drop(columns=['segment'], inplace=True)

    ###-------------------ADD TITLES AND AXIS LABELS-------------------###
    fig.suptitle(
        f'{y_title} data from {start_date_str} to {end_date_str}',
        fontsize=18
    )
    plt.xlabel('Time', fontsize=16)

    # Format the x-axis to display detailed time and date
    date_format = mdates.DateFormatter('%H:%M:%S\n%Y-%m-%d')
    axs[2].xaxis.set_major_formatter(date_format)

    # Add y-axis labels with units
    unit = df.attrs.get('units', {}).get(y_mag_label, 'unknown')
    axs[0].set_ylabel(f'${field}_x$ [{unit}]')
    axs[1].set_ylabel(f'${field}_y$ [{unit}]')
    axs[2].set_ylabel(f'${field}_z$ [{unit}]')
    axs[3].set_ylabel(f'|{field}| [{unit}]')

    ###-------------------ADJUST LAYOUT AND DISPLAY PLOT-------------------###
    plt.tight_layout()
    save_figure(fig)
    plt.show()
    plt.close()
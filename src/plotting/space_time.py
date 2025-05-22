# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:42:37 2025

@author: richarj2
"""

def plot_scalar_field(df, data_col, **kwargs):
    """
    Plots a scalar field (e.g., a specific data column) over time.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to be plotted. The DataFrame must have a DateTimeIndex
        and include the specified column for the scalar field.

    data_col : str
        Label for the scalar field to be plotted, such as 'pressure' or 'temperature'. This
        should be the name of the column to plot, prefixed by the appropriate source key.

    source : str
        Key identifying the data source to be used in plotting. This is used to access the
        appropriate field in the DataFrame based on the `keys` dictionary.

    keys : dict
        Dictionary mapping sources to keys. The `keys[source]` provides the specific key used
        to fetch the corresponding data from the DataFrame.

    delta : Timedelta, optional
        Time gap threshold for segmenting data into discrete blocks. This allows visualisation
        of different segments in case of large time gaps. Defaults to 1 minute.

    Returns
    -------
    None
        This procedure directly generates and displays a plot for the scalar field data over time.
    """
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
    """
    Plots the vector field data over time, visualising the components of a vector field (e.g., B-field)
    and its magnitude.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to be plotted. The DataFrame must have a DateTimeIndex
        and contain columns for the vector field components.

    field : str
        Label for the vector field to be plotted (e.g., 'B' for the B-field).

    source : str
        Key identifying the data source to be used in plotting. This is used to access the appropriate
        column from the DataFrame.

    keys : dict
        Dictionary mapping sources to keys. The `keys[source]` provides the specific key used to fetch
        the corresponding data from the DataFrame.

    coords : str, optional
        Coordinate system in which to plot the data. Defaults to 'GSE' (Geocentric Solar Ecliptic).

    delta : pandas.Timedelta, optional
        Time gap threshold for segmenting data into discrete blocks. Defaults to 1 minute.

    Returns
    -------
    None
        This procedure directly generates and displays a plot with multiple subplots for the vector field components.
    """
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


def plot_orbit(df, plane='yz', coords='GSE', **kwargs):
    """
    Plots the vector field data over time, visualising the components of a vector field (e.g., B-field)
    and its magnitude.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to be plotted. The DataFrame must have a DateTimeIndex
        and contain columns for the vector field components.

    source : str
        Key identifying the data source to be used in plotting. This is used to access the appropriate
        column from the DataFrame.

    keys : dict
        Dictionary mapping sources to keys. The `keys[source]` provides the specific key used to fetch
        the corresponding data from the DataFrame.

    coords : str, optional
        Coordinate system in which to plot the data. Defaults to 'GSE' (Geocentric Solar Ecliptic).

    delta : pandas.Timedelta, optional
        Time gap threshold for segmenting data into discrete blocks. Defaults to 1 minute.

    Returns
    -------
    None
        This procedure directly generates and displays a plot with multiple subplots for the vector field components.
    """
    df = df.copy()
    display       = kwargs.get('display','Scatter')
    bin_width     = kwargs.get('bin_width',None)
    x_name        = kwargs.get('x_name',None)
    y_name        = kwargs.get('y_name',None)

    sc_key        = kwargs.get('sc_key', None)
    models        = kwargs.get('models', None)

    regions       = kwargs.get('region_nums',None)
    region_labels = kwargs.get('region_labels',None)

    sign_y        = kwargs.get('sign_y',False)
    sign_z        = kwargs.get('sign_z',False)
    pos_x         = kwargs.get('pos_x',False)
    abs_x         = kwargs.get('abs_x',False)
    centre_Earth  = kwargs.get('centre_Earth',True)
    brief_title   = kwargs.get('brief_title',None)

    is_heat = False
    if display == 'Heat':
        is_heat = True

    if brief_title is None:
        start_year = df.index[0].strftime('%Y')
        end_year = df.index[-1].strftime('%Y')
        brief_title = f'Orbit from {start_year} to {end_year} in {plane} plane.'

    ###-------------------VALIDATE INPUTS AND CONFIGURE PLOT KEYS-------------------###
    if plane == 'x-rho':
        x_comp = 'x'
        y_comp = 'rho'
        x_label = f'r_x_{coords}'
        y_label = 'r_rho'
        y_coord = f'r_y_{coords}'
        z_coord = f'r_z_{coords}'
        if sc_key is not None:
            x_label = f'{x_label}_{sc_key}'
            y_label = f'{y_label}_{sc_key}'
            y_coord = f'r_y_{coords}_{sc_key}'
            z_coord = f'r_z_{coords}_{sc_key}'

        check_labels(df, x_label, y_coord, z_coord)

        df[y_label] = np.sqrt(df[y_coord]**2 + df[z_coord]**2)
        if sign_y:
            df[y_label] *= np.sign(df[y_coord])
        if sign_z:
            df[y_label] *= np.sign(df[z_coord])
    else:
        components = ('x','y','z')
        x_comp = plane[0]
        y_comp = plane[1]
        if x_comp not in components or y_comp not in components or x_comp == y_comp:
            raise ValueError('Plane "{plane}" not a valid choice.')

        x_label = f'r_{x_comp}_{coords}'
        y_label = f'r_{y_comp}_{coords}'

        # Find the unused component (z_comp)
        z_comp = next(comp for comp in components if comp not in (x_comp, y_comp))
        z_label = f'r_{z_comp}_{coords}'

        if sc_key is not None:
            x_label = f'{x_label}_{sc_key}'
            y_label = f'{y_label}_{sc_key}'
            z_label = f'{z_label}_{sc_key}'

        check_labels(df, x_label, y_label, z_label)

    if pos_x:
        df = df[df[x_label]>0]
    if abs_x:
        df[x_label] = np.abs(df[x_label])

    unit = df.attrs.get('units', {}).get(x_label, None)

    ###-------------------PLOTTING POSITION-------------------###
    fig, ax = plt.subplots()

    if display == 'Heat':
        n_bins = (calculate_bins(df[x_label],bin_width), calculate_bins(df[y_label],bin_width))
        h = ax.hist2d(df[x_label], df[y_label],
                      bins=n_bins, norm=mpl.colors.LogNorm(), cmap='hot')

        cbar = fig.colorbar(h[3], ax=ax)
        cbar.ax.tick_params(colors=black)
        cbar.set_label('Number of Points', color=black)
        cbar.outline.set_edgecolor(black)
        ax.set_facecolor('k')

    elif display == 'Scatter':
        ax.scatter(df[x_label], df[y_label], c='b', s=0.3)

    elif display == 'Scatter_gradient':
        t = Ticktock(df.index.to_pydatetime(), 'UTC').CDF
        norm = plt.Normalize(t.min(), t.max())
        cmap = plt.get_cmap('plasma')
        plt.scatter(df[x_label], df[y_label],
                    c=t, cmap=cmap, norm=norm, s=0.3, label='Time gradient')

    elif display == 'Scatter_regions':
        cmap = plt.get_cmap('tab20c')  # You can use 'tab20' or other colour maps for more distinct colours
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
    if models == 'Bow_shock':
        # Calculate bow shock boundaries using pressure data from OMNI
        pressures = df['p_flow_OMNI']
        velocities = df['v_x_GSE_OMNI']
        bs_jel = bs_boundaries('jelinek', Pd=np.median(pressures, vsw=np.median(velocities)))

        bs_x_coords = bs_jel.get(x_comp)
        bs_y_coords = bs_jel.get(y_comp)

        ax.plot(bs_x_coords, bs_y_coords, label='Median BS', linestyle='--', color='b')

    elif models == 'Typical_Both':

        # Bow shock
        bs_jel = bs_boundaries('jelinek')
        bs_x_coords = bs_jel.get(x_comp)
        bs_y_coords = bs_jel.get(y_comp)
        y_neg = bs_jel.get('y') < 0 # Stand-off is in -ve quadrant

        bs_R0 = bs_jel.get('R0')
        alpha = bs_jel.get('alpha')

        bs_stand_off_x = bs_R0*np.cos(alpha)
        bs_stand_off_y = bs_R0*np.sin(alpha)

        # Magnetopause
        mp_shu = mp_boundaries('shue')
        mp_x_coords = mp_shu.get(x_comp)
        mp_y_coords = mp_shu.get(y_comp)
        y_neg = mp_shu.get('y') < 0 # Stand-off is in -ve quadrant

        mp_R0 = mp_shu.get('R0')

        mp_stand_off_x = mp_R0*np.cos(alpha)
        mp_stand_off_y = mp_R0*np.sin(alpha)

        ax.plot([0,bs_stand_off_x],[0,-bs_stand_off_y],c='w',ls=':',zorder=1)

        # Bow shock
        ax.plot(bs_x_coords[y_neg], bs_y_coords[y_neg], color='lime', lw=3, label='Bow shock')
        ax.scatter(bs_stand_off_x, -bs_stand_off_y, c='lime')
        #ax.text(bs_stand_off_x + 0.5, -mp_stand_off_y + 0.5, f'$R_0$ = {bs_R0:.1f} $R_E$, {np.degrees(alpha):.1f}$^\\circ$',
        #        fontsize=10, color='lime', backgroundcolor='k', ha='right', va='center')

        # Magnetopause
        ax.plot(mp_x_coords[y_neg], mp_y_coords[y_neg], color='magenta', lw=3, ls='--', label='Magnetopause')
        ax.scatter(mp_stand_off_x, -mp_stand_off_y, c='magenta')
        #ax.text(mp_stand_off_x - 0.5, -mp_stand_off_y + 0.5, f'$R_0$ = {mp_R0:.1f} $R_E$',
        #        fontsize=10, color='magenta', backgroundcolor='k', ha='left', va='center')

    elif models == 'Typical':

        # Bow shock
        bs_jel = bs_boundaries('jelinek')
        bs_x_coords = bs_jel.get(x_comp)
        bs_y_coords = bs_jel.get(y_comp)
        y_neg = bs_jel.get('y') < 0 # Stand-off is in -ve quadrant

        bs_R0 = bs_jel.get('R0')
        alpha = bs_jel.get('alpha')

        bs_stand_off_x = bs_R0*np.cos(alpha)
        bs_stand_off_y = bs_R0*np.sin(alpha)

        ax.plot([0,bs_stand_off_x],[0,-bs_stand_off_y],c='w',ls=':',lw=2,zorder=1)

        # Bow shock
        ax.plot(bs_x_coords[y_neg], bs_y_coords[y_neg], color='lime', lw=3, label='Bow shock')
        ax.scatter(bs_stand_off_x, -bs_stand_off_y, c='lime')


    elif models == 'Simple_BS':
        # Calculate boundaries for each model using typical values
        bs_jel = bs_boundaries('jelinek')
        bs_x_coords = bs_jel.get(x_comp)
        bs_y_coords = bs_jel.get(y_comp)
        y_pos = bs_jel.get('y')<0
        # Stand-off is in -ve quadrant

        ax.plot(bs_x_coords[y_pos], bs_y_coords[y_pos], color='lime', lw=3)

        bd_R0 = bs_jel.get('R0')
        alpha = bs_jel.get('alpha')

        stand_off_x = bd_R0*np.cos(alpha)
        stand_off_y = bd_R0*np.sin(alpha)
        ax.scatter(stand_off_x, -stand_off_y, c='lime')
        ax.text(stand_off_x - 0.5, -stand_off_y + 0.5, f'$R_0$ = {bd_R0:.1f} $R_E$, {np.degrees(alpha):.1f}$^\\circ$', fontsize=10, color='lime')
        ax.plot([0,stand_off_x],[0,-stand_off_y],ls=':',c='w')

    # Plot Earth at the origin
    create_half_circle_marker(ax, center=(0, 0), radius=1, full=False)

    ###-------------------ADD TITLES AND AXIS LABELS-------------------###
    ax.set_aspect('equal', adjustable='box')

    if plane in ('xy',):
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
    elif plane in ('xz','yz'):
        plt.gca().invert_xaxis()
    elif plane in ('x-rho'):
        if centre_Earth:
            plt.xlim(0)
            plt.ylim(0)
        plt.gca().invert_xaxis()
    elif plane in ('yx',):
        plt.gca().invert_yaxis()

    x_axis_label = create_label(x_label, data_name=x_name, unit=unit)
    y_axis_label = create_label(y_label, data_name=y_name, unit=unit)
    ax.set_xlabel(x_axis_label, c=black)
    ax.set_ylabel(y_axis_label, c=black)

    ###-------------------ADJUST LAYOUT AND DISPLAY PLOT-------------------##
    add_legend(fig, ax, heat=is_heat)
    dark_mode_fig(fig,black,white,is_heat)
    add_figure_title(fig, brief_title, x_axis_label, y_axis_label)
    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()


def plot_apogee_perigee(df):

    r_mag = np.sqrt(df['r_x_GSE']**2+df['r_y_GSE']**2+df['r_z_GSE']**2)

    window_size = 30 * 24 * 60
    window_size = 7 * 24 * 60

    # Compute the rolling maximum
    apogees = r_mag.rolling(window=window_size, center=True).max()
    perigees = r_mag.rolling(window=window_size, center=True).min()

    apo_max = np.max(apogees)
    apo_min = np.min(apogees)

    apo_max_time = apogees.idxmax()
    apo_min_time = apogees.idxmin()

    fig, ax = plt.subplots()

    r_mag.plot(color='grey', lw=0.5)
    apogees.plot(color='r', lw=2.25)
    perigees.plot(color='r', lw=2.25)

    # Max and min apogees
    ax.scatter(apo_max_time, apo_max, c='r', s=60, zorder=5)
    ax.scatter(apo_min_time, apo_min, c='r', s=60, zorder=5)

    ax.text(apo_max_time, apo_max+0.2, f'{apo_max:.2f}', color='r', ha='left', va='bottom')
    ax.text(apo_min_time, apo_min+0.2, f'{apo_min:.2f}', color='r', ha='left', va='bottom')

    # Times
    ax.axvline(apo_max_time, 0, 0.01, color='r', lw=0.5)
    ax.axvline(apo_min_time, 0, 0.01, color='r', lw=0.5)

    ax.text(apo_max_time, -0.01, f'{apo_max_time.year}', color='r', ha='center', va='top', transform=ax.get_xaxis_transform())
    ax.text(apo_min_time, -0.01, f'{apo_min_time.year}', color='r', ha='center', va='top', transform=ax.get_xaxis_transform())

    ax.set_xlabel('Time', c=black)
    ax.set_ylabel(r'Radial Distance [$R_E$]', c=black)

    add_figure_title(fig, 'Cluster\'s orbit showing Apogee and Perigee')
    dark_mode_fig(fig,black,white)
    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()
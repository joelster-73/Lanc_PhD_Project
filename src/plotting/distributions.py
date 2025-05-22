# -*- coding: utf-8 -*-
"""
Created on Fri May 16 10:38:41 2025

@author: richarj2
"""

def plot_freq_hist(df, *columns, **kwargs):
    """
    Plots a frequency histogram for specified DataFrame columns. Optionally calculates and plots
    the histogram of the difference between two specified columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data to be plotted. The DataFrame should have a DateTimeIndex
        and include the specified columns.

    *columns : str
        One or more column names to be plotted as histograms. For `difference=True`, exactly two
        column names must be provided.

    difference : bool, optional
        If True, plots the histogram of the difference between the first two specified columns.
        Defaults to False.

    Returns
    -------
    None
        This procedure directly generates and displays the plot.
    """

    data_names = kwargs.get('data_names',None)
    perc       = kwargs.get('perc',0)
    bin_width  = kwargs.get('bin_width',None)
    fit_gaus   = kwargs.get('fit_gaus',True)
    want_legend   = kwargs.get('want_legend',False)
    brief_title   = kwargs.get('brief_title',None)


    df = df.copy()

    ###-------------------VALIDATE INPUTS-------------------###
    if isinstance(df, pd.DataFrame):
        if not columns:
            raise ValueError('No columns specified for plotting.')
        for c in columns:
            if c not in df:
                raise KeyError(f'Column "{c}" not found in the DataFrame.')

    ###-------------------INITIALISE FIGURE-------------------###
    cmap = plt.get_cmap('Set1')
    fig, ax = plt.subplots()

    ###-------------------PLOT HISTOGRAM-------------------###
    if isinstance(df, pd.Series) or len(columns)==1:
        if isinstance(df, pd.DataFrame):
            col = columns[0]
            x_unit = df.attrs['units'].get(col, None)
            series = df[col]
        else:
            x_unit = df.attrs.get('units', None)
            col = df.name

        series = series.to_numpy()
        series = series[~np.isnan(series)]

        if x_unit == 'rad':
            series = np.degrees(series)
            x_unit = 'deg'

        n_bins = calculate_bins(series,bin_width)
        counts, bins, _ = plt.hist(
            series, bins=n_bins, alpha=1.0,
            color='b', edgecolor='grey'
        )
        if fit_gaus:
            plot_gaussian(ax,counts,bins,c='w',ls='-',name='Frequency Histogram')

        perc_range = (np.percentile(series, perc), np.percentile(series, 100-perc))

        data_name = data_names[0] if data_names is not None else None
        data_str = data_string(col)
        x_label = create_label(data_str, unit=x_unit, data_name=data_name)

    else:
        # Plot histograms for individual columns
        alpha = 1.0 if len(columns) == 1 else 0.4
        colours = [cmap(i % cmap.N) for i in range(len(columns))]

        # Adjust bin count based on data range

        if bin_width is None:
            for c in columns:
                data_range = int(max(df[c]) - min(df[c]))
                if data_range > n_bins:
                    n_bins = data_range

        # Plot each column
        for i, column in enumerate(columns):
            if bin_width is not None:
                n_bins = calculate_bins(df[c],bin_width)
            the_label = column if data_names is None else data_names[i]
            if x_unit == 'rad':
                df[column] = np.degrees(df[column])
            df[column].plot(
                kind='hist', bins=n_bins, alpha=alpha,
                label=f'${data_string(the_label)}$', color=colours[i],
                edgecolor='grey'
            )
        x_unit = df.attrs['units'].get(columns[0], None)
        if x_unit == 'rad':
            x_unit = 'deg'
        x_label = f'Data [{x_unit}]'

    ###-------------------SET LABELS AND TITLE-------------------###
    ax.set_xlabel(x_label, c=black)
    ax.set_ylabel('Frequency', c=black)
    if brief_title is None:
        brief_title = f'Frequency Histogram with {n_bins} bins.'
    ax.set_xlim(perc_range)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:,.0f}'))

    add_legend(fig, ax, legend_on=want_legend)
    add_figure_title(fig, brief_title)
    dark_mode_fig(fig,black,white)
    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()


def plot_gaussian(ax,counts,bins,c='w',ls='-',name='Frequency Histogram',detailed=False,return_peak=False):

    mids = 0.5 * (bins[1:] + bins[:-1])

    non_zero_mask = counts > 0
    counts = counts[non_zero_mask]
    mids = mids[non_zero_mask]

    A, mu, std = gaussian_fit(mids, counts, name=name, detailed=detailed)
    if detailed:
        try:
            label=f'$\\mu$: ${mu:.1uL}$\n$\\sigma$: ${std:.1uL}$'
        except:
            label=f'$\\mu$: {mu:.3g}\n$\\sigma$: {std:.3g}'
    else:
        label=f'$\\mu$: {mu:.3g}\n$\\sigma$: {std:.3g}'

    # Plot the fitted normal distribution
    bin_width = bins[1] - bins[0]
    xmin = mids[0] - 0.5 * bin_width
    xmax = mids[-1] + 0.5 * bin_width
    x = np.linspace(xmin, xmax, 500)
    p = gaussian(x, A.n, mu.n, std.n)
    ax.plot(
        x, p, c=c, ls=ls, linewidth=2
    )
    ax.plot([],[],' ',label=label)
    if return_peak:
        return mu, gaussian(mu.n, A.n, mu.n, std.n)

def plot_bimodal(ax,counts,bins,c='w',ls='-',name='Frequency Histogram',detailed=False,return_peak=False):

    mids = 0.5 * (bins[1:] + bins[:-1])
    non_zero_mask = counts > 0
    counts = counts[non_zero_mask]
    mids = mids[non_zero_mask]

    A1, mu1, std1, A2, mu2, std2 = bimodal_fit(mids, counts, name=name, detailed=True)
    if detailed:
        try:
            label1=f'$\\mu_1$: ${mu1:.1uL}$\n$\\sigma_1$: ${std1:.1uL}$'
            label2=f'$\\mu_2$: ${mu2:.1uL}$\n$\\sigma_2$: ${std2:.1uL}$'
        except:
            label1=f'$\\mu_1$: {mu1:.3g}\n$\\sigma_1$: {std1:.3g}'
            label2=f'$\\mu_2$: {mu2:.3g}\n$\\sigma_2$: {std2:.3g}'
    else:
        label1=f'$\\mu_1$: {mu1:.3g}\n$\\sigma_1$: {std1:.3g}'
        label2=f'$\\mu_2$: {mu2:.3g}\n$\\sigma_2$: {std2:.3g}'

    # Plot the fitted normal distribution
    bin_width = bins[1] - bins[0]
    xmin = mids[0] - 0.5 * bin_width
    xmax = mids[-1] + 0.5 * bin_width
    x = np.linspace(xmin, xmax, 500)
    p = bimodal(x, A1.n, mu1.n, std1.n, A2.n, mu2.n, std2.n)
    ax.plot(
        x, p, c=c, ls=ls, linewidth=2
    )
    ax.plot([],[],' ',label=label1)
    ax.plot([],[],' ',label=label2)
    if return_peak:
        h1 = bimodal(mu1.n, A1.n, mu1.n, std1.n, A2.n, mu2.n, std2.n)
        h2 = bimodal(mu2.n, A1.n, mu1.n, std1.n, A2.n, mu2.n, std2.n)
        return (mu1,mu2), (h1,h2)

def plot_bimodal_offset(ax,counts,bins,c='w',ls='-',name='Frequency Histogram',detailed=False,return_peak=False):

    mids = 0.5 * (bins[1:] + bins[:-1])
    non_zero_mask = counts > 0
    counts = counts[non_zero_mask]
    mids = mids[non_zero_mask]

    A1, mu1, std1, A2, mu2, std2, off = bimodal_fit_offset(mids, counts, name=name, detailed=detailed)
    if detailed:
        try:
            label1=f'$\\mu_1$: ${mu1:.1uL}$\n$\\sigma_1$: ${std1:.1uL}$'
            label2=f'$\\mu_2$: ${mu2:.1uL}$\n$\\sigma_2$: ${std2:.1uL}$'
        except:
            label1=f'$\\mu_1$: {mu1:.3g}\n$\\sigma_1$: {std1:.3g}'
            label2=f'$\\mu_2$: {mu2:.3g}\n$\\sigma_2$: {std2:.3g}'
    else:
        label1=f'$\\mu_1$: {mu1:.3g}\n$\\sigma_1$: {std1:.3g}'
        label2=f'$\\mu_2$: {mu2:.3g}\n$\\sigma_2$: {std2:.3g}'

    # Plot the fitted normal distribution
    bin_width = bins[1] - bins[0]
    xmin = mids[0] - 0.5 * bin_width
    xmax = mids[-1] + 0.5 * bin_width
    x = np.linspace(xmin, xmax, 500)
    p = bimodal_offset(x, A1.n, mu1.n, std1.n, A2.n, mu2.n, std2.n, off.n)
    ax.plot(
        x, p, c=c, ls=ls, linewidth=2
    )
    ax.plot([],[],' ',label=label1)
    ax.plot([],[],' ',label=label2)
    if return_peak:
        h1 = bimodal_offset(mu1.n, A1.n, mu1.n, std1.n, A2.n, mu2.n, std2.n, off.n)
        h2 = bimodal_offset(mu2.n, A1.n, mu1.n, std1.n, A2.n, mu2.n, std2.n, off.n)
        return (mu1,mu2), (h1,h2)

def plot_lognormal(ax, counts, bins, c='w', ls='-', name='Frequency Histogram', detailed=False, return_peak=False):
    mids = 0.5 * (bins[1:] + bins[:-1])
    non_zero_mask = counts > 0
    counts = counts[non_zero_mask]
    mids = mids[non_zero_mask]

    A, mu, sigma, mode = lognormal_fit(mids, counts, name=name, detailed=detailed)

    if detailed:
        try:
            label=f'$\\mu$: ${mu:.1uL}$\n$\\sigma$: ${sigma:.1uL}$'
        except:
            label=f'$\\mu$: {mu:.3g}\n$\\sigma$: {sigma:.3g}'
    else:
        label=f'$\\mu$: {mu:.3g}\n$\\sigma$: {sigma:.3g}'

    # Plot the fitted log-normal distribution
    bin_width = bins[1] - bins[0]
    xmin = mids[0] - 0.5 * bin_width
    xmax = mids[-1] + 0.5 * bin_width
    x = np.linspace(xmin, xmax, 500)
    p = lognormal(x, A.n, mu.n, sigma.n)
    ax.plot(
        x, p, c=c, ls=ls, linewidth=2
    )
    ax.plot([],[],' ',label=label)
    if return_peak:
        return mode, lognormal(mode.n, A.n, mu.n, sigma.n)


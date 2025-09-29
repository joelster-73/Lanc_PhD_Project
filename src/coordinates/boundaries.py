import numpy as np
from matplotlib import pyplot as plt

from .spatial import car_to_aGSE, car_to_aGSE_constant, aGSE_to_car_constant

def calc_msh_r_diff(df, surface, model=None, aberration='model', position_key=None, data_key=None, column_names=None, **kwargs):

    if model is None:
        model = 'jelinek' if surface=='BS' else 'shue'

    if aberration=='simple':
        simple_ab=True
    elif aberration=='model':
        simple_ab = True if model in ('jelinek','shue') else False
    elif aberration=='complete':
        simple_ab = False

    if column_names is None:
        column_names = {
            'r_x_name'  : 'r_x_GSE',
            'r_y_name'  : 'r_y_GSE',
            'r_z_name'  : 'r_z_GSE',
            'r_name'    : 'r',
            'r_ax_name' : 'r_x_aGSE',
            'r_ay_name' : 'r_y_aGSE',
            'r_az_name' : 'r_z_aGSE',
            'v_x_name'  : 'v_x_GSE',
            'v_y_name'  : 'v_y_GSE',
            'v_z_name'  : 'v_z_GSE',
            'p_name'    : 'p_flow'
        }
    else:
        column_names = column_names.copy()


    # Update position-related names
    if position_key is not None:
        for key in ['r_x_name', 'r_y_name', 'r_z_name', 'r_name', 'r_ax_name', 'r_ay_name', 'r_az_name']:
            column_names[key] += f'_{position_key}'

    # Update data-related names
    if data_key is not None:
        for key in ['v_x_name', 'v_y_name', 'v_z_name', 'p_name']:
            column_names[key] += f'_{data_key}'

    for key, val in column_names.items():
        if key in ('r_ax_name','r_ay_name','r_az_name'):
            continue
        if val not in df:
            if key=='r_name':
                r_x_name = column_names['r_x_name']
                r_y_name = column_names['r_y_name']
                r_z_name = column_names['r_z_name']
                df[val] = np.linalg.norm(df[[r_x_name,r_y_name,r_z_name]],axis=1)
                print(key,'not in df; has been inserted')
            else:
                print(key,'not in df')

    # New df
    df_ab = car_to_aGSE(df, column_names=column_names, simple=simple_ab)

    r_name    = column_names['r_name']
    r_ax_name = column_names['r_ax_name']
    r_ay_name = column_names['r_ay_name']
    r_az_name = column_names['r_az_name']
    v_x_name  = column_names['v_x_name']
    p_name    = column_names['p_name']


    valid_mask = ~df_ab[v_x_name].isna()
    if valid_mask.any():
        try:
            p = df.loc[valid_mask,p_name].to_numpy()
        except:
            p = 2.056

        theta_ps = np.arccos(
            df_ab.loc[valid_mask, r_ax_name] / df_ab.loc[valid_mask, r_name]
        )

        # Compute the radial distances based on the selected model
        if surface == 'BOTH':
            r_mp  = mp_shue1998(theta_ps, Pd=p, **kwargs)
            r_bs  = bs_jelinek2012(theta_ps, Pd=p, **kwargs)

            df_ab.loc[valid_mask, 'r_MP'] = r_mp
            df_ab.loc[valid_mask, 'r_BS'] = r_bs
            df_ab.loc[valid_mask, 'r_phi'] = theta_ps
            df_ab.loc[valid_mask, 'r_F'] = (df_ab.loc[valid_mask, r_name] - df_ab.loc[valid_mask, 'r_MP']) / (df_ab.loc[valid_mask, 'r_BS'] - df_ab.loc[valid_mask, 'r_MP'])

            df_ab.loc[~valid_mask, ['r_MP','r_BS','r_phi','r_F']] = np.nan

        else:

            if surface == 'BS':
                if model == 'jelinek':
                    r  = bs_jelinek2012(theta_ps, Pd=p, **kwargs)
                else:
                    raise ValueError(f'Model {model} not valid')

            elif surface == 'MP':
                if model == 'shue':
                    r  = mp_shue1998(theta_ps, Pd=p, **kwargs)
                elif model == 'jelinek':
                    r  = mp_jelinek2012(theta_ps, Pd=p, **kwargs)
                else:
                    raise ValueError(f'Model {model} not valid')

            else:
                raise ValueError(f'Surface {surface} not valid')

            # Set NaN for invalid rows
            df_ab.loc[~valid_mask, f'r_{surface}'] = np.nan
            df_ab.loc[~valid_mask, f'r_{surface}_diff'] = np.nan

            df_ab.loc[valid_mask, f'r_{surface}'] = r
            df_ab.loc[valid_mask, f'r_{surface}_diff'] = df_ab.loc[valid_mask, r_name] - df_ab.loc[valid_mask, f'r_{surface}']


    df_ab.loc[~valid_mask, r_ax_name] = np.nan
    df_ab.loc[~valid_mask, r_ay_name] = np.nan
    df_ab.loc[~valid_mask, r_az_name] = np.nan

    return df_ab

def msh_boundaries(model, surface='BS', aberration='model', **kwargs):
    """
    Computes the bow shock boundary (BSB) using a specified model.

    Parameters
    ----------
    model : str
        The model to be used for computing the bow shock boundary.
        Currently supports the following model:
        - 'jelinek' for the Jelinek et al. (2012) model.
    **kwargs :
        Additional parameters passed to the model function (`bs_jelinek2012`).
        These parameters depend on the specific model being used and are passed directly to the model function.

    Returns
    -------
    x : numpy.ndarray
        The x-coordinates of the bow shock boundary in GSE coordinates.
    yz : numpy.ndarray
        The yz-coordinates of the bow shock boundary in GSE coordinates.
    """
    if aberration=='simple':
        simple_ab=True
    elif aberration=='model':
        simple_ab = True if model in ('jelinek','shue') else False
    elif aberration=='complete':
        simple_ab = False

    # Azimuthal angle - default is x-y plane
    phi = kwargs.get('phi',np.pi/2)

    # Generate the range of theta values
    thetas = kwargs.get('thetas',np.linspace(-np.pi/2, np.pi/2, 500))

    x = np.cos(thetas)
    y = np.sin(thetas) * np.sin(phi)
    z = np.sin(thetas) * np.cos(phi)

    x_p, y_p, z_p, rotation, alphas = car_to_aGSE_constant(x,y,z,True,simple_ab,**kwargs)
    theta_ps = np.arccos(x_p) # angle from aberrated axis

    # Compute the radial distances based on the selected model
    if surface == 'BS':
        if model == 'jelinek':
            r  = bs_jelinek2012(theta_ps, **kwargs)
            R0 = bs_jelinek2012(0, **kwargs)
        else:
            raise ValueError(f'Model {model} not valid')
    elif surface == 'MP':
        if model == 'shue':
            r  = mp_shue1998(theta_ps, **kwargs)
            R0 = mp_shue1998(0, **kwargs)
        elif model == 'jelinek':
            r  = mp_jelinek2012(theta_ps, **kwargs)
            R0 = mp_jelinek2012(0, **kwargs)
        else:
            raise ValueError(f'Model {model} not valid')
    else:
        raise ValueError(f'Surface {surface} not valid')

    x_p *=  r
    y_p *=  r
    z_p *=  r

    # Invert back to standard GSE
    x, y, z, rotate_inv = aGSE_to_car_constant(x_p,y_p,z_p,True,simple_ab,rotation,**kwargs)
    rho = np.sqrt(y**2 + z**2)

    nose = rotate_inv.apply([R0,0,0])
    alpha_tot = np.arccos(nose[0]/R0)

    return {'x': x, 'y': y, 'z': z, 'r': r, 'rho': rho, 'R0': R0, 'nose': nose, 'alpha_z': alphas['alpha_z'], 'alpha_y': alphas['alpha_y'], 'alpha_tot': alpha_tot}



def bs_jelinek2012(theta, **kwargs):
    """
    Computes the boundary distance for the magnetosphere based on the Jelinek et al. (2012) model.

    This model calculates the boundary distance for a given angle and dynamic pressure (in nPa).

    Parameters
    ----------
    theta : numpy.ndarray
        The angle from the x-axis (model assumes cylindrical symmetry).
        Example: theta = np.arange(-np.pi + 0.01, np.pi - 0.01, 0.001).
    **kwargs :
        Additional parameters for the model.
        - "Pd" : float, optional, default=2.056
            The dynamic pressure in nPa.

    Returns
    -------
    r : numpy.ndarray
        The boundary distance for each value of `theta` in the input array.
        The values are in terms of the radial distance from the Earth's center.
    """
    # Retrieve dynamic pressure from kwargs, with default value
    Pd = kwargs.get('Pd', 2.056)

    # Constants for the model determined from least squares
    lam = 1.17
    R = 15.02
    epsilon = 6.55

    # Compute R0 based on dynamic pressure if not provided
    R0 = R * Pd ** (-1 / epsilon) # stand-off distance

    # Compute cosine and sine of theta
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    # Calculate the boundary distance
    r = np.where(
        np.isclose(theta, np.pi),
        np.inf,
        2 * R0 / (cos_th + np.sqrt(cos_th ** 2 + sin_th ** 2 * lam ** 2))
    )

    return r




def mp_jelinek2012(theta, **kwargs):
    """
    Computes the boundary distance for the magnetosphere based on the Jelinek et al. (2012) model.

    This model calculates the boundary distance for a given angle and dynamic pressure (in nPa).

    Parameters
    ----------
    theta : numpy.ndarray
        The angle from the x-axis (model assumes cylindrical symmetry).
        Example: theta = np.arange(-np.pi + 0.01, np.pi - 0.01, 0.001).
    **kwargs :
        Additional parameters for the model.
        - "Pd" : float, optional, default=2.056
            The dynamic pressure in nPa.

    Returns
    -------
    r : numpy.ndarray
        The boundary distance for each value of `theta` in the input array.
        The values are in terms of the radial distance from the Earth's center.
    """
    # Retrieve dynamic pressure from kwargs, with default value
    Pd = kwargs.get('Pd', 2.056)

    # Constants for the model
    lam = 1.54
    R = 12.82
    epsilon = 5.26

    # Compute R0 based on dynamic pressure
    R0 = R * Pd ** (-1 / epsilon) # stand-off distance

    # Compute cosine and sine of theta
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    # Calculate the boundary distance
    r = np.where(
        np.isclose(theta, np.pi),
        np.inf,
        2 * R0 / (cos_th + np.sqrt(cos_th ** 2 + sin_th ** 2 * lam ** 2))
    )

    return r


def mp_shue1998(theta, **kwargs):
    """
    Computes the magnetopause (MP) distance based on the Shue et al. (1998) model.

    This model calculates the magnetopause distance for a given angle, dynamic pressure (in nPa),
    and Bz component of the interplanetary magnetic field (IMF) (in nT).

    Parameters
    ----------
    theta : numpy.ndarray
        The angle from the x-axis (model assumes cylindrical symmetry).
        Example: theta = np.arange(-np.pi + 0.01, np.pi - 0.01, 0.001).
    **kwargs :
        Additional parameters for the model.
        - "Pd" : float, optional, default=2.056
            The dynamic pressure in nPa.
        - "Bz" : float, optional, default=-0.001
            The z-component of the IMF in nT.

    Returns
    -------
    r : numpy.ndarray
        The magnetopause distance for each value of `theta` in the input array.
        The values are in terms of the radial distance from the Earth's center.
    """
    # Retrieve dynamic pressure and IMF Bz from kwargs, with default values
    Pd = kwargs.get('Pd', 2.056)
    Bz = kwargs.get('Bz', -0.001)

    # Compute R0 and a based on Pd and Bz
    R0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd ** (-1 / 6.6)
    a = (0.58 - 0.007 * Bz) * (1 + 0.024 * np.log(Pd))

    # Calculate the magnetopause distance
    r = np.where(
        np.isclose(theta, np.pi),
        np.inf,
        R0 * (2 / (1 + np.cos(theta))) ** a
    )

    return r


def plot_magnetosheath_boundaries():
    """
    Plots the magnetosheath boundaries using the Jelinek and Shue models.

    This function generates a plot of the magnetopause (MP) and bow shock (BS) boundaries based on typical solar wind conditions.
    The plot includes boundaries calculated from the Jelinek et al. (2012) model for both the MP and BS, as well as the Shue 1998 model for the MP.

    The Earth is plotted at the origin, and the boundary lines are displayed for each model.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Displays the plot with the magnetosheath boundaries.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Grid settings
    plt.grid(linestyle='--', lw=0.5)

    # Calculate boundaries for each model
    bs_jel = msh_boundaries('jelinek','bs')
    mp_jel = msh_boundaries('jelinek','mp')
    mp_shu = msh_boundaries('shue','mp')

    # Plot Earth at the origin
    plt.scatter(0, 0, color='blue', marker='o', s=800)  # Earth

    # Plot the boundaries for each model
    plt.plot(bs_jel[0], bs_jel[1], label='Jelinek BS', linestyle='-', color='blue')
    plt.plot(mp_jel[0], mp_jel[1], label='Jelinek MP', linestyle='-.', color='green')
    plt.plot(mp_shu[0], mp_shu[1], label='Shue MP', linestyle=':', color='red')

    # Add labels and title
    plt.suptitle('Magnetosheath Boundaries for Typical Solar Wind Conditions', fontsize=18)
    plt.xlabel(r'$r_x$ [$R_E$] (GSE)', fontsize=16)
    plt.gca().invert_xaxis()  # Invert x-axis for GSE
    plt.ylabel(r'$\sqrt{r_y^2 + r_z^2}$ [$R_E$] (GSE)', fontsize=16)

    # Adjust y-axis position
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # Add legend
    plt.legend(loc="upper left")

    # Show the plot
    plt.show()

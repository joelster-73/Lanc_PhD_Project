import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .config import DEFAULT_COLUMN_NAMES
from .spatial import car_to_aGSE, car_to_aGSE_constant, aGSE_to_car_constant, cartesian_to_spherical


def calc_msh_r_diff(df, surface, model=None, aberration='model', position_key=None, data_key=None, column_names=None, inc_nose=False, **kwargs):

    if model is None:
        model = 'jelinek' if surface =='BS' else 'shue'

    if aberration=='simple':
        simple_ab=True
    elif aberration=='model':
        simple_ab = True if model in ('jelinek','shue') else False
    elif aberration=='complete':
        simple_ab = False

    if column_names is None:
        column_names = DEFAULT_COLUMN_NAMES
    else:
        column_names = column_names.copy()

    if position_key is not None:
        for key in ['r_x_name', 'r_y_name', 'r_z_name', 'r_name', 'r_ax_name', 'r_ay_name', 'r_az_name']:
            column_names[key] += f'_{position_key}'

    if data_key is not None:
        for key in ['v_x_name', 'v_y_name', 'v_z_name', 'p_name', 'bz_name']:
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
    if df_ab.empty:
        raise Exception('Coordinates could not be aberrated.')

    r_name    = column_names['r_name']
    r_ax_name = column_names['r_ax_name']
    r_ay_name = column_names['r_ay_name']
    r_az_name = column_names['r_az_name']
    v_x_name  = column_names['v_x_name']
    p_name    = column_names['p_name']
    bz_name   = column_names['bz_name']

    # Checks data for aberration
    valid_mask = ~df_ab[v_x_name].isna()
    if not valid_mask.any():
        print('Cannot calculate boundaries.')
        return pd.DataFrame()

    # Defaults if data not passed in for empirical models
    try:
        p = df.loc[valid_mask,p_name].to_numpy()
    except:
        p = 2.056

    try:
        Bz = df.loc[valid_mask,bz_name].to_numpy()
    except:
        Bz = -0.001

    theta_ps = np.arccos(
        df_ab.loc[valid_mask, r_ax_name] / df_ab.loc[valid_mask, r_name]
    )

    # Compute the radial distances based on the selected model
    if surface == 'BOTH':
        df_ab.loc[valid_mask, 'r_phi'] = theta_ps

        if model=='shue':
            print('Using Shue mp.')
            r_mp      = mp_shue1998(theta_ps, Pd=p, Bz=Bz, **kwargs)
            r_mp_nose = mp_shue1998(0, Pd=p, Bz=Bz, **kwargs)
        else:
            print('Using Jelínek mp.')
            r_mp       = mp_jelinek2012(theta_ps, Pd=p, **kwargs)
            r_mp_nose  = mp_jelinek2012(0, Pd=p, **kwargs)

        df_ab.loc[valid_mask, 'r_MP'] = r_mp

        print('Using Jelínek bs.')
        df_ab.loc[valid_mask, 'r_BS'] = bs_jelinek2012(theta_ps, Pd=p, **kwargs)
        r_bs_nose                     = bs_jelinek2012(0, Pd=p, **kwargs)

        if inc_nose:
            print('include')

        df_ab.loc[valid_mask, 'r_F'] = (df_ab.loc[valid_mask, r_name] - df_ab.loc[valid_mask, 'r_MP']) / (df_ab.loc[valid_mask, 'r_BS'] - df_ab.loc[valid_mask, 'r_MP'])

        df_ab.loc[~valid_mask, ['r_MP','r_BS','r_phi','r_F']] = np.nan

    else:

        if surface == 'BS':
            if model == 'jelinek':
                print('Using Jelínek bs.')
                r  = bs_jelinek2012(theta_ps, Pd=p, **kwargs)
            else:
                raise ValueError(f'Model {model} not valid')

        elif surface == 'MP':
            if model == 'shue':
                print('Using Shue mp.')
                r  = mp_shue1998(theta_ps, Pd=p, Bz=Bz, **kwargs)
            elif model == 'jelinek':
                print('Using Jelínek mp.')
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


def calc_normal_for_sc(df, surface, model=None, aberration='model', position_key=None, data_key=None, column_names=None, **kwargs):

    if model is None:
        model = 'jelinek' if surface =='BS' else 'shue'

    if aberration=='simple':
        simple_ab=True
    elif aberration=='model':
        simple_ab = True if model in ('jelinek','shue') else False
    elif aberration=='complete':
        simple_ab = False

    if column_names is None:
        column_names = DEFAULT_COLUMN_NAMES

    else:
        column_names = column_names.copy()

    if position_key is not None:
        for key in ['r_x_name', 'r_y_name', 'r_z_name', 'r_name', 'r_ax_name', 'r_ay_name', 'r_az_name']:
            column_names[key] += f'_{position_key}'

    if data_key is not None:
        for key in ['v_x_name', 'v_y_name', 'v_z_name', 'p_name', 'bz_name']:
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
    df_ab, rotation_matrix = car_to_aGSE(df, column_names=column_names, simple=simple_ab, return_rotation=True)

    r_ax_name = column_names['r_ax_name']
    r_ay_name = column_names['r_ay_name']
    r_az_name = column_names['r_az_name']
    v_x_name  = column_names['v_x_name']
    p_name    = column_names['p_name']
    bz_name   = column_names['bz_name']

    valid_mask = ~df_ab[v_x_name].isna()
    if not valid_mask.any():
        raise ValueError('No valid velocity data.')

    try:
        p = df.loc[valid_mask,p_name].to_numpy()
    except:
        print('Using default pressure.')
        p = 2.056

    try:
        B = df.loc[valid_mask,bz_name].to_numpy()
    except:
        print('Using default velocity.')
        B = -0.001

    _, theta_ps, phis_ps = cartesian_to_spherical(df_ab.loc[valid_mask, r_ax_name],
                                                  df_ab.loc[valid_mask, r_ay_name],
                                                  df_ab.loc[valid_mask, r_az_name])

    # Compute the normal based on the selected model and sc location
    if surface == 'MP':
        if model == 'shue':
            print('Using Shue mp.')
            n = mp_shue1998_normal(theta_ps.to_numpy(), phis_ps.to_numpy(), Pd=p, Bz=B)

        elif model == 'jelinek':
            print('Using Jelínek mp.')
            raise ValueError('Jelínek mp normal not implemented.')

        else:
            raise ValueError(f'Model {model} not valid')

    elif surface == 'BS':
        if model == 'jelinek':
            print('Using Jelínek bs.')
            raise ValueError('Jelínek bs normal not implemented.')

        else:
            raise ValueError(f'Model {model} not valid')

    else:
        raise ValueError(f'Surface {surface} not valid')

    # Convert back to GSE from aberrated
    rotate_inv = rotation_matrix.inv()
    n_GSE      =  rotate_inv.apply(n)

    return pd.DataFrame(n_GSE, index=df_ab.loc[valid_mask].index, columns=[f'N{comp}_GSE_{surface}' for comp in ('x','y','z')])



# %% Models

def bs_jelinek2012(theta, **kwargs):

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


def mp_shue1998_normal(theta, phi, **kwargs):

    # Retrieve dynamic pressure and IMF Bz from kwargs, with default values
    Pd = kwargs.get('Pd', 2.056)
    Bz = kwargs.get('Bz', -0.001)

    # Compute a based on Pd and Bz
    alpha = (0.58 - 0.007 * Bz) * (1 + 0.024 * np.log(Pd))

    # Unit vectors
    sin_th, cos_th = np.sin(theta), np.cos(theta)
    sin_ph, cos_ph = np.sin(phi),   np.cos(phi)

    e_r = np.stack([cos_th,  sin_th*sin_ph, sin_th*cos_ph], axis=-1)
    e_t = np.stack([-sin_th, cos_th*sin_ph, cos_th*cos_ph], axis=-1)

    n = e_r - (alpha * sin_th / (1 + cos_th))[:, None] * e_t
    n /= np.linalg.norm(n, axis=-1, keepdims=True)

    return n


def plot_magnetosheath_boundaries():

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

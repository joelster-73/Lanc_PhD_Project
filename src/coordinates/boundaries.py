import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .spatial import car_to_aGSE, aGSE_to_car, cartesian_to_spherical
from .magnetic import convert_GSE_to_GSM_with_angles

from ..config import DEFAULT_VALUES

P_SW  = DEFAULT_VALUES.get('sw',{}).get('p')
V_SW   = DEFAULT_VALUES.get('sw',{}).get('v')
V_MSH  = DEFAULT_VALUES.get('msh',{}).get('v')
BZ_SW  = DEFAULT_VALUES.get('sw',{}).get('Bz')

def calc_msh_dist(df, mp='shue', bs='jelinek', aberration='model', position_key=None, data_key=None):

    simple_ab = False
    if aberration in ('simple','model'):
        simple_ab = True

    key_p = ''
    if position_key is not None:
        key_p = f'_{position_key}'

    key_d = ''
    if data_key is not None:
        key_d = f'_{data_key}'

    r_mag  = df[f'r_mag{key_p}'].to_numpy()

    rx = df[f'r_x_GSE{key_p}'].to_numpy()
    ry = df[f'r_y_GSE{key_p}'].to_numpy()
    rz = df[f'r_z_GSE{key_p}'].to_numpy()

    vx = df[f'V_x_GSE{key_d}'].to_numpy()
    vy = df[f'V_y_GSE{key_d}'].to_numpy()
    vz = df[f'V_z_GSE{key_d}'].to_numpy()

    # Rotated to aGSE
    coords_ab = car_to_aGSE(rx, ry, rz, vx, vy, vz, simple_ab, return_rotation=False)
    rx_ab = coords_ab[:,0]

    p = df[f'P_flow{key_d}'].to_numpy()
    p = np.where(np.isnan(p), P_SW, p) # default pressure from config (above)

    Bz = df[f'B_z_GSM{key_d}'].to_numpy()
    Bz = np.where(np.isnan(Bz), BZ_SW, Bz) # default field from config (above)

    theta_ps = np.arccos(rx_ab / r_mag) # cone angle

    # Updating new dict

    data_ab = {}
    data_ab.update({'r_x_aGSE{key_p}': coords_ab[:, 0], 'r_y_aGSE{key_p}': coords_ab[:, 1], 'r_z_aGSE{key_p}': coords_ab[:, 2], 'r_phi': theta_ps})

    # Compute the radial distances based on the selected model

    if mp=='shue':
        #print('Using Shue mp.')
        r_mp      = mp_shue1998(theta_ps, Pd=p, Bz=Bz)
    else:
        #print('Using Jelínek mp.')
        r_mp       = mp_jelinek2012(theta_ps, Pd=p)

    if bs=='jelinek':
        #print('Using Jelínek bs.')
        r_bs       = bs_jelinek2012(theta_ps, Pd=p)

    data_ab.update({'r_MP': r_mp, 'r_BS': r_bs})

    r_F = (r_mag - r_mp) / (r_bs - r_mp) # distance into MSH based on MP and BS at that cone angle
    data_ab['r_F'] = r_F

    return pd.DataFrame(data_ab, index=df.index)


def vector_component_surface(df, sc, region, data_pop):
    """
    Calculate the perpendicular and parallel component of a vector along the normal of the bow shock or magnetopause surface

    """
    surfaces  = {'sw': 'BS', 'msh': 'MP'}
    norm_vecs = {'field': ('B',), 'plasma': ('B','E','V','S')}

    surface = surfaces[region]
    couple_vecs = norm_vecs[data_pop]

    suffix = f'_{sc}'

    if surface=='MP': # BS not currently implemented
        normals = calc_normal_for_sc(df, surface, position_key=sc, data_key='sw')

        normals_gsm = convert_GSE_to_GSM_with_angles(normals, (list(normals.columns),), df_coords=df, coords_suffix='sw')
        df = pd.concat([df,normals_gsm],axis=1)

        norm_cols = [f'N{comp}_GSM_{surface}' for comp in ('x','y','z')]
        N = df[norm_cols].to_numpy()
        for vec in couple_vecs:

            A_cols = [f'{vec}_{comp}_GSM{suffix}' for comp in ('x','y','z')]
            if A_cols[0] not in df:
                A_cols[0] = A_cols[0].replace('GSM','GSE')

            A = df[A_cols].to_numpy()
            A_dot_N   = np.einsum('ij,ij->i', A, N)
            A_norm_sq = np.einsum('ij,ij->i', A, A)

            with np.errstate(divide='ignore', invalid='ignore'):
                tangential_mag = np.sqrt(A_norm_sq - (A_dot_N ** 2))
                tangential_mag = np.nan_to_num(tangential_mag)  # Replace NaNs with 0

            df[f'{vec}_perp{suffix}']     = A_dot_N
            df[f'{vec}_parallel{suffix}'] = tangential_mag

        df.rename(columns={col: f'{col}{suffix}' for col in norm_cols}, inplace=True) # adds _sc suffix


def calc_normal_for_sc(df, surface, model='shue', aberration='model', position_key=None, data_key=None, **kwargs):
    """
    Claculates the normal of a surface at the cone angle of the spacecraft
    """

    simple_ab = False
    if aberration in ('simple','model'):
        simple_ab = True

    key_p = ''
    if position_key is not None:
        key_p = f'_{position_key}'

    key_d = ''
    if data_key is not None:
        key_d = f'_{data_key}'

    r_mag  = df[f'r_mag{key_p}'].to_numpy()

    rx = df[f'r_x_GSE{key_p}'].to_numpy()
    ry = df[f'r_y_GSE{key_p}'].to_numpy()
    rz = df[f'r_z_GSE{key_p}'].to_numpy()

    vx = df[f'V_x_GSE{key_d}'].to_numpy()
    vy = df[f'V_y_GSE{key_d}'].to_numpy()
    vz = df[f'V_z_GSE{key_d}'].to_numpy()

    # Rotated to aGSE
    coords_ab, rotation_matrix, _ = car_to_aGSE(rx, ry, rz, vx, vy, vz, simple_ab, return_rotation=True)
    rx_ab = coords_ab[:,0]

    p = df[f'P_flow{key_d}'].to_numpy()
    p = np.where(np.isnan(p), P_SW, p) # default pressure from config (above)

    Bz = df[f'B_z_GSM{key_d}'].to_numpy()
    Bz = np.where(np.isnan(Bz), BZ_SW, Bz) # default field from config (above)

    theta_ps = np.arccos(rx_ab / r_mag) # cone angle

    # Updating new dict

    _, theta_ps, phis_ps = cartesian_to_spherical(coords_ab[:,0], coords_ab[:,1], coords_ab[:,2])

    # Compute the normal based on the selected model and sc location
    if (surface, model) == ('MP', 'shue'):
        n = mp_shue1998_normal(theta_ps, phis_ps, Pd=p, Bz=Bz)

    else:
        print(f'({surface}, {model}) not implemented.')

    # Convert back to GSE from aberrated
    rotate_inv = rotation_matrix.inv()
    n_GSE      =  rotate_inv.apply(n)

    return pd.DataFrame(n_GSE, index=df.index, columns=[f'N{comp}_GSE_{surface}' for comp in ('x','y','z')])

# %% Models

def bsn_jelinek2012(pressure=2.056):

    R = 15.02
    epsilon = 6.55

    return R * pressure ** (-1 / epsilon) # stand-off distance


def bs_jelinek2012(theta, **kwargs):

    # Retrieve dynamic pressure from kwargs, with default value
    Pd = kwargs.get('Pd', P_SW)

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
    return np.where(np.isclose(theta, np.pi), np.inf, 2 * R0 / (cos_th + np.sqrt(cos_th ** 2 + sin_th ** 2 * lam ** 2)))


def mp_jelinek2012(theta, **kwargs):

    # Retrieve dynamic pressure from kwargs, with default value
    Pd = kwargs.get('Pd', P_SW)

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
    return np.where(np.isclose(theta, np.pi), np.inf, 2 * R0 / (cos_th + np.sqrt(cos_th ** 2 + sin_th ** 2 * lam ** 2)))


def mp_shue1998(theta, **kwargs):

    # Retrieve dynamic pressure and IMF Bz from kwargs, with default values
    Pd = kwargs.get('Pd', P_SW)
    Bz = kwargs.get('Bz', BZ_SW)

    # Compute R0 and a based on Pd and Bz
    R0 = (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd ** (-1 / 6.6)
    a = (0.58 - 0.007 * Bz) * (1 + 0.024 * np.log(Pd))

    # Calculate the magnetopause distance
    return np.where(np.isclose(theta, np.pi), np.inf, R0 * (2 / (1 + np.cos(theta))) ** a)


def mp_shue1998_normal(theta, phi, **kwargs):

    # Retrieve dynamic pressure and IMF Bz from kwargs, with default values
    Pd = kwargs.get('Pd', P_SW)
    Bz = kwargs.get('Bz', BZ_SW)

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


# %% planar-boundaries


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


def msh_boundaries(model, surface='BS', aberration='model', **kwargs):
    """
    The boundary of the magnetosheath surface in a particular plane, e.g., X-Y plane
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

    x_p, y_p, z_p, rotation, alphas = car_to_aGSE(x,y,z,True,simple_ab,**kwargs)
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
    x, y, z, rotate_inv = aGSE_to_car(x_p,y_p,z_p,True,simple_ab,rotation,**kwargs)
    rho = np.sqrt(y**2 + z**2)

    nose = rotate_inv.apply([R0,0,0])
    alpha_tot = np.arccos(nose[0]/R0)

    return {'x': x, 'y': y, 'z': z, 'r': r, 'rho': rho, 'R0': R0, 'nose': nose, 'alpha_z': alphas['alpha_z'], 'alpha_y': alphas['alpha_y'], 'alpha_tot': alpha_tot}
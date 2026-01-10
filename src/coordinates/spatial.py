import pandas as pd
import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64 #ensures backward compatibility with code expecting np.float_

from scipy.spatial.transform import Rotation as R
from uncertainties import UFloat, unumpy as unp
from spacepy.coordinates import Coords
from spacepy.time import Ticktock


from .config import DEFAULT_COLUMN_NAMES
from ..processing.utils import add_unit

v_Earth = 29.78 # km/s


def car_to_aGSE(df, column_names=None, simple=False, return_rotation=False):

    if column_names is None:
        column_names = DEFAULT_COLUMN_NAMES
    else:
        column_names = column_names.copy()

    r_x_name = column_names['r_x_name']
    r_y_name = column_names['r_y_name']
    r_z_name = column_names['r_z_name']
    r_name   = column_names['r_name']
    r_ax_name = column_names['r_ax_name']
    r_ay_name = column_names['r_ay_name']
    r_az_name = column_names['r_az_name']
    v_x_name  = column_names['v_x_name']
    v_y_name  = column_names['v_y_name']
    v_z_name  = column_names['v_z_name']

    df_aGSE = pd.DataFrame(index=df.index)

    # Magnitude of cluster vector
    df_aGSE[r_name] = (df[r_x_name]**2 + df[r_y_name]**2 + df[r_z_name]**2) ** 0.5

    try:
        df_aGSE[v_x_name] = df[v_x_name]
    except:
        df_aGSE[v_x_name] = np.full(len(df),-400) # default is v_x = -400

    if simple:
        v_Earth = 30
        df_aGSE[v_y_name] = np.zeros(len(df))
        df_aGSE[v_z_name] = np.zeros(len(df))
    else:
        try:
            df_aGSE[v_y_name] = df[v_y_name]
        except:
            df_aGSE[v_y_name] = np.zeros(len(df)) # default is v_y = 0

        try:
            df_aGSE[v_z_name] = df[v_z_name]
        except:
            df_aGSE[v_z_name] = np.zeros(len(df)) # default is v_z = 0

    valid_mask = ~df_aGSE[v_x_name].isna()

    # Rotation only for valid rows
    if not valid_mask.any():
        print('No valid data for aberration.')
        if return_rotation:
            return pd.DataFrame(), np.array([])
        return pd.DataFrame()

    # Aberration
    df_aGSE.loc[valid_mask, 'v_y_shift'] = df_aGSE.loc[valid_mask, v_y_name] + v_Earth

    df_aGSE.loc[valid_mask, 'alpha_z'] = -np.arctan(
        df_aGSE.loc[valid_mask, 'v_y_shift'] / np.abs(df_aGSE.loc[valid_mask, v_x_name])
    )
    df_aGSE.loc[valid_mask, 'alpha_y'] = np.arctan(
        -df_aGSE.loc[valid_mask, v_z_name] /
        np.sqrt(df_aGSE.loc[valid_mask, v_x_name]**2 + df_aGSE.loc[valid_mask, 'v_y_shift']**2)
    )

    R_z = R.from_euler('z', -df_aGSE.loc[valid_mask, 'alpha_z'].to_numpy(), degrees=False)
    R_y = R.from_euler('y',  df_aGSE.loc[valid_mask, 'alpha_y'].to_numpy(), degrees=False)

    rotation = R_y * R_z
    coords = np.column_stack((
        df.loc[valid_mask, r_x_name],
        df.loc[valid_mask, r_y_name],
        df.loc[valid_mask, r_z_name]
    ))

    rotated_coords = rotation.apply(coords)
    df_aGSE.loc[valid_mask, r_ax_name] = rotated_coords[:, 0]
    df_aGSE.loc[valid_mask, r_ay_name] = rotated_coords[:, 1]
    df_aGSE.loc[valid_mask, r_az_name] = rotated_coords[:, 2]

    if return_rotation:
        return df_aGSE, rotation
    return df_aGSE

def car_to_aGSE_constant(x, y, z, return_rotation=False, simple=False, **kwargs):

    # Same solar wind conditions/transformation applied to all coordinates

    coords = np.column_stack((x, y, z))

    v_x = kwargs.get('v_sw_x',-400)

    if simple:
        v_Earth = 30
        v_y, v_z = 0, 0
    else:
        v_y = kwargs.get('v_sw_y',0)
        v_z = kwargs.get('v_sw_z',0)

    v_y_shift = v_y + v_Earth

    # Rotation about Z
    alpha_z = -np.arctan(v_y_shift / np.abs(v_x))
    R_z = R.from_euler('z', -alpha_z, degrees=False)

    # Rotation about Y
    alpha_y = np.arctan(-v_z/np.sqrt(v_x**2+v_y_shift**2))
    R_y     = R.from_euler('y', alpha_y, degrees=False)

    #"_p" is for "prime"

    rotation = R_y * R_z
    x_p, y_p, z_p = rotation.apply(coords).T

    if return_rotation:
        return x_p, y_p, z_p, rotation, {'alpha_z': alpha_z, 'alpha_y': alpha_y}
    return x_p, y_p, z_p



def aGSE_to_car_constant(x_p, y_p, z_p, return_rotation=False, simple=False, rotation_matrix=None, **kwargs):

    # Same solar wind conditions/transformation applied to all coordinates

    coords_p = np.column_stack((x_p, y_p, z_p))

    if rotation_matrix is None:
        v_x = kwargs.get('v_sw_x',-400)

        if simple:
            v_Earth = 30
            v_y, v_z = 0, 0
        else:
            v_y = kwargs.get('v_sw_y',0)
            v_z = kwargs.get('v_sw_z',0)

        v_y_shift = v_y + v_Earth

        # Rotation about Z
        alpha_z = -np.arctan(v_y_shift / np.abs(v_x))
        R_z = R.from_euler('z', -alpha_z, degrees=False)

        # Rotation about Y
        alpha_y = np.arctan(-v_z/np.sqrt(v_x**2+v_y_shift**2))
        R_y     = R.from_euler('y', alpha_y, degrees=False)

        #"_p" is for "prime"
        rotation_matrix = R_y * R_z

    rotate_inv = rotation_matrix.inv()

    x, y, z =  rotate_inv.apply(coords_p).T

    if return_rotation:
        return x, y, z, rotate_inv

    return x, y, z


def insert_sph_coords(df, field='r', coords='GSE', **kwargs):


    x_col = kwargs.get('x_col',f'{field}_x_{coords}')
    y_col = kwargs.get('y_col',f'{field}_y_{coords}')
    z_col = kwargs.get('z_col',f'{field}_z_{coords}')

    r, theta, phi = cartesian_to_spherical(df[x_col], df[y_col], df[z_col])

    df[f'|{field}|'] = r
    df[f'{field}_theta_{coords}'] = theta
    df[f'{field}_phi_{coords}'] = phi

    units = df.attrs['units']
    units[f'|{field}|'] = units[x_col]
    units[f'{field}_theta_{coords}'] = add_unit('theta')
    units[f'{field}_phi_{coords}'] = add_unit('phi')

def insert_cyl_coords(df, field='r', coords='GSE', **kwargs):


    x_col = kwargs.get('x_col',f'{field}_x_{coords}')
    y_col = kwargs.get('y_col',f'{field}_y_{coords}')
    z_col = kwargs.get('z_col',f'{field}_z_{coords}')

    # '_' is x
    _, rho, phi = cartesian_to_cylindrical(df[x_col], df[y_col], df[z_col])

    df[f'{field}_rho'] = rho
    df[f'{field}_phi'] = phi

    units = df.attrs['units']
    units[f'{field}_rho'] = units[x_col]
    units[f'{field}_phi'] = add_unit('phi')

def insert_car_coords(df, field='r', coords='GSE', **kwargs):


    r_col = kwargs.get('r_col',f'{field}_mag')
    th_col = kwargs.get('th_col',f'{field}_theta_{coords}')
    ph_col = kwargs.get('ph_col',f'{field}_phi_{coords}')
    mag_data = df[r_col].to_numpy()
    the_data = df[th_col].to_numpy()
    phi_data = df[ph_col].to_numpy()

    x, y, z = spherical_to_cartesian(mag_data, the_data, phi_data)

    df[f'{field}_cos(th)'] = x
    df[f'{field}_sin(th)_sin(ph)'] = y
    df[f'{field}_sin(th)_cos(ph)'] = z

    units = df.attrs['units']
    field_unit = units[r_col]
    units[f'{field}_cos(th)'] = field_unit
    units[f'{field}_sin(th)_sin(ph)'] = field_unit
    units[f'{field}_sin(th)_cos(ph)'] = field_unit


def cartesian_to_cylindrical(x, y, z):


    rho = np.sqrt(y**2 + z**2)
    phi = np.arctan2(y, z)
    return x, rho, phi


def cartesian_to_spherical(x, y, z):

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.full_like(r, np.nan, dtype=float)
    phi   = np.full_like(r, np.nan, dtype=float)

    #theta = np.arccos(x / r)
    #phi = np.arctan2(y, z)

    mask = (r != 0)
    theta[mask] = np.arccos(np.clip(x[mask] / r[mask], -1.0, 1.0))
    phi[mask]   = np.arctan2(y[mask], z[mask])
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):

    if isinstance(theta,UFloat) or isinstance(phi,UFloat):
        x = r * unp.cos(theta)
        y = r * unp.sin(theta) * unp.sin(phi)
        z = r * unp.sin(theta) * unp.cos(phi)

    else:
        x = r * np.cos(theta)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.sin(theta) * np.cos(phi)

    return x, y, z


# %% GEO

def convert_GEO_position(glat, glon, times):
    """
    glat and glon in degrees
    """

    radius = 1.0  # Earth radii (ground)
    glat = np.radians(float(glat))
    glon = np.radians(float(glon))

    R_geo = radius * np.array([np.cos(glat)*np.cos(glon), np.cos(glat)*np.sin(glon), np.sin(glat)])
    R_geo = np.tile(R_geo, (len(times),1))

    ticks = Ticktock(times.to_pydatetime(), 'UTC')

    R_pos = Coords(R_geo, 'GEO', 'car', ticks=ticks)
    R_gse = R_pos.convert('GSE', 'car')
    R_gse = R_gse.data

    return pd.DataFrame(R_gse, index=times, columns=[f'r_{c}_GSE' for c in ('x','y','z')])

def convert_GEO_positions(df_positions):

    times = df_positions.index
    lat_rad = np.radians(df_positions['latitude'].to_numpy())
    lon_rad = np.radians(df_positions['longitude'].to_numpy())

    # GEO Cartesian positions (Earth radii)
    X = np.cos(lat_rad) * np.cos(lon_rad)
    Y = np.cos(lat_rad) * np.sin(lon_rad)
    Z = np.sin(lat_rad)
    R_geo = np.column_stack([X, Y, Z])

    # Tick times
    ticks = Ticktock(times.to_pydatetime(), 'UTC')

    # Convert GEO -> GSE
    R_coords = Coords(R_geo, 'GEO', 'car', ticks=ticks)
    R_gse = R_coords.convert('GSE', 'car').data

    return pd.DataFrame(R_gse, index=times, columns=['r_x_GSE', 'r_y_GSE', 'r_z_GSE'])


def convert_GEO_position_aGSE(glat, glon, times, coords='GSE', df_sw=None, V_earth=29.78):
    """
    glat and glon in degrees
    """

    radius = 1.0  # Earth radii (ground)
    glat = np.radians(float(glat))
    glon = np.radians(float(glon))

    R_geo = radius * np.array([np.cos(glat)*np.cos(glon), np.cos(glat)*np.sin(glon), np.sin(glat)])
    R_geo = np.tile(R_geo, (len(times),1))

    ticks = Ticktock(times.to_pydatetime(), 'UTC')

    R_pos = Coords(R_geo, 'GEO', 'car', ticks=ticks)
    R_gse = R_pos.convert('GSE', 'car')
    R_gse = R_gse.data

    if coords=='GSE' or df_sw is None:
        if df_sw is None:
            print('Don\'t have solar wind data; returning GSE data')
        return pd.DataFrame(R_gse, index=times, columns=[f'r_{c}_GSE' for c in ('x','y','z')])

    overlap = times.intersection(df_sw.index)

    # Aberration including Earth orbital speed
    V_vals  = df_sw.loc[overlap, ['V_x_GSE', 'V_y_GSE']].values
    alpha      = -np.arctan((V_earth + V_vals[:,1])/np.abs(V_vals[:,0]))
    cosa, sina = np.cos(alpha), np.sin(alpha)

    r_x_aGSE =  R_gse[:,0]*cosa + R_gse[:,1]*sina
    r_y_aGSE = -R_gse[:,0]*sina + R_gse[:,1]*cosa
    r_z_aGSE =  R_gse[:,2]  # Z unchanged

    R_agse = np.stack([r_x_aGSE, r_y_aGSE, r_z_aGSE], axis=1)

    return pd.DataFrame(R_agse, index=times, columns=[f'r_{c}_aGSE' for c in ('x','y','z')])

import pandas as pd
import numpy as np

if not hasattr(np, "float_"):
    np.float_ = np.float64 #ensures backward compatibility with code expecting np.float_

from scipy.spatial.transform import Rotation as R
from uncertainties import unumpy as unp
from spacepy.coordinates import Coords
from spacepy.time import Ticktock

from ..processing.utils import add_unit
from ..config import DEFAULT_VALUES

v_Earth = 29.78 # km/s

P_DYN  = DEFAULT_VALUES.get('sw',{}).get('p')
V_SW   = DEFAULT_VALUES.get('sw',{}).get('v')
V_MSH  = DEFAULT_VALUES.get('msh',{}).get('v')
BZ_SW  = DEFAULT_VALUES.get('sw',{}).get('Bz')


def car_to_aGSE(rx, ry, rz, vx=np.nan, vy=np.nan, vz=np.nan, simple=False, return_rotation=False):
    """
    r_x_aGSE = rotated_coords[:,0]
    r_y_aGSE = rotated_coords[:,1]
    r_z_aGSE = rotated_coords[:,2]
    """

    rotation, alpha_y, alpha_z = calc_agse_matrix(vx, vy, vz, v_Earth, simple)
    coords = np.column_stack((rx,ry,rz))

    rotated_coords = rotation.apply(coords)

    if return_rotation:
        return rotated_coords, rotation, {'alpha_z': alpha_z, 'alpha_y': alpha_y}

    return rotated_coords

def calc_agse_matrix(vx, vy, vz, vE, simple=False):

    vx = np.where(np.isnan(vx), V_SW, vx)

    if simple:
        vE = 30
        vy = 0
        vz = 0
    else:
        vy = np.where(np.isnan(vy), 0, vy)
        vz = np.where(np.isnan(vy), 0, vy)

    # Aberration
    vy_shift = vy + vE

    alpha_z = -np.arctan(vy_shift / np.abs(vx))
    alpha_y =  np.arctan(-vz / np.sqrt(vx**2 + vy_shift**2))

    R_z = R.from_euler('z', -alpha_z, degrees=False)
    R_y = R.from_euler('y',  alpha_y, degrees=False)

    return R_y * R_z, alpha_y, alpha_z


def aGSE_to_car(x_p, y_p, z_p, return_rotation=False, simple=False, rotation_matrix=None, **kwargs):

    # Same solar wind conditions/transformation applied to all coordinates

    coords_p = np.column_stack((x_p, y_p, z_p))

    if rotation_matrix is None:
        v_x = kwargs.get('v_x',np.nan)
        v_y = kwargs.get('v_y',np.nan)
        v_z = kwargs.get('v_z',np.nan)

        rotation_matrix, _, _ = calc_agse_matrix(v_x, v_y, v_z, simple)

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

    if np.asarray(theta).dtype == object or np.asarray(phi).dtype == object:
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

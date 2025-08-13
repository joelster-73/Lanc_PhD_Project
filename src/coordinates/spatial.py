import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

if not hasattr(np, "float_"):
    np.float_ = np.float64 #ensures backward compatibility with code expecting np.float_

from ..processing.utils import add_unit

v_Earth = 29.78 # km/s


def car_to_aGSE(df, position_key=None, data_key=None, simple=False):


    df_aGSE = pd.DataFrame(index=df.index)

    r_x_name = 'r_x_GSE'
    r_y_name = 'r_y_GSE'
    r_z_name = 'r_z_GSE'
    r_name   = 'r'

    r_ax_name = 'r_x_aGSE'
    r_ay_name = 'r_y_aGSE'
    r_az_name = 'r_z_aGSE'

    if position_key is not None:
        for name in (r_x_name,r_y_name,r_z_name,r_name,r_ax_name,r_ay_name,r_az_name):
            name += f'_{position_key}'

    v_x_name = 'v_x_GSE'
    v_y_name = 'v_y_GSE'
    v_z_name = 'v_z_GSE'

    if data_key is not None:
        for name in (v_x_name,v_y_name,v_z_name):
            name += f'_{data_key}'

    # Magnitude of cluster vector
    df_aGSE[r_name] = np.sqrt(df[r_x_name]**2 +
                              df[r_y_name]**2 +
                              df[r_z_name]**2)
    try:
        df_aGSE[v_x_name] = df[v_x_name]
    except:
        df_aGSE[v_x_name] = np.full(len(df),-400) # default is v_x=-400

    if simple:
        df_aGSE[v_y_name] = np.zeros(len(df))
        df_aGSE[v_z_name] = np.zeros(len(df))
    else:
        try:
            df_aGSE[v_y_name] = df[v_y_name]
        except:
            df_aGSE[v_y_name] = np.zeros(len(df)) # default is v_y=0

        try:
            df_aGSE[v_z_name] = df[v_z_name]
        except:
            df_aGSE[v_z_name] = np.zeros(len(df)) # default is v_z=0

    valid_mask = ~df_aGSE[v_x_name].isna()

    # Rotation only for valid rows
    if valid_mask.any():

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


    return df_aGSE

def car_to_aGSE_constant(x, y, z, return_rotation=False, simple=False, **kwargs):

    # Same solar wind conditions/transformation applied to all coordinates

    coords = np.column_stack((x, y, z))

    v_x = kwargs.get('v_sw_x',-400)

    if simple:
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
    """
    Converts the Cartesian coordinates of a field to spherical and cylindrical coordinates, then adds the results as new columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the Cartesian coordinates for the field (e.g., 'r_x_GSE', 'r_y_GSE', 'r_z_GSE').

    field : str, optional
        The name of the field to process (default is 'r').

    coords : str, optional
        The coordinate system used (default is 'GSE').

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added columns for the magnitude, cylindrical, and spherical coordinates of the field.
    """
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
    """
    Converts the Cartesian coordinates of a field to spherical and cylindrical coordinates, then adds the results as new columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the Cartesian coordinates for the field (e.g., 'r_x_GSE', 'r_y_GSE', 'r_z_GSE').

    field : str, optional
        The name of the field to process (default is 'r').

    coords : str, optional
        The coordinate system used (default is 'GSE').

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added columns for the magnitude, cylindrical, and spherical coordinates of the field.
    """
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
    """
    Converts the Cartesian coordinates of a field to spherical and cylindrical coordinates, then adds the results as new columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the Cartesian coordinates for the field (e.g., 'r_x_GSE', 'r_y_GSE', 'r_z_GSE').

    field : str, optional
        The name of the field to process (default is 'r').

    coords : str, optional
        The coordinate system used (default is 'GSE').

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with added columns for the magnitude, cylindrical, and spherical coordinates of the field.
    """
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
    """
    Converts Cartesian coordinates (x, y, z) to cylindrical coordinates (r, rho, phi).

    Parameters
    ----------
    x, y, z : numpy.ndarray
        The x, y, z coordinates in Cartesian coordinates.

    Returns
    -------
    tuple
        A tuple containing the x coordinate (unchanged), the radial distance rho, and the azimuthal angle phi in cylindrical coordinates.
    """
    rho = np.sqrt(y**2 + z**2)
    phi = np.arctan2(y, z)
    return x, rho, phi


def cartesian_to_spherical(x, y, z):
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).

    Parameters
    ----------
    x, y, z : numpy.ndarray
        The x, y, z coordinates in Cartesian coordinates.

    Returns
    -------
    tuple
        A tuple containing the radial distance r, the polar angle theta, and the azimuthal angle phi in spherical coordinates.
    """
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(x / r)
    phi = np.arctan2(y, z)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """
    Converts spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).

    Parameters
    ----------
    r : numpy.ndarray
        The radial distance in spherical coordinates.

    theta : numpy.ndarray
        The polar angle (in radians) in spherical coordinates.

    phi : numpy.ndarray
        The azimuthal angle (in radians) in spherical coordinates.

    Returns
    -------
    tuple
        A tuple containing the x, y, and z coordinates in Cartesian coordinates.
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.sin(theta) * np.cos(phi)
    return x, y, z


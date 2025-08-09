import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

if not hasattr(np, "float_"):
    np.float_ = np.float64 #ensures backward compatibility with code expecting np.float_

from .boundaries import bs_jelinek2012
from ..processing.utils import add_unit


def calc_bs_pos(df, **kwargs):
    """
    Calculates the distance of the spacecraft and bow shock using the Jelinek 2012 model.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing spacecraft position data (in GSE coordinates) and plasma data.

    **kwargs :
        - sc_key : str, required
            The spacecraft key used to select the appropriate GSE coordinates (e.g., 'C1', 'C2', etc.).
        - Additional arguments can be passed as needed.

    Returns
    -------
    pd.DataFrame
        A DataFrame with 'epoch', 'r_<sc_key>', and 'r_BS' columns, where 'r_<sc_key>' is the spacecraft distance
        and 'r_BS' is the distance to the bow shock (calculated using the Jelinek 2012 model).
    """

    # Retrieve the spacecraft key from kwargs
    sc_key   = kwargs.get('sc_key', None)

    # New df
    df_bs = pd.DataFrame(index=df.index)

    r_x_name = f'r_x_GSE_{sc_key}'
    r_y_name = f'r_y_GSE_{sc_key}'
    r_z_name = f'r_z_GSE_{sc_key}'

    # Magnitude of cluster vector
    df_bs[f'r_{sc_key}'] = np.sqrt(df[r_x_name]**2 +
                                   df[r_y_name]**2 +
                                   df[r_z_name]**2)
    try:
        df_bs['v_x_GSE_OMNI'] = df['v_x_GSE_OMNI']
    except:
        df_bs['v_x_GSE_OMNI'] = np.full(len(df),-400) # default is v_x=-400

    try:
        df_bs['v_y_GSE_OMNI'] = df['v_y_GSE_OMNI']
    except:
        df_bs['v_y_GSE_OMNI'] = np.zeros(len(df)) # default is v_y=0

    try:
        df_bs['v_z_GSE_OMNI'] = df['v_z_GSE_OMNI']
    except:
        df_bs['v_z_GSE_OMNI'] = np.zeros(len(df)) # default is v_z=0

    valid_mask = ~df_bs['v_x_GSE_OMNI'].isna()

    # Rotation only for valid rows
    if valid_mask.any():
        v_Earth = 29.78
        df_bs.loc[valid_mask, 'v_y_shift'] = df_bs.loc[valid_mask, 'v_y_GSE_OMNI'] + v_Earth

        df_bs.loc[valid_mask, 'alpha_z'] = -np.arctan(
            df_bs.loc[valid_mask, 'v_y_shift'] / np.abs(df_bs.loc[valid_mask, 'v_x_GSE_OMNI'])
        )
        df_bs.loc[valid_mask, 'alpha_y'] = np.arctan(
            -df_bs.loc[valid_mask, 'v_z_GSE_OMNI'] /
            np.sqrt(df_bs.loc[valid_mask, 'v_x_GSE_OMNI']**2 + df_bs.loc[valid_mask, 'v_y_shift']**2)
        )

        R_z = R.from_euler('z', -df_bs.loc[valid_mask, 'alpha_z'].to_numpy(), degrees=False)
        R_y = R.from_euler('y',  df_bs.loc[valid_mask, 'alpha_y'].to_numpy(), degrees=False)

        rotation = R_y * R_z
        coords = np.column_stack((
            df.loc[valid_mask, r_x_name],
            df.loc[valid_mask, r_y_name],
            df.loc[valid_mask, r_z_name]
        ))

        rotated_coords = rotation.apply(coords)
        df_bs.loc[valid_mask, f'r_x_aGSE_{sc_key}'] = rotated_coords[:, 0]
        df_bs.loc[valid_mask, f'r_y_aGSE_{sc_key}'] = rotated_coords[:, 1]
        df_bs.loc[valid_mask, f'r_z_aGSE_{sc_key}'] = rotated_coords[:, 2]

        # r_BS only for valid rows
        try:
            p = df.loc[valid_mask,'p_flow_OMNI'].to_numpy()
        except:
            p = 2.056

        theta_ps = np.arccos(
            df_bs.loc[valid_mask, f'r_x_aGSE_{sc_key}'] / df_bs.loc[valid_mask, f'r_{sc_key}']
        )
        df_bs.loc[valid_mask, 'r_BS'] = bs_jelinek2012(theta_ps, Pd=p)

    # Set NaN for invalid rows
    df_bs.loc[~valid_mask, 'r_BS'] = np.nan
    df_bs.loc[~valid_mask, f'r_x_aGSE_{sc_key}'] = np.nan
    df_bs.loc[~valid_mask, f'r_y_aGSE_{sc_key}'] = np.nan
    df_bs.loc[~valid_mask, f'r_z_aGSE_{sc_key}'] = np.nan

    return df_bs



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


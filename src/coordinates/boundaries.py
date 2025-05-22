import numpy as np
from scipy.spatial.transform import Rotation as R

from matplotlib import pyplot as plt


def msh_boundaries(model, surface='bs', **kwargs):
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
    # Azimuthal angle - default is x-y plane
    phi = kwargs.get('phi',np.pi/2)

    # Generate the range of theta values
    thetas = kwargs.get('thetas',np.linspace(-np.pi/2, np.pi/2, 500))

    x = np.cos(thetas)
    y = np.sin(thetas) * np.sin(phi)
    z = np.sin(thetas) * np.cos(phi)
    coords = np.column_stack((x, y, z))

    # Aberrated coordinates
    v_x = kwargs.get('v_sw_x',-400)
    v_y = kwargs.get('v_sw_y',0)
    v_z = kwargs.get('v_sw_z',0)

    # Rotation about Z
    v_Earth  = 29.78
    v_y_rest = (v_y + v_Earth)
    alpha_z  = -np.arctan(v_y_rest/np.abs(v_x))
    R_z      = R.from_euler('z', -alpha_z)

    # Rotation about Y
    alpha_y = np.arctan(-v_z/np.sqrt(v_x**2+v_y_rest**2))
    R_y     = R.from_euler('y', alpha_y)

    #"_p" is for "prime"

    rotation = R_y * R_z
    x_p, y_p, z_p =  rotation.apply(coords).T

    theta_ps = np.arccos(x_p) # angle from aberrated axis

    # Compute the radial distances based on the selected model
    if surface == 'bs':
        if model == 'jelinek':
            r  = bs_jelinek2012(theta_ps, **kwargs)
            R0 = bs_jelinek2012(0, **kwargs)
        else:
            raise ValueError(f'Model {model} not valid')
    elif surface == 'mp':
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
    coords_p = np.column_stack((x_p, y_p, z_p))

    rotate_inv = rotation.inv()

    x, y, z =  rotate_inv.apply(coords_p).T
    rho = np.sqrt(y**2 + z**2)

    nose = rotate_inv.apply([R0,0,0])

    return {'x': x, 'y': y, 'z': z, 'rho': rho, 'R0': R0, 'nose': nose, 'alpha_z': alpha_z, 'alpha_y': alpha_y, 'r': r}



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
    bs_jel = bs_boundaries('jelinek')
    mp_jel = mp_boundaries('jelinek')
    mp_shu = mp_boundaries('shue')

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

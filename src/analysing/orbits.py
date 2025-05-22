# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:18:16 2025

@author: richarj2
"""


def fit_ellipse(x, y):
    """
    Fits a rotated ellipse to the data.

    Parameters:
    - x, y (array-like): Arrays of x and y positions.

    Returns:
    - a (float): Semi-major axis.
    - b (float): Semi-minor axis.
    - centre (tuple): Coordinates of the ellipse's centre (x_c, y_c).
    - theta (float): Rotation angle of the ellipse (in radians).
    """
    # Step 1: Construct the design matrix
    M = np.column_stack((x**2, x * y, y**2, x, y, np.ones(len(x))))
    b = np.zeros(len(x))

    # Step 2: Solve the general ellipse equation
    coef, _, _, _ = lstsq(M, b, rcond=None)
    A, B, C, D, E, F = coef
    print(coef)

    # Step 3: Centre of the ellipse
    x_c = (2 * C * D - B * E) / (B**2 - 4 * A * C)
    y_c = (2 * A * E - B * D) / (B**2 - 4 * A * C)

    # Step 4: Rotation angle
    theta = 0.5 * np.arctan2(B, A - C)

    # Step 5: Semi-major and semi-minor axes
    up = 2 * (A * E**2 + C * D**2 + F * B**2 - 2 * B * D * E - A * C * F)
    down1 = (B**2 - 4 * A * C) * ((A + C) - np.sqrt((A - C)**2 + B**2))
    down2 = (B**2 - 4 * A * C) * ((A + C) + np.sqrt((A - C)**2 + B**2))
    a = np.sqrt(up / down1)
    b = np.sqrt(up / down2)
    e = np.sqrt(1 - b**2/a**2)

    return a, b, (x_c, y_c), theta, e


def calc_period(df, x_label, y_label):
    """
    Calculate the orbit period based on how often the data crosses into the positive octant (x > 0, y > 0) in a 2D case.

    The period is approximated by the average time between successive crossings into the positive octant.

    Parameters:
    - df: pandas DataFrame with time as the index and columns corresponding to the x and y positions.
    - x_label: str, the column name for the x-coordinate.
    - y_label: str, the column name for the y-coordinate.

    Returns:
    - period: float, the average time period (in days) between successive crossings into the positive octant.
    """
    # Detect crossing of the x-axis (y goes from +ve to -ve or vice versa, and x is -ve or +ve)
    crossing_x = ((df[y_label].shift(1) <= 0) & (df[y_label] > 0) & (df[x_label] > 0))

    # Extract the times of the crossings
    crossing_times = df.index[crossing_x]

    print(crossing_times)

    # If there are fewer than two crossings, return None or a message indicating insufficient data
    if len(crossing_times) < 2:
        return None  # Or alternatively, return some indication like "Insufficient Data"

    # Calculate the time differences (periods) between successive crossings
    periods = crossing_times[1:] - crossing_times[:-1]

    # Estimate the period as two times the average time between x-axis crossings (since it's roughly half the period)
    estimated_period = periods.mean().total_seconds() / (86400)  # Convert from seconds to days

    return estimated_period

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:19:02 2025

@author: richarj2
"""
import data_plotting
from data_plotting import save_figure, add_figure_title, add_legend
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

proc_cros_dir      = '../Data/Processed_Data/Cluster1/Crossings/'

def calc_R_bs(Pd=2.056):
    R = 15.02
    epsilon = 6.55

    return R * Pd ** (-1 / epsilon)

def calc_r_bs(theta, R_bs=None, Pd=None):
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)
    lam = 1.17
    if R_bs is None:
        R_bs = calc_R_bs(Pd)
    return 2 * R_bs / (cos_th + np.sqrt(cos_th ** 2 + sin_th ** 2 * lam ** 2))

def calc_R_mp(Pd=2.056,Bz=-0.001):

    return (10.22 + 1.29 * np.tanh(0.184 * (Bz + 8.14))) * Pd ** (-1 / 6.6)

def calc_r_mp(theta, R_mp=None, Pd=None, Bz=-0.001):
    cos_th = np.cos(theta)
    if R_mp is None:
        R_mp = calc_R_mp(Pd,Bz)
    a = (0.58 - 0.007 * Bz) * (1 + 0.024 * np.log(Pd))
    return R_mp * (2 / (1 + cos_th)) ** a

def calc_B_bs_nose(B_sw, Pd, R_mp=None, R_bs=None):

    Bz = B_sw[2]
    B_imf = convert_imf_from_gse(B_sw)

    if R_mp is None:
        R_mp = calc_R_mp(Pd,Bz)

    if R_bs is None:
        R_bs = calc_R_bs(Pd)

    x = 0
    y = 0
    z = convert_from_gse(R_bs,R_mp)

    return calc_B_msh(x, y, z, B_imf, Pd, R_mp, R_bs)

def calc_B_mp_nose(B_sw, Pd, R_mp=None, R_bs=None):

    Bz = B_sw[2]
    B_imf = convert_imf_from_gse(B_sw)

    if R_mp is None:
        R_mp = calc_R_mp(Pd,Bz)

    if R_bs is None:
        R_bs = calc_R_bs(Pd)

    x = 0
    y = 0
    z = convert_from_gse(R_mp,R_mp)

    return calc_B_msh(x, y, z, B_imf, Pd, R_mp, R_bs)

def calc_B_msh(x, y, z, B_imf, Pd, R_mp=None, R_bs=None):
    # Compute radial distance
    r = np.sqrt(x**2 + y**2 + z**2)

    # Solar wind magnetic field components
    Bx, By, Bz = B_imf

    if R_mp is None:
        R_mp = calc_R_mp(Pd,Bz)

    if R_bs is None:
        R_bs = calc_R_bs(Pd)

    # MP: Magnetosphere
    v_mp = calc_v_mp(R_mp) - 0.001 # accounts for precision errors near MP
    if r - z < v_mp**2:
        return 0  # Return 0 if inside magnetosphere

    # BS: Bow shock
    v_bs = calc_v_bs(R_mp,R_bs) + 0.001 # accounts for precision errors near BS
    if r - z > v_bs**2:
        return np.linalg.norm(B_imf)  # Return B_imf if outside bow shock

    # Magnetic field compression and displacement terms
    B_con_term = (1 + (v_mp**2) / (v_bs**2 - v_mp**2)) * B_imf

    # Displacement terms
    factor = 1 / (r * (r - z))
    B_dis_x = factor * (Bx * (r - x**2 / (r - z)) - By * (x * y / (r - z)) + Bz * (x / 2))
    B_dis_y = factor * (-Bx * (x * y / (r - z)) + By * (r - y**2 / (r - z)) + Bz * (y / 2))
    B_dis_z = (z / r - 1) / ((r - z)**2) * (-Bx * x - By * y + Bz * (r - z) / 2)
    B_dis = np.array([B_dis_x, B_dis_y, B_dis_z])

    B_dis_term = (v_mp**2 * v_bs**2 / (v_bs**2 - v_mp**2)) * B_dis

    B_msh = B_con_term + B_dis_term

    return np.linalg.norm(B_msh)

def calc_v_mp(R_mp):

    return np.sqrt(R_mp)

def calc_v_bs(R_mp,R_bs):

    return np.sqrt(2*R_bs-R_mp)

def convert_to_gse(z,R_mp):

    return 0.5*(R_mp) - z

def convert_from_gse(xp,R_mp):

    return 0.5*(R_mp) - xp

def get_u_from_x(x,R_mp,R_bs):

    v = calc_v_bs(R_mp,R_bs)
    z = convert_from_gse(x,R_mp)
    diff = v**2 + 2*z

    diff[diff<0] = 0 # accounts for overflow

    return np.sqrt(diff)

def get_bs_pos(x,R_mp,R_bs):
    # In paper cartesian frame

    outside = x >= R_bs
    inside = x < R_bs
    if np.sum(outside):
        xs_out = convert_from_gse(x[outside], R_mp)
        ys_out = np.zeros(len(x[outside]))

    xs_in = x[inside]
    u = get_u_from_x(xs_in,R_mp,R_bs)
    v = calc_v_bs(R_mp,R_bs)

    u_neg = -u

    pos_x = 0.5 * (u**2 - v**2)

    pos1_y = u * v
    pos2_y = u_neg * v

    x_positions = np.concatenate((pos_x, pos_x))
    y_positions = np.concatenate((pos1_y, pos2_y))

    #print(x_positions)
    if np.sum(outside):
        x_positions = np.concatenate((xs_out,x_positions))
        y_positions = np.concatenate((ys_out,y_positions))

    return x_positions, y_positions

def convert_imf_from_gse(B_imf):

    Bx = B_imf[2]
    By = B_imf[1]
    Bz = -B_imf[0]

    return np.array([Bx,By,Bz])

def B_properties(B_values):

    B_min = np.min(B_values)
    B_avg = np.mean(B_values)
    B_max = np.max(B_values)

    return B_min, B_avg, B_max




def plot_field_lines(ax,B_sw,R_mp,R_bs,xmin=0):

    x_max = ax.get_xlim()[1]
    x_pos = np.linspace(xmin,x_max,int(np.linalg.norm(B_sw))+5)

    xs, zs = get_bs_pos(x_pos,R_mp,R_bs)
    xs = convert_to_gse(xs, R_mp)

    Bx = B_sw[0] # B-rho
    Bz = B_sw[2]

    y_min, y_max = ax.get_ylim()

    for i in range(len(xs)):
        x = xs[i]
        z = zs[i]
        if (z > 0 and z > y_max) or (z < 0 and z < y_min):
            continue
        if Bx == 0:
            if z > 0:
                ax.plot([x, x], [z, y_max], c='w', lw=1)
                line_mid = 0.5*(z+y_max)

            elif z < 0:
                ax.plot([x, x], [z, y_min], c='w', lw=1)
                line_mid = 0.5*(z+y_min)

            else:
                ax.axvline(x, c='w', lw=1)
                line_mid = 0

            ax.annotate('', xy=(x, line_mid+np.sign(Bz)*0.5), xytext=(x, line_mid),
                        arrowprops=dict(arrowstyle='->', color='w'))
            slope = np.sign(Bz)*np.inf

        else:
            # Calculate the slope and draw the line
            slope = Bz / Bx
            y_start = slope * (x - xs[0])
            y_end = slope * (x - xs[-1])

            if z > 0:
                ax.plot([xs[0], xs[-1]], [y_start, y_end], c='w', lw=1)
            elif z < 0:
                ax.plot([xs[0], xs[-1]], [y_start, y_end], c='w', lw=1)

            # Arrow
            dx = np.sign(Bx) * 0.5
            dy = np.abs(dx) * np.sign(Bz) * np.abs(slope)
            x_start = x
            y_start = 0
            x_end = x_start + dx
            y_end = y_start + dy

            ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                        arrowprops=dict(arrowstyle='->', color='w'))


def plot_over_x(B_sw,Pd,R_mp=None,R_bs=None,ratio=True):

    Bz = B_sw[2]

    if R_mp is None:
        R_mp = calc_R_mp(Pd,Bz)

    if R_bs is None:
        R_bs = calc_R_bs(Pd)

    B_imf = convert_imf_from_gse(B_sw) # switching x and z essentially

    xs_gse = np.linspace(int(R_mp)-2,int(R_bs)+2,100)
    zs = convert_from_gse(xs_gse,R_mp)

    Bs_msh = np.empty(len(zs))
    for i, z in enumerate(zs):
        Bs_msh[i] = calc_B_msh(0,0,z,B_imf,Pd,R_mp,R_bs)

    bmag_bs = calc_B_bs_nose(B_sw, Pd, R_mp, R_bs)
    bmag_mp = calc_B_mp_nose(B_sw, Pd, R_mp, R_bs)
    bmag_avg = np.mean(Bs_msh)

    if ratio:
        Bs_msh /= np.linalg.norm(B_imf)
        bmag_bs /= np.linalg.norm(B_imf)
        bmag_mp /= np.linalg.norm(B_imf)
        bmag_avg /= np.linalg.norm(B_imf)

    fig, ax = plt.subplots()

    ax.plot(xs_gse,Bs_msh,c='b')
    ax.set_xlabel('Distance along Earth-Sun line [$R_E$]')
    if ratio:
        ax.set_ylabel(r'$\frac{B_{msh}}{B_{imf}}$', rotation=0, labelpad=15)
    else:
        ax.set_ylabel('$B_{msh}$ (nT)')

    plt.text(R_bs+1,1.1,'$B_{IMF}$')
    plt.text((R_bs+R_mp)/2,bmag_bs-0.15,'$B_{MSH}$')
    plt.text(R_mp-0.1,0.05,'Magnetosphere')
    plt.text(R_bs+0.5,bmag_bs-0.1,f'{bmag_bs:.2g}')
    plt.text(R_mp-0.5,bmag_mp-0.1,f'{bmag_mp:.2g}')

    plt.title('Comparing magnetic field strengths')
    plt.gca().invert_xaxis()
    plt.show()


def plot_along_bs(B_sw,Pd,ratio=True,phi=0):

    R_mp = calc_R_mp(Pd,B_sw[2])
    R_bs = calc_R_bs(Pd)

    B_imf = convert_imf_from_gse(B_sw)

    v = calc_v_bs(R_mp,R_bs)
    us = np.linspace(-v,v,201)
    rhos = us * v
    zs = 0.5*(us**2-v**2)
    xs = convert_from_gse(zs,R_mp)

    B_msh = np.empty(len(zs))
    for i, (rho,z) in enumerate(zip(rhos,zs)):
        # phi = 0 => y=0 plane
        B_msh[i] = calc_B_msh(rho*np.cos(phi), rho*np.sin(phi), z, B_imf, Pd, R_mp, R_bs)

    bmag_bs = calc_B_bs_nose(B_sw, Pd, R_mp, R_bs)
    bmag_avg = np.mean(B_msh)
    if ratio:
        B_msh /= np.linalg.norm(B_imf)
        bmag_bs /= np.linalg.norm(B_imf)
        bmag_avg /= np.linalg.norm(B_imf)

    thetas = np.arctan(rhos/xs)

    fig, ax = plt.subplots()

    ax.plot(np.degrees(thetas),B_msh,c='b')
    ax.set_xlabel('Angle from Earth-Sun line (deg)')
    if ratio:
        ax.set_ylabel(r'$\frac{B_{msh}}{B_{imf}}$', rotation=0, labelpad=15)
    else:
        ax.set_ylabel('B_msh (nT)')
    plt.title(f'Comparing magnetic field strengths. IMF: {np.linalg.norm(B_sw):.2g} nT')
    plt.show()

def plot_through_msh(B_sw,Pd,R_mp=None,R_bs=None,ratio=True,plot_field=True,phi=0):
    Bz = B_sw[2]

    if R_mp is None:
        R_mp = calc_R_mp(Pd,Bz)

    if R_bs is None:
        R_bs = calc_R_bs(Pd)

    B_imf = convert_imf_from_gse(B_sw)

    z_min = 0 # mid way between R_mp and Earth
    x_min = convert_to_gse(0,R_mp)
    xs_gse = np.linspace(x_min,int(R_bs)+2,501)

    v_bs = calc_v_bs(R_mp,R_bs)
    rho_max = np.sqrt(2*z_min+v_bs**2)*v_bs
    rhos = np.linspace(-rho_max,rho_max,501) # choose range to be such that get +/- 90 deg coverage

    # Through MSH
    X, R = np.meshgrid(xs_gse, rhos)
    Z = convert_from_gse(X,R_mp)
    B_msh = np.zeros_like(X)

    for i in range(X.shape[0]): # xs

        for j in range(X.shape[1]): # rhos
            x, y, z = R[i, j]*np.cos(phi), R[i, j]*np.sin(phi), Z[i, j]
            B_msh[i,j] = calc_B_msh(x, y, z, B_imf, Pd, R_mp, R_bs)

    B_values = B_msh.flatten()
    B_msh_inside = B_values[B_values>np.linalg.norm(B_imf)]

    # Along bow shock
    us_bs = np.linspace(-v_bs,v_bs,501)
    rhos_bs = us_bs * v_bs
    zs_bs = 0.5*(us_bs**2-v_bs**2)

    B_msh_bs = np.empty(len(zs_bs))
    for i, (rho,z) in enumerate(zip(rhos_bs,zs_bs)):
        B_msh_bs[i] = calc_B_msh(rho*np.cos(phi), rho*np.sin(phi), z, B_imf, Pd, R_mp, R_bs)

    # Scaling
    bmag_bs = calc_B_bs_nose(B_sw, Pd, R_mp, R_bs)
    bmag_mp = calc_B_mp_nose(B_sw, Pd, R_mp, R_bs)
    if ratio:
        B_msh /= np.linalg.norm(B_imf)
        B_msh_bs /= np.linalg.norm(B_imf)
        bmag_bs /= np.linalg.norm(B_imf)
        bmag_mp /= np.linalg.norm(B_imf)
        B_msh_inside /= np.linalg.norm(B_imf)

    # Plotting
    fig = plt.figure()
    gs = GridSpec(5, 5, figure=fig)  # 5x5 grid
    ax_main = fig.add_subplot(gs[:-1, :-1])  # Main heatmap (top-left 4x4)
    ax_bottom = fig.add_subplot(gs[-1, :-1], sharex=ax_main)  # Bottom plot (1x4)
    ax_right = fig.add_subplot(gs[:-1, -1], sharey=ax_main)  # Right plot (4x1)

    X_GSE = X
    R_GSE = R

    # Main heatmap
    heatmap = ax_main.pcolormesh(X_GSE, R_GSE, B_msh, shading='auto', cmap='inferno')
    fig.colorbar(heatmap, ax=ax_main,
                 label=r'$\frac{B_{msh}}{B_{imf}}$' if ratio else '$B_{msh}$ [nT]')
    ax_main.axhline(ls=':',lw=0.5,c='w')
    ax_main.text(R_bs-0.45,0.25,f'{bmag_bs:.3g}',c='w')
    ax_main.scatter(R_bs,0,c='w',s=3)
    ax_main.invert_xaxis()

    ax_main.set_xlabel('X GSE [$R_E$]')
    if phi == 0:
        perp_axis = 'Z GSE [$R_E$]'
    elif phi == np.pi/2:
        perp_axis = 'Y GSE [$R_E$]'
    else:
        perp_axis = 'YZ (${phi}^\\circ$) GSE [$R_E$]'

    ax_main.set_ylabel(perp_axis)
    bmin, bavg, bmax = B_properties(B_msh_inside)
    ax_main.set_title(f'Min: {bmin:.2f}, Avg: {bavg:.2f}, Max: {bmax:.2f}')

    # Bottom plot: B along YZ = 0
    r_ind = np.argmin(np.abs(rhos))
    B_along_x = B_msh[r_ind,:]
    X_along_x = X_GSE[r_ind,:]

    ax_bottom.plot(X_along_x, B_along_x,c='b')
    ax_bottom.set_xlabel('X GSE [$R_E$]')
    ax_bottom.set_ylabel('B [nT]')
    ax_bottom.grid()

    ax_bottom.set_title(f'BS: {bmag_bs:.2f}, MP: {bmag_mp:.2f}')

    # Right plot: B along Y = 0
    B_along_bs = B_msh_bs
    R_along_bs = rhos_bs

    ax_right.plot(B_along_bs, R_along_bs,c='b')
    ax_right.set_ylabel(perp_axis)
    ax_right.set_xlabel('B [nT]')
    ax_right.grid()

    bmin, bavg, bmax = B_properties(B_along_bs)
    ax_right.set_title(f'Min: {bmin:.2f}\nAvg: {bavg:.2f}, Max: {bmax:.2f}')

    # Title
    plt.suptitle(r'$B_{msh}$ strength through BS. '
                 f'IMF: {np.linalg.norm(B_sw):.2g} nT')
    plt.tight_layout()  # Adjust title spacing
    plt.show()


def plot_over_bs(B_sw,Pd,ratio=True):

    # Stand offs
    R_mp = calc_R_mp(Pd, B_sw[2])
    R_bs = calc_R_bs(Pd)
    B_imf = convert_imf_from_gse(B_sw)

    # Define a grid in polar coordinates
    v_bs = calc_v_bs(R_mp,R_bs)
    phis = np.linspace(0, np.pi, 501)
    rhos = np.linspace(-v_bs**2, v_bs**2, 501)

    # Create polar grid and convert to Cartesian
    R, P = np.meshgrid(rhos, phis)  # (radius, angle)
    X = R * np.cos(P)
    Y = R * np.sin(P)

    U = R / v_bs
    Z = 0.5 * (U**2 - v_bs**2)

    # Initialise magnetic field strength array
    B_msh = np.zeros_like(R)

    for i in range(R.shape[0]): # rhos

        for j in range(R.shape[1]): # phis
            x, y, z = X[i, j], Y[i, j], Z[i, j]
            B_msh[i,j] = calc_B_msh(x, y, z, B_imf, Pd, R_mp, R_bs)


    bmag_bs = calc_B_bs_nose(B_sw, Pd, R_mp, R_bs)
    bmag_avg = np.mean(B_msh)
    if ratio:
        B_msh /= np.linalg.norm(B_imf)
        bmag_bs /= np.linalg.norm(B_imf)
        bmag_avg /= np.linalg.norm(B_imf)


    fig = plt.figure()
    gs = GridSpec(5, 5, figure=fig)  # 5x5 grid
    ax_main = fig.add_subplot(gs[:-1, :-1])  # Main heatmap (top-left 4x4)
    ax_bottom = fig.add_subplot(gs[-1, :-1], sharex=ax_main)  # Bottom plot (1x4)
    ax_right = fig.add_subplot(gs[:-1, -1], sharey=ax_main)  # Right plot (4x1)

    Z_GSE = X
    Y_GSE = Y

    # Main heatmap
    heatmap = ax_main.pcolormesh(Y_GSE, Z_GSE, B_msh, shading='auto', cmap='inferno')
    fig.colorbar(heatmap, ax=ax_main, label=r'$\frac{B_{msh}}{B_{imf}}$' if ratio else '$B_{msh}$ [nT]')
    ax_main.set_xlabel('Y GSE [$R_E$]')
    ax_main.set_ylabel('Z GSE [$R_E$]')
    ax_main.set_facecolor('k')

    bmin, bavg, bmax = B_properties(B_msh)
    ax_main.set_title(f'Min: {bmin:.2f}, Avg: {bavg:.2f}, Max: {bmax:.2f}')

    # Bottom plot: B along Z = 0
    phi_ind = int(np.where(np.isclose(phis, np.pi/2))[0][0])
    B_along_y = B_msh[phi_ind]
    Y_along_z = Y_GSE[phi_ind]

    ax_bottom.plot(Y_along_z, B_along_y,c='b')
    ax_bottom.set_xlabel('Y GSE [$R_E$]')
    ax_bottom.set_ylabel('B [nT]')
    ax_bottom.grid()

    # Right plot: B along Y = 0
    phi_ind = int(np.where(np.isclose(phis, 0))[0][0])
    B_along_z = B_msh[phi_ind]
    Z_along_y = Z_GSE[phi_ind]

    ax_right.plot(B_along_z, Z_along_y,c='b')
    ax_right.set_ylabel('Z GSE [$R_E$]')
    ax_right.set_xlabel('B [nT]')
    ax_right.grid()

    # Title
    plt.suptitle(r'$B_{msh}$ Strength Across BS. '
                 f'IMF: {np.linalg.norm(B_sw):.2g} nT')
    plt.tight_layout()  # Adjust title spacing
    plt.show()


def plot_over_B(Pd,B_unit=np.array([0,0,-1]),location='Over BS',quantity='avg',ratio=True,standoffs=False,max_B=120,show_both=False, save_data=False):
    
    if show_both:
        ratio=False
    
    B_unit /= np.linalg.norm(B_unit) # ensures unit vector
    steps=10
    Bs = np.linspace(0,max_B,steps*max_B+1)
    if ratio:
        Bs[0] = 0.01 # prevents divide by 0

    R_bs = calc_R_bs(Pd)
    
    B_msh = np.empty(len(Bs))
    Rs_mp = np.empty(len(Bs))
    Rs_bs = np.empty(len(Bs))
    
    B_ratio = np.empty(len(Bs))

    for i, B in enumerate(Bs):
        B_sw = B*B_unit
        R_mp = calc_R_mp(Pd,B_sw[2])
        B_imf = convert_imf_from_gse(B_sw)
        if location == 'BS':
            B_msh[i] = calc_B_bs_nose(B_sw, Pd, R_mp, R_bs)
        elif location == 'MP':
            B_msh[i] = calc_B_mp_nose(B_sw, Pd, R_mp, R_bs)
        elif location == 'Over BS':
            v_bs = calc_v_bs(R_mp,R_bs)
            phis = np.linspace(0, np.pi, 101)
            rhos = np.linspace(-v_bs**2, v_bs**2, 101)

            R, P = np.meshgrid(rhos, phis)
            X = R * np.cos(P)
            Y = R * np.sin(P)
            U = R / v_bs
            Z = 0.5 * (U**2 - v_bs**2)

            B_msh_i = np.zeros_like(R)
            for j in range(R.shape[0]): # rhos
                for k in range(R.shape[1]): # phis
                    x, y, z = X[j, k], Y[j, k], Z[j, k]
                    B_msh_i[j,k] = calc_B_msh(x, y, z, B_imf, Pd, R_mp, R_bs)

            if quantity == 'min':
                B_msh[i] = np.min(B_msh_i)
            elif quantity == 'max':
                B_msh[i] = np.max(B_msh_i)
            else:
                B_msh[i] = np.mean(B_msh_i)

        if ratio:
            B_msh[i] /= B
            
        if show_both:
            if i==0:
                continue
            B_ratio[i] = B_msh[i] / B

        Rs_bs[i] = R_bs
        Rs_mp[i] = R_mp

    fig, ax = plt.subplots()

    if show_both:
        ax.plot(Bs, B_msh, c='cyan', lw=3, marker='o', markersize=7, markerfacecolor='w', markevery=steps)
    else:
        ax.plot(Bs, B_msh, c='b', marker='o', markersize=7, markerfacecolor='w', markevery=steps)
        
    ax.set_xlabel(r'$|B|_{imf}$  [nT]')
    if ratio:
        ax.set_ylabel(r'$\frac{B_{msh}}{B_{imf}}$', rotation=0, labelpad=15)
    else:
        ax.set_ylabel(r'$|B|_{msh}$  [nT]')
    
    ax.grid(ls=':',c='gray')
    
    if show_both:
        B_ratio[0] = B_ratio[1]
        
        ax2 = ax.twinx()
        
        ax2.plot([], [], c='cyan', marker='o', markersize=7, markerfacecolor='w', label=r'$B_{msh}$')
        ax2.plot(Bs, B_ratio, c='k', marker='v', markersize=7, markerfacecolor='w', markevery=steps, label=r'$B_{msh}/B_{imf}$')
        
        ax2.text(Bs[0]+0.5, B_ratio[0]+0.01, f'{B_ratio[0]:.2f}', horizontalalignment='left', verticalalignment='center')
        ax2.text(Bs[-1] ,B_ratio[-1]+0.05, f'{B_ratio[-1]:.2f}', horizontalalignment='center', verticalalignment='center')
        ax2.set_ylabel(r'$\frac{B_{msh}}{B_{imf}}$', rotation=0, labelpad=15)
        add_legend(fig,ax2,loc='upper right')

    if standoffs:
        ax2 = ax.twinx()
        ax2.plot(Bs,Rs_bs,ls='--',c='r')
        ax2.plot(Bs,Rs_mp,ls='--',c='g')
        ax2.set_ylabel('Stand-off distances [$R_E$]')
    
    if show_both:
        title = 'Investigating Magnetosheath Compression'

    elif location == 'BS':
        title = 'Comparing magnetic field strengths at BS nose'
    elif location == 'MP':
        title = 'Comparing magnetic field strengths at MP nose'
    elif location == 'Over BS':
        title = 'Investigating Magnetosheath Compression over the Bowshock surface'
    add_figure_title(fig,title)
    plt.tight_layout();
    save_figure(fig)
    plt.show()
    plt.close()

    if save_data:
        if ratio:
            y = np.cumsum(B_msh * np.gradient(Bs))  # Approximate integration using cumulative sum

            imf = np.insert(Bs, 0, 0)
            msh = np.insert(y, 0, 0)
            rho = np.insert(B_msh, 0, B_msh[0])
        else:
            imf = Bs
            Bs[0] = Bs[1]
            msh = B_msh
            rho = B_msh/Bs

        file_path = os.path.join(proc_cros_dir, 'compression_ratios.npz')
        np.savez(file_path, B_imf=imf, B_msh=msh, B_rho=rho)

def plot_over_P(B_sw,location='Over BS',ratio=True,standoffs=False):

    Bz = B_sw[2]
    B_imf = convert_imf_from_gse(B_sw)

    Ps = np.linspace(1,20,501)

    B_msh = np.empty(len(Ps))
    Rs_mp = np.empty(len(Ps))
    Rs_bs = np.empty(len(Ps))

    for i, P in enumerate(Ps):
        R_mp = calc_R_mp(P,Bz)
        R_bs = calc_R_bs(P)
        Rs_bs[i] = R_bs
        Rs_mp[i] = R_mp
        if location == 'BS':
            B_msh[i] = calc_B_bs_nose(B_sw, P, R_mp, R_bs)
        elif location == 'MP':
            B_msh[i] = calc_B_mp_nose(B_sw, P, R_mp, R_bs)
        elif location == 'Over BS':
            v_bs = calc_v_bs(R_mp,R_bs)
            phis = np.linspace(0, np.pi, 101)
            rhos = np.linspace(-v_bs**2, v_bs**2, 101)

            R, Q = np.meshgrid(rhos, phis)
            X = R * np.cos(Q)
            Y = R * np.sin(Q)
            U = R / v_bs
            Z = 0.5 * (U**2 - v_bs**2)

            B_msh_i = np.zeros_like(R)
            for j in range(R.shape[0]): # rhos
                for k in range(R.shape[1]): # phis
                    x, y, z = X[j, k], Y[j, k], Z[j, k]
                    B_msh_i[j,k] = calc_B_msh(x, y, z, B_imf, P, R_mp, R_bs)

            B_msh[i] = np.mean(B_msh_i)

    if ratio:
        B_msh /= np.linalg.norm(B_imf)

    fig, ax = plt.subplots()

    ax.plot(Ps,B_msh,c='b')
    ax.set_xlabel('Bulk flow pressure (nPa)')
    if ratio:
        ax.set_ylabel(r'$\frac{B_{msh}}{B_{imf}}$', rotation=0, labelpad=15)
    else:
        ax.set_ylabel(r'$B_{msh}$ (nT)')

    if standoffs:
        ax2 = ax.twinx()
        ax2.plot(Ps,Rs_bs,ls='--',c='r')
        ax2.plot(Ps,Rs_mp,ls='--',c='g')
        ax2.set_ylabel('Stand-off distances [$R_E$]')

    if location == 'BS':
        title = 'Comparing magnetic field strengths at BS nose'
    elif location == 'MP':
        title = 'Comparing magnetic field strengths at MP nose'
    elif location == 'Over BS':
        title = 'Comparing magnetic field strengths over the BS surface'
    plt.title(title)
    plt.show()

Pd_0=1
R_mp_kob = 9
R_bs_kob = 12.5

def run_plots(B_sw, Pd, B_unit):

    plot_over_x(B_sw, Pd)

    plot_along_bs(B_sw,Pd)
    plot_over_bs(B_sw,Pd,ratio=True)

    plot_through_msh(B_sw,Pd,plot_field=False,phi=0)
    plot_through_msh(B_sw,Pd,plot_field=False,phi=np.pi/2)

    plot_over_B(Pd,B_unit,quantity='min',standoffs=True)
    plot_over_B(Pd,B_unit,quantity='min',ratio=False,standoffs=False,save_data=True)

    plot_over_B(Pd,B_unit,standoffs=True)
    plot_over_B(Pd,B_unit,ratio=False,standoffs=False,save_data=False)

    plot_over_P(B_sw,standoffs=True)


print('B-field perp')

B_dir = np.array([0, 0, -1])
B_unit_0 =  B_dir/np.linalg.norm(B_dir)
B_sw_0 = 10*B_unit_0 # B in IMF in GSE

#run_plots(B_sw_0, Pd_0, B_unit_0)
plot_over_B(Pd_0,B_unit_0,quantity='min',show_both=True,max_B=30)

#print('B-field perp positive')

B_dir = np.array([0, 0, 1])
B_unit_0 =  B_dir/np.linalg.norm(B_dir)
B_sw_0 = 10*B_unit_0 # B in IMF in GSE


#run_plots(B_sw_0, Pd_0, B_unit_0)


#print('\nB-field 45 deg')


B_dir = np.array([0, 1, -1])
B_unit_0 =  B_dir/np.linalg.norm(B_dir)
B_sw_0 = 10*B_unit_0 # B in IMF in GSE


#run_plots(B_sw_0, Pd_0, B_unit_0)

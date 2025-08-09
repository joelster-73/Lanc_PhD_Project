# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""


# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
n = 100
step_start = 40
step_end = 60
noise_level = 0.1
max_lag = 30

offset = 1

# Fixed signal 1
x = np.zeros(n)
x[step_start:step_end] = 1
x += np.random.normal(0, noise_level, n) + offset

# Template for signal 2 (to shift left over time)
y_template = np.zeros(n)
y_template[step_start:step_end] = 1
y_template += np.random.normal(0, noise_level, n) + offset

# Define lag values (shift y left → simulate shifting x right)
lags = np.arange(-max_lag, max_lag + 1)
corr = np.zeros_like(lags, dtype=float)

# Plot setup
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
t = np.arange(n)

# Plot fixed signal x
line1, = ax1.plot(t, x, label='Signal 1', color='blue')
# Placeholder for shifting y
line2, = ax1.plot([], [], label='Signal 2', color='red')
ax1.set_ylim(0.5, 2.5)
ax1.set_xlim(0, n)
ax1.set_xlabel('Amplitude')
ax1.legend(loc='upper left', edgecolor='white')
ax1.set_title('Shifting Step Signal')

# Cross-correlation plot
corr_line, = ax2.plot([], [], color='purple')
ax2.set_xlim(lags[0], lags[-1])
ax2.set_ylim(-1, 1)
ax2.set_title('Cross-correlation')
ax2.set_xlabel('Time Lag')
ax2.set_ylabel('Correlation')



lag_text = ax1.text(0.85, 0.95, '', transform=ax1.transAxes,
                    verticalalignment='top', fontsize=10, bbox=dict(facecolor='white', edgecolor='white', alpha=0.7))

def animate(i):
    lag = lags[i]

    # Shift signal 2 to the left by lag → same as shifting signal 1 to the right
    y_shifted = np.zeros(n)
    if lag < 0:
        y_shifted[-lag:] = y_template[:n+lag]  # shift left
    elif lag > 0:
        y_shifted[:n-lag] = y_template[lag:]   # shift right
    else:
        y_shifted = y_template.copy()

    # Compute correlation over overlapping region
    mask = (x != 0) & (y_shifted != 0)
    if np.any(mask):
        x_seg = x[mask] - np.mean(x[mask])
        y_seg = y_shifted[mask] - np.mean(y_shifted[mask])
        denom = np.linalg.norm(x_seg) * np.linalg.norm(y_seg)
        corr_val = np.dot(x_seg, y_seg) / denom if denom != 0 else 0
    else:
        corr_val = 0
    corr[i] = corr_val

    # Update shifting signal plot
    y_visible = np.where(y_shifted != 0, y_shifted, np.nan)  # hide zero regions
    line2.set_data(t, y_visible)

    # Update correlation plot
    corr_line.set_data(lags[:i+1], corr[:i+1])

    lag_text.set_text(f'Time Lag: {lags[i]+max_lag}')

    return line1, line2, corr_line, lag_text

ani = animation.FuncAnimation(fig, animate, frames=len(lags), interval=100, blit=True, repeat=False)

lag_ticks = ax2.get_xticks()
ax2.set_xticks(ticks=lag_ticks, labels=lag_ticks+max_lag)

plt.tight_layout()
plt.show()


#ani.save("cross_correlation.gif", writer="pillow")
ani.save('cross_correlation_animation.mp4', writer='ffmpeg', fps=10)
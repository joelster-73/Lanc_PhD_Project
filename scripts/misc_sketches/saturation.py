# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:33:37 2025

@author: richarj2
"""


# %%
import numpy as np
import matplotlib.pyplot as plt

def my_func(x, k=1):
    return 1-np.exp(-k*x)

xs = np.linspace(0,1,100)
ys = my_func(xs,5)

fig, ax = plt.subplots(figsize=(12,6),dpi=400,facecolor='black')

ax.plot(xs,ys,c='w',lw=2)
ax.set_xlabel('Driver',c='w',fontsize=20)
ax.set_ylabel('Response',c='w',fontsize=20)

ax.set_facecolor('black')
#ax.tick_params(colors='white')              # tick labels
#ax.spines['bottom'].set_color('white')      # bottom axis
#ax.spines['top'].set_color('white')         # top axis
#ax.spines['left'].set_color('white')        # left axis
#ax.spines['right'].set_color('white')       # right axis
ax.xaxis.label.set_color('white')           # x-axis label
ax.yaxis.label.set_color('white')           # y-axis label
ax.title.set_color('white')  

x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

# Draw white arrows for axes
arrow_style = dict(facecolor='white', edgecolor='white', linewidth=3, headwidth=12, headlength=12)

# X-axis arrow
ax.annotate('', xy=(x_max, 0), xytext=(x_min, 0),
            arrowprops=dict(arrowstyle='->', color='white', lw=3))

# Y-axis arrow
ax.annotate('', xy=(0, y_max), xytext=(0, y_min),
            arrowprops=dict(arrowstyle='->', color='white', lw=3))


plt.show()
# %% Imports

from section_0 import cluster1, omni
# %% Orbit
from src.plotting.space_time import plot_orbit

plot_orbit(cluster1, plane='x-rho', coords='GSE', models='Typical Both',
           display='Heat', bin_width=0.1, brief_title='Cluster\'s orbit from 2001 to 2023', equal_axes=True, df_omni=omni)


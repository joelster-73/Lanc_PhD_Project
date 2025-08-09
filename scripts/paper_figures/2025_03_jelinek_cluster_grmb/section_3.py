# %% Imports

from section_0 import c1_omni, c1_sw_jel, c1_sw_grmb, c1_omni_jel_0


# %% Initial_comparison
from src.plotting.comparing.parameter import compare_columns
from src.config import COMPRESSIONS_DIR

compare_columns(c1_omni_jel_0, 'B_avg_C1', 'B_avg_OMNI',
                display='Heat', bin_width=(0.2,0.4), brief_title='Comparing |B| for Cluster and OMNI',
                compressions=COMPRESSIONS_DIR, contam_info=True)
# %% Kobel


#Find correct kobel figure
#Switch y-axes around, use same colours tho so blue is the B_msh



# %% Against_r
from src.plotting.comparing.parameter import investigate_difference

# the parameter file needs looking at - it's awful
# probably split into different files

investigate_difference(c1_omni_jel_0, 'B_avg_C1', 'B_avg_OMNI', ind_col='r_bs_diff_C1',
                display='Heat', bin_width=(0.05,0.5), brief_title='Comparing |B| for Cluster and OMNI',
                x_data_name='r_C1 - r_BS')

# %% Best_Buffer
from src.config import COMPRESSIONS_DIR
from src.methods.bowshock_filtering.bowshock_buffer import plot_best_buffer

plot_best_buffer(c1_omni, 'r_bs_diff_C1', 'B_avg_C1', 'B_avg_OMNI', scale='lin',
                 compressions=COMPRESSIONS_DIR, buff_max=6, data_name='Jelínek BS')

# %% Not_needed

from data_plotting import plot_method_sketch

plot_method_sketch()



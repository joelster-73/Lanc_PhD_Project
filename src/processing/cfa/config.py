# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:24:20 2025

@author: richarj2
"""
# event last processed used to remove duplicates

HTML_TAG_STRINGS = ['Year','Fractional day of year',
            'Arrival time of shock [seconds of day]','Shock type',
            'X GSE location of spacecraft [Re]','Y GSE location of spacecraft [Re]','Z GSE location of spacecraft [Re]',
            'Event last processed','Propagation delay to Earth [minutes]',
            'Ni [n/cc]', 'Vx GSE [km/s]', 'Vy GSE [km/s]', 'Vz GSE [km/s]',
            'Bx GSE [nT]', 'By GSE [nT]', 'Bz GSE [nT]', 'Method selected']

HTML_TAGS_UP_DW = ['Ni [n/cc]', 'Vx GSE [km/s]', 'Vy GSE [km/s]', 'Vz GSE [km/s]',
                  'Bx GSE [nT]', 'By GSE [nT]', 'Bz GSE [nT]']

HTML_TAG_LABELS = ['year', 'day', 'time_of_day', 'type', 'r_x_GSE', 'r_y_GSE', 'r_z_GSE',
                   'process_time','delay_s', 'ni_up', 'ni_dw', 'v_x_GSE_up', 'v_x_GSE_dw', 'v_y_GSE_up', 'v_y_GSE_dw', 'v_z_GSE_up', 'v_z_GSE_dw',
                   'B_x_GSE_up', 'B_x_GSE_dw', 'B_y_GSE_up', 'B_y_GSE_dw', 'B_z_GSE_up', 'B_z_GSE_dw',
                   'method', 'Nx', 'Ny', 'Nz', 'v_sh']

# tags with uncertainty to splot
HTML_TAGS_UNCERTAINTY = ['ni_up', 'ni_dw', 'delay_s', 'v_x_GSE_up', 'v_x_GSE_dw', 'v_y_GSE_up', 'v_y_GSE_dw', 'v_z_GSE_up', 'v_z_GSE_dw',
                         'B_x_GSE_up', 'B_x_GSE_dw', 'B_y_GSE_up', 'B_y_GSE_dw', 'B_z_GSE_up', 'B_z_GSE_dw', 'Nx', 'Ny', 'Nz', 'v_sh']




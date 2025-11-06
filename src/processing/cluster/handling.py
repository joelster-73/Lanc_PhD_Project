import os
import re
import glob

import numpy as np
import pandas as pd
from spacepy import pycdf
import spacepy.pycdf.istp

from scipy.constants import mu_0, m_p
from scipy.constants import physical_constants
m_a = physical_constants['alpha particle mass'][0]

from ..handling import create_log_file, log_missing_file
from ..writing import write_to_cdf
from ..dataframes import add_df_units, resample_data
from ..reading import import_processed_data
from ..utils import create_directory
from ...coordinates.magnetic import calc_B_GSM_angles, GSE_to_GSM_with_angles
from ...config import R_E, PROC_OMNI_DIR_5MIN

def process_cluster_files(directory, data_directory, variables, sample_interval='1min', time_col='epoch', year=None, sub_folders=False, overwrite=True, quality_directory=None, quality_variables=None):

    print('Processing Cluster.')

    ###----------DIRECTORIES----------###

    directory_name = os.path.basename(os.path.normpath(directory))
    if 'FGM' in directory_name:
        save_directory = os.path.join(data_directory, 'field')
    else:
        save_directory = os.path.join(data_directory, 'plasma')

    log_file_path = os.path.join(save_directory, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
    create_log_file(log_file_path)

    raw_dir = os.path.join(save_directory, 'raw')
    create_directory(raw_dir)

    if sample_interval != 'None':
        samp_dir = os.path.join(save_directory, sample_interval)
        create_directory(samp_dir)

    # Added as CDF attribute
    if '5VPS' in directory_name:
        raw_sample = '5VPS'
    elif 'SPIN' in directory_name or 'CIS' in directory_name:
        raw_sample = 'SPIN'

    ###----------FILES----------###

    files_by_year = get_cluster_files(directory, year, sub_folders)

    quality_files = {}
    if quality_directory and quality_variables:
        quality_files = get_cluster_files(quality_directory, sub_folders=sub_folders)

    for the_year, files in files_by_year.items():
        print(f'Processing {the_year} data.')
        yearly_list = []

        # Loop through each daily file in the year
        for cdf_file in files:
            file_date = os.path.basename(cdf_file).split('__')[1].split('_')[0]
            try:  # Bad data check
                new_df = pd.DataFrame(extract_cluster_data(cdf_file, variables))
                if new_df.empty:
                    log_missing_file(log_file_path, file_date, 'Empty file.')
                    continue

                # Should move to a separate update function !
                if 'FGM' in directory:
                    gsm = calc_B_GSM_angles(new_df, time_col=time_col).drop(columns=['B_mag'])
                    new_df = pd.concat([new_df, gsm], axis=1)

                yearly_list.append(new_df)
                if False:
                    print(f'{file_date} file appended.')

            except Exception as e:
                log_missing_file(log_file_path, file_date, e)

        if not yearly_list:
            print(f'No data for {the_year}')
            continue

        yearly_df = pd.concat(yearly_list, ignore_index=True)

        if len(quality_files)>0:
            print('Adding quality...')
            quality_list = []

            for qual_file in quality_files.get(the_year,[]):
                qual_df = pd.DataFrame(extract_cluster_data(qual_file, quality_variables))
                if qual_df.empty:
                    continue

                quality_list.append(qual_df)

            if len(quality_list)>0:
                quality_df = pd.concat(quality_list, ignore_index=True)
                quality_df['epoch'] = pd.to_datetime(quality_df['epoch'], errors='coerce')
                quality_df = quality_df.dropna(subset=['epoch'])

                merged = pd.merge_asof(yearly_df, quality_df, left_on='epoch', right_on='epoch', direction='backward')

                yearly_df['quality'] = merged['quality'].values
                print('Quality added.')
            else:
                yearly_df['quality'] = -2 # indicate no quality data
                print('No quality data.')

        add_df_units(yearly_df)

        ###----------WRITE RAW DATA----------###
        output_file = os.path.join(raw_dir, f'{directory_name}_{the_year}.cdf')

        attributes = {'sample_interval': raw_sample, 'time_col': time_col, 'R_E': R_E}
        write_to_cdf(yearly_df, output_file, attributes, overwrite)

        ###----------RESAMPLE----------###

        if sample_interval == 'None':
            continue

        print('Resampling.')
        # resample and write to file
        sampled_df = resample_data(yearly_df, time_col, sample_interval)
        add_df_units(sampled_df)

        output_file = os.path.join(samp_dir, f'{directory_name}_{the_year}.cdf')
        attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E': R_E}
        write_to_cdf(sampled_df, output_file, attributes, overwrite)

        print(f'{the_year} processed.')


def get_cluster_files(directory=None, year=None, sub_folders=False):

    files_by_year = {}

    if directory is None:
        directory = os.getcwd()

    if sub_folders:
        # Loop through subdirectories named as years
        for sub_folder in sorted(os.listdir(directory)):
            sub_folder_path = os.path.join(directory, sub_folder)
            if year and sub_folder != str(year):
                continue  # Skip folders not matching the specified year

            pattern = os.path.join(sub_folder_path, '*.cdf')  # Match all .cdf files in the folder
            for cdf_file in sorted(glob.glob(pattern)):
                if 'CAVEATS' in cdf_file:
                    continue
                files_by_year.setdefault(sub_folder, []).append(cdf_file)
    else:
        # Build the search pattern
        if year:
            pattern = os.path.join(directory, f'*__{year}*.cdf')
        else:
            pattern = os.path.join(directory, '*.cdf')

        # Use glob to find files matching the pattern

        for cdf_file in sorted(glob.glob(pattern)):
            if 'CAVEATS' in cdf_file:
                continue

            # Extract the year from the filename assuming format data-name__YYYYMMDD_version.cdf
            file_year = os.path.basename(cdf_file).split('__')[1][:4]
            files_by_year.setdefault(file_year, []).append(cdf_file)

    if year and str(year) not in files_by_year:
        raise ValueError(f'No files found for {year}.')
    if not files_by_year:
        raise ValueError('No files found.')

    return files_by_year


def extract_cluster_data(cdf_file, variables):


    data_dict = {}
    expected_len = None

    with pycdf.CDF(cdf_file) as cdf:

        for var_name, var_code in variables.items():

            if var_code not in cdf:
                continue

            data = cdf[var_code].copy()
            spacepy.pycdf.istp.nanfill(data)
            data = data[...]

            if isinstance(data,float) or isinstance(data,int):
                # Empty dataset
                return {}

            ###-----------CHECKING LENGTHS----------###
            current_len = len(data) if data.ndim > 0 else 1
            if expected_len is None:
                expected_len = current_len
            elif current_len != expected_len:
                raise ValueError('Data arrays not same length.')


            if data.ndim == 2 and data.shape[1] == 3:  # Assuming a 2D array for vector components
                if var_name == 'r':
                    data /= R_E  # Scales distances to multiples of Earth radii

                data_dict[f'{var_name}_mag']   = np.sqrt(data[:,0]**2+data[:,1]**2+data[:,2]**2)
                data_dict[f'{var_name}_x_GSE'] = data[:, 0]
                data_dict[f'{var_name}_y_GSE'] = data[:, 1]
                data_dict[f'{var_name}_z_GSE'] = data[:, 2]

            else:
                if var_name == 'T_ion':
                    # data is in MK
                    data *= 1e6
                data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed

    return data_dict

# %%
def combine_spin_data(spin_directory, fvps_directory=None, year=None, omni_dir=PROC_OMNI_DIR_5MIN):

    field_dir = os.path.join(spin_directory, 'field', 'raw')
    plasma_dir = os.path.join(spin_directory, 'plasma', 'raw')
    combined_dir = os.path.join(spin_directory, 'combined', 'raw')

    year_range = range(2001,2023)
    if year is not None:
        year_range = (year,)

    for year in year_range:
        try:
            field_df = import_processed_data(field_dir, year=year)
            plasma_df = import_processed_data(plasma_dir, year=year)
        except:
            print(f'No data for {year}.')
            continue

        print(f'Processing {year} data.')

        ###----------GSE to GSM----------###

        # Align field on plasma timestamps (as plasma contains mode, quality etc. numbers)
        field_df = field_df.reindex(plasma_df.index, method=None).interpolate(method='time')

        merged_df = pd.concat([field_df, plasma_df], axis=1)
        vec_coords = 'GSE'

        if fvps_directory is not None:

            convert_cols = ('V',)

            fvps_df = import_processed_data(fvps_directory, year=year)
            gsm_vectors = GSE_to_GSM_with_angles(merged_df, [[f'{vec}_{comp}_GSE' for comp in ('x','y','z')] for vec in convert_cols], df_coords=fvps_df, ref='B', interp=True)

            for vec in convert_cols:
                merged_df.drop(columns=[f'{vec}_x_GSE', f'{vec}_y_GSE', f'{vec}_z_GSE'],inplace=True)

            merged_df = pd.concat([merged_df,gsm_vectors], axis=1)
            vec_coords = 'GSM'

            # Freeing up memory
            del fvps_df

            print('GSE to GSM.')

        # For now treating all species together
        merged_df.rename(columns={'P_ion': 'P_th', 'T_ion': 'T_tot', 'N_ion': 'N_tot'}, inplace=True)


        print('Calculations...')
        ###----------CONTAMINATION----------###

        merged_df.loc[merged_df['B_avg']>200,'B_avg'] = np.nan
        for comp in ('x','y','z'):
            merged_df.loc[merged_df[f'B_{comp}_GSM'].abs()>150,f'B_{comp}_GSM'] = np.nan

        merged_df.loc[merged_df['V_mag']>2e3,'V_mag'] = np.nan
        for comp, limit in zip(('x','y','z'),(2000,400,400)):
            merged_df.loc[np.abs(merged_df[f'V_{comp}_GSM'])>limit,f'V_{comp}_GSM'] = np.nan

        merged_df.loc[merged_df['N_tot']>500,'N_tot'] = np.nan
        merged_df.loc[merged_df['P_th']>100,'P_th'] = np.nan

        ###----------CALCULATIONS----------###

        # Clock Angle: theta = atan2(By,Bz)
        merged_df['B_clock'] = np.arctan2(merged_df[f'B_y_{vec_coords}'], merged_df[f'B_z_{vec_coords}'])

        # Kan and Lee Electric Field: E_R = |V| * B_T * sin^2 (clock/2)
        merged_df['E_R'] = (merged_df['V_mag'] * np.sqrt(merged_df[f'B_y_{vec_coords}']**2+merged_df[f'B_z_{vec_coords}']**2) * (np.sin(merged_df['B_clock']/2))**2) * 1e-3
        # V *= 1e3, B *= 1e-9, E *= 1e3, so E_R *= 1e-3

        ###----------PRESSURES----------###

        try: # find ratio of alpha and proton densities
            omni_df = import_processed_data(omni_dir,year=year)
            merged_df = pd.merge_asof(
                merged_df.sort_index(),
                omni_df[['na_np_ratio']].sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'   # take the closest value on or before the timestamp
            )

            del omni_df
        except:
            merged_df['na_np_ratio'] = 0.05

        # Dynamic pressure
        # P = 0.5 * rho * V^2

        m_avg = (m_p + merged_df['na_np_ratio'] * m_a) / (merged_df['na_np_ratio'] + 1)
        merged_df['P_flow'] = 0.5 * m_avg * merged_df['N_tot']  * merged_df['V_mag']**2 * 1e21
        # N *= 1e6, V *= 1e6, P *= 1e9, so P_flow *= 1e21

        # Beta = p_th / p_mag, p_mag = B^2/2mu_0
        merged_df['beta'] = merged_df['P_th'] / (merged_df['B_avg']**2) * (2*mu_0) * 1e9
        # p_dyn *= 1e-9, B_avg *= 1e18, so beta *= 1e9

        ###----------CROSS PRODUCTS----------###

        # E = -V x B = B x V
        merged_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]] = np.cross(merged_df[[f'B_{comp}_{vec_coords}' for comp in ('x','y','z')]],merged_df[[f'V_{comp}_{vec_coords}' for comp in ('x','y','z')]]) * 1e-3
        # V *= 1e3, B *= 1e-9, and E *= 1e3 so e_gse *= 1e-3
        merged_df['E_mag'] =  np.linalg.norm(merged_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]],axis=1)

        # S = E x H = E x B / mu_0
        merged_df[[f'S_{comp}_{vec_coords}' for comp in ('x','y','z')]] = np.cross(merged_df[[f'E_{comp}_{vec_coords}' for comp in ('x','y','z')]],merged_df[[f'B_{comp}_{vec_coords}' for comp in ('x','y','z')]]) * 1e-6 / mu_0
        # E *= 1e-3, B *= 1e-9, and S *= 1e6 so s_gse *= 1e-6
        merged_df['S_mag'] =  np.linalg.norm(merged_df[[f'S_{comp}_{vec_coords}' for comp in ('x','y','z')]],axis=1)

        ###----------WRITE TO FILE----------###
        if not os.path.exists(combined_dir):
            os.makedirs(combined_dir)

        output_file = os.path.join(combined_dir, f'C1_SPIN_{year}.cdf')

        print(f'{year} processed.')
        write_to_cdf(merged_df, output_file, {'R_E': R_E}, overwrite=True, reset_index=True)


def filter_spin_data(spin_directory, region='msh', year=None):

    data_dir = os.path.join(spin_directory, 'combined', 'raw') # input
    region_dir = os.path.join(spin_directory, region, 'raw')   # output

    # Modes for CIS instrument
    region_modes = {'msh': [8,9,10,11,12,13,14],
                    'sw':  [0,1,2,3,4,5]}
    quality_high = [3,4]   # -2 custom flag for when no quality data available

    if year is not None:
        year_range = (year,)

    else:
        year_range = [int(re.search(r'\d{4}', f).group()) for f in os.listdir(data_dir) if re.search(r'\d{4}', f)]

    for year in year_range:
        try:
            spin_df = import_processed_data(data_dir, year=year)
        except:
            print(f'No data for {year}.')
            continue

        mask = np.ones(len(spin_df),dtype=bool)
        #print(np.sum(mask))
        if 'mode' in spin_df:
            mask &= (np.isin(spin_df.loc[:,'mode'],region_modes.get(region)))
            if np.sum(mask)==0:
                print(f'No {region} data in correct mode for {year}.')
                continue

        if 'quality' in spin_df:
            mask &= (np.isin(spin_df.loc[:,'quality'],quality_high))
            if np.sum(mask)==0:
                print(f'No {region} data of good quality for {year}.')
                continue

        spin_df = spin_df.loc[mask]
        if spin_df.empty:
            print(f'No {region} data for {year}.')
            continue

        ###----------WRITE TO FILE----------###
        if not os.path.exists(region_dir):
            os.makedirs(region_dir)

        output_file = os.path.join(region_dir, f'C1_SPIN_{region}_{year}.cdf')
        write_to_cdf(spin_df, output_file, {'R_E': R_E}, True, reset_index=True)

        print(f'{year} processed.')

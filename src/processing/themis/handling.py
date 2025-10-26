import os
import glob
import re

import numpy as np
import pandas as pd
import itertools as it
from scipy.constants import k as kB, e, mu_0, m_p, physical_constants
m_a = physical_constants['alpha particle mass'][0]

from spacepy import pycdf
import spacepy.pycdf.istp

from ..writing import write_to_cdf
from ..dataframes import add_df_units, resample_data
from ..handling import create_log_file, log_missing_file
from ..utils import create_directory
from ..reading import import_processed_data

from ...coordinates.magnetic import calc_B_GSM_angles
from ...config import R_E, PROC_OMNI_DIR_5MIN


def process_themis_files(spacecraft, themis_directories, proc_directory, fgm_labels, pos_labels, data_resolution='1/128s', sample_intervals='1min', time_col='epoch', year=None, sub_folders=False, overwrite=True, priority_suffices=('fgh','fgl','fge','fgs')):
    """
    The raw data uses fgs as the priority.
    For the 1min and 5min, the fgh is the priority.
    """
    spacecraft_dir = themis_directories[spacecraft]
    fgm_directory = f'{spacecraft_dir}/FGM'
    pos_directory = f'{spacecraft_dir}/STATE'

    out_directory = os.path.join(proc_directory[spacecraft],'FGM')
    create_directory(out_directory)

    fgm_directory_name = os.path.basename(os.path.normpath(fgm_directory))
    pos_directory_name = os.path.basename(os.path.normpath(pos_directory))

    fgm_log_file_path = os.path.join(out_directory, f'{fgm_directory_name}_files_not_added.txt')  # Stores not loaded files
    pos_log_file_path = os.path.join(out_directory, f'{pos_directory_name}_files_not_added.txt')  # Stores not loaded files

    create_log_file(fgm_log_file_path)
    create_log_file(pos_log_file_path)

    if sample_intervals=='none':
        sample_intervals='raw'
    if isinstance(sample_intervals,str):
        sample_intervals = (sample_intervals,)

    for sample_interval in sample_intervals:
        samp_dir = os.path.join(out_directory, sample_interval)
        create_directory(samp_dir)

    pos_variables = pos_labels[spacecraft]

    print(f'Processing THEMIS: {spacecraft}.')
    fgm_files_by_year = get_themis_files(fgm_directory, year, sub_folders)
    pos_files_by_year = get_themis_files(pos_directory, year, sub_folders)

    # Process each year's files
    for (fgm_year, fgm_files), (pos_year, pos_files) in zip(fgm_files_by_year.items(), pos_files_by_year.items()):
        print(f'Processing {fgm_year} data.')

        ###---------------STATE DATA------------###

        pos_yearly_list = []

        # Loop through each daily file in the year
        for pos_file in pos_files:

            try:
                # position data at every minute
                pos_dict, _ = extract_themis_data(pos_file, pos_variables)
                pos_df = pd.DataFrame(pos_dict)
                pos_yearly_list.append(pos_df)
            except:
                #print(f'{pos_file} empty.')
                log_missing_file(pos_log_file_path, pos_file)
                continue

        ###---------------FGM DATA------------###

        fgm_yearly_list = []

        for fgm_file in fgm_files:

            success = False
            for suffix in priority_suffices:
                fgm_variables = fgm_labels[spacecraft][suffix]

                try:
                    fgm_dict, _ = extract_themis_data(fgm_file, fgm_variables)
                    if not fgm_dict:  # Check if the dictionary is empty
                        raise ValueError(f'{fgm_file} empty.')

                    fgm_df = pd.DataFrame(fgm_dict)
                    valid_b = ~fgm_df['B_avg'].isna()
                    if np.sum(valid_b)==0:  # Check if valid B data
                        raise ValueError(f'{fgm_file} empty.')

                    # GSM angles
                    gsm = calc_B_GSM_angles(fgm_df, time_col=time_col)
                    fgm_df = pd.concat([fgm_df, gsm], axis=1)
                    fgm_yearly_list.append(fgm_df)

                    success = True
                    break  # exit suffix loop if successful

                except (ValueError, FileNotFoundError, pd.errors.EmptyDataError):
                    # Continue trying next suffix if specific expected errors
                    continue

                except Exception as e:
                    log_missing_file(fgm_log_file_path, fgm_file, e)
                    success = True  # treat this as handled and exit suffix loop
                    break

            if not success:
                # If none of the suffixes worked, log that the file was skipped
                log_missing_file(fgm_log_file_path, fgm_file, 'No valid suffix data found')

        ###---------------COMBINING------------###
        if not (fgm_yearly_list and pos_yearly_list):
            print(f'No data for {fgm_year}')
            continue

        # Position data
        pos_yearly_df = pd.concat(pos_yearly_list, ignore_index=True)
        add_df_units(pos_yearly_df)

        # Field data
        fgm_yearly_df = pd.concat(fgm_yearly_list, ignore_index=True)
        add_df_units(fgm_yearly_df)

        del pos_yearly_list
        del fgm_yearly_list
        del gsm
        del fgm_df
        del fgm_dict
        del pos_df

        # Resampling has to be done after as THEMIS files overlap on times
        for sample_interval in sample_intervals:
            samp_dir = os.path.join(out_directory, sample_interval)

            if sample_interval=='raw':
                merged_df = pd.merge(pos_yearly_df, fgm_yearly_df, left_on=time_col, right_on=time_col, how='outer')
            else:
                sampled_pos_df = resample_data(pos_yearly_df, time_col, sample_interval)
                sampled_fgm_df = resample_data(fgm_yearly_df, time_col, sample_interval)

                # Merge
                merged_df = pd.merge(sampled_pos_df, sampled_fgm_df, left_on=time_col, right_on=time_col, how='outer')
                # Clear up memory
                del sampled_pos_df
                del sampled_fgm_df

            add_df_units(merged_df)

            print(f'{sample_interval} reprocessed.')

            output_file = os.path.join(samp_dir, f'{fgm_directory_name}_{fgm_year}.cdf')
            attributes  = {'sample_interval': data_resolution, 'time_col': time_col, 'R_E': R_E}
            write_to_cdf(merged_df, output_file, attributes, overwrite)

            del merged_df

        print(f'{fgm_year} processed.')

def process_themis_plasma(spacecraft, themis_directories, proc_directories, plasma_labels, sample_intervals='raw', time_col='epoch', year=None, sub_folders=False, overwrite=True):

    spacecraft_dir = themis_directories[spacecraft]
    plasma_dir = f'{spacecraft_dir}/MOM'

    out_directory = os.path.join(proc_directories[spacecraft],'MOM')
    create_directory(out_directory)

    directory_name = os.path.basename(os.path.normpath(plasma_dir))
    log_file_path = os.path.join(out_directory, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
    create_log_file(log_file_path)

    if sample_intervals=='none':
        sample_intervals='raw'
    if isinstance(sample_intervals,str):
        sample_intervals = (sample_intervals,)

    for sample_interval in sample_intervals:
        samp_dir = os.path.join(out_directory, sample_interval)
        create_directory(samp_dir)

    plasma_variables = plasma_labels[spacecraft]

    print(f'Processing THEMIS: {spacecraft}.')
    files_by_year = get_themis_files(plasma_dir, year, sub_folders)

    # Process each year's files
    for year, files in files_by_year.items():
        print(f'Processing {year} data.')
        yearly_plas_list = []
        yearly_flag_list = []


        for file in files:

            try:
                # flag data is not exact same resolution - need to double check
                plasma_dict, flag_dict = extract_themis_data(file, plasma_variables)
                yearly_plas_list.append(pd.DataFrame(plasma_dict))
                yearly_flag_list.append(pd.DataFrame(flag_dict))

            except ValueError as e:
                raise Exception(e)
            except:
                log_missing_file(log_file_path, file)
                continue

        ###---------------RESAMPLING------------###
        if not (yearly_plas_list and yearly_flag_list):
            print(f'No data for {year}')
            continue


        df_year_plasma = pd.concat(yearly_plas_list, ignore_index=True).sort_values(by=['epoch'])
        df_year_flag = pd.concat(yearly_flag_list, ignore_index=True).sort_values(by=['epoch_flag'])

        df_year = pd.merge_asof(df_year_plasma, df_year_flag, left_on='epoch', right_on='epoch_flag', direction='backward')
        df_year.drop(columns=['epoch_flag'],inplace=True)
        add_df_units(df_year)

        # Resampling has to be done after as THEMIS files overlap on times
        for sample_interval in sample_intervals:
            samp_dir = os.path.join(out_directory, sample_interval)

            if sample_interval=='raw':
                df_sampled = df_year

            else:
                df_sampled = resample_data(df_year, time_col, sample_interval)

            print(f'{sample_interval} reprocessed.')

            output_file = os.path.join(samp_dir, f'{directory_name}_{year}.cdf')
            attributes = {'sample_interval': 'spin', 'time_col': time_col, 'R_E': R_E}
            write_to_cdf(df_sampled, output_file, attributes, overwrite)

            print(f'{year} processed.')


def get_themis_files(directory=None, year=None, sub_folders=False):

    files_by_year = {}

    if directory is None:
        directory = os.getcwd()

    if sub_folders:
        # Loop through subdirectories named as years
        for sub_folder in sorted(os.listdir(directory)):
            if sub_folder=='possibly_corrupted_files':
                continue
            sub_folder_path = os.path.join(directory, sub_folder)
            if year and sub_folder != str(year):
                continue  # Skip folders not matching the specified year
            elif not os.path.isdir(sub_folder_path):
                continue

            pattern = os.path.join(sub_folder_path, '*.cdf')  # Match all .cdf files in the folder
            files = sorted(glob.glob(pattern))
            latest_files = select_latest_versions(files)
            files_by_year[sub_folder] = latest_files
    else:

        # Build the search pattern based on the presence of a specific year
        if year:
            pattern = os.path.join(directory, f'*_{year}*.cdf')  # Match only files with the given year
        else:
            pattern = os.path.join(directory, '*.cdf')  # Match all .cdf files if no year is specified

        # Use glob to find files matching the pattern
        files = sorted(glob.glob(pattern))
        latest_files = select_latest_versions(files)
        for file in latest_files:
            # Extract the year from the filename assuming format data-name__YYYYMMDD_version.cdf
            file_year = os.path.basename(file).split('_')[3][:4]
            files_by_year.setdefault(file_year, []).append(file)

    if year and str(year) not in files_by_year:
        raise ValueError(f'No files found for {year}.')
    if not files_by_year:
        raise ValueError('No files found.')

    return files_by_year


def extract_themis_data(cdf_file, variables):

    info_dict = {}
    flag_dict = {}

    with pycdf.CDF(cdf_file) as cdf:

        for var_name, var_code in variables.items():

            if var_code not in cdf:
                continue

            data = cdf[var_code].copy()
            if len(data)==0:
                continue
            spacepy.pycdf.istp.nanfill(data) # Replaces fill values with NANs
            data = data[...]

            if 'flag' in var_name:
                data_dict = flag_dict
            else:
                data_dict = info_dict

            if var_name == 'T_vec': # special case

                # Field aligned temperature; scalar is mean of components
                temp = (data[:,0] + data[:,1] + data[:,2]) / 3
                temp *= (e / kB)  # (eV → J) / (J/K) = K
                data_dict['T_ion'] = temp

            elif data.ndim == 2 and data.shape[1] == 3:  # Assuming a 2D array for vector components

                system = 'GSE'
                if var_name == 'r':
                    data /= R_E  # Scales distances to multiples of Earth radii

                elif '_GSE' in var_name or '_GSM' in var_name:
                    if '_GSM' in var_name:
                        system = 'GSM'

                    var_name = var_name.split('_')[0]

                data_dict[f'{var_name}_x_{system}'] = data[:, 0]
                data_dict[f'{var_name}_y_{system}'] = data[:, 1]
                data_dict[f'{var_name}_z_{system}'] = data[:, 2]

            else:
                if 'time' in var_name:
                    data = pd.to_datetime(data, unit='s', origin='unix')
                    var_name = var_name.replace('time','epoch') # consistency with Cluster

                elif var_name.startswith('P_'): # eV/cc
                    data *= (e * 1e6 * 1e9) # eV -> J -> /m3 -> nPa

                data_dict[var_name] = data  # pycdf extracts as datetime, no conversion from epoch needed

    return info_dict, flag_dict

def select_latest_versions(files):
    file_map = {}

    for file in files:
        # Extract the date and version from the filename (e.g., "tha_l1_state_20160506_v01.cdf")
        filename = os.path.basename(file)
        match = re.search(r'_(\d{8})_v(\d{2})\.cdf$', filename)
        if not match:
            continue  # Skip files that don't match the expected pattern

        date = match.group(1)
        version = int(match.group(2))

        # Keep only the latest version for each date
        if date not in file_map or version > file_map[date][1]:
            file_map[date] = (file, version)

    # Extract only the file paths
    return [file_version[0] for file_version in file_map.values()]

# %%

def combine_spin_data(spacecraft, proc_directories, year=None, omni_dir=PROC_OMNI_DIR_5MIN):

    spacecraft_dir = proc_directories[spacecraft]

    field_dir = os.path.join(spacecraft_dir, 'FGM', 'raw')
    plasma_dir = os.path.join(spacecraft_dir, 'MOM', 'raw')
    combined_dir = os.path.join(spacecraft_dir, 'combined', 'raw')

    if year is not None:
        year_range = (year,)

    else:

        field_years = [int(re.search(r'\d{4}', f).group()) for f in os.listdir(field_dir) if re.search(r'\d{4}', f)]
        plasma_years = [int(re.search(r'\d{4}', f).group()) for f in os.listdir(plasma_dir) if re.search(r'\d{4}', f)]

        year_range = list(range(max(plasma_years[0],field_years[0]),min(plasma_years[-1],field_years[-1])+1))

    next_year_field = pd.DataFrame()
    next_year_plasma = pd.DataFrame()

    for y_i, year in enumerate(year_range):
        next_year = year + 1

        print(f'Processing {year} data.')

        field_df  = import_processed_data(field_dir, year=year)
        plasma_df = import_processed_data(plasma_dir, year=year)

        if not next_year_field.empty:
            field_df = pd.concat([field_df,next_year_field])
            field_df.sort_index(inplace=True)

            next_year_field = pd.DataFrame()

        if not next_year_plasma.empty:
            plasma_df = pd.concat([plasma_df,next_year_plasma])
            plasma_df.sort_index(inplace=True)

            next_year_plasma = pd.DataFrame()

        if y_i != (len(year_range)-1):
            next_year_field = field_df.loc[field_df.index.year==next_year]
            field_df = field_df.loc[field_df.index.year!=next_year]

            next_year_plasma = plasma_df.loc[plasma_df.index.year==next_year]
            plasma_df = plasma_df.loc[plasma_df.index.year!=next_year]

        field_df = field_df[~field_df.index.duplicated(keep='first')]
        pos_df = field_df[[f'r_{comp}_GSE' for comp in ('x','y','z')]]
        field_df.drop(columns=[f'r_{comp}_GSE' for comp in ('x','y','z')],inplace=True)

        merged_df = pd.merge_asof(
            plasma_df.reset_index(),
            field_df.reset_index(),
            on='epoch',
            direction='nearest',
            tolerance=pd.Timedelta('3s')  # adjust as needed
        )

        merged_df = merged_df.set_index('epoch')

        del field_df
        del plasma_df

        vec_coords = 'GSM'

        # For now treating all species together
        merged_df.rename(columns={'P_ion': 'P_th', 'T_ion': 'T_tot', 'N_ion': 'N_tot'}, inplace=True)

        # Mistake made in original processing
        merged_df.rename(columns={'V_GSE_x_GSE': 'V_x_GSE', 'V_GSE_y_GSE': 'V_y_GSE', 'V_GSE_z_GSE': 'V_z_GSE',
                          'V_GSM_x_GSE': 'V_x_GSM', 'V_GSM_y_GSE': 'V_y_GSM', 'V_GSM_z_GSE': 'V_z_GSM'}, inplace=True)

        # Using magnitude as flow for now
        merged_df['V_flow'] = np.linalg.norm(merged_df[[f'V_{comp}_GSE' for comp in ('x','y','z')]], axis=1)

        print('Calculations...')
        ###----------CALCULATIONS----------###

        # Clock Angle: theta = atan2(By,Bz)
        merged_df['B_clock'] = np.arctan2(merged_df[f'B_y_{vec_coords}'], merged_df[f'B_z_{vec_coords}'])

        # Kan and Lee Electric Field: E_R = |V| * B_T * sin^2 (clock/2)
        merged_df['E_R'] = (merged_df['V_flow'] * np.sqrt(merged_df[f'B_y_{vec_coords}']**2+merged_df[f'B_z_{vec_coords}']**2) * (np.sin(merged_df['B_clock']/2))**2) * 1e-3
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
        merged_df['P_flow'] = 0.5 * m_avg * merged_df['N_tot']  * merged_df['V_flow']**2 * 1e21
        # N *= 1e6, V *= 1e6, P *= 1e9, so P_flow *= 1e21

        # Beta = p_dyn / p_mag, p_mag = B^2/2mu_0
        merged_df['beta'] = merged_df['P_flow'] / (merged_df['B_avg']**2) * (2*mu_0) * 1e9
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

        output_file = os.path.join(combined_dir, f'{spacecraft.upper()}_SPIN_{year}.cdf')

        merged_df = pd.concat([pos_df, merged_df], axis=1)
        add_df_units(merged_df)

        print(f'{year} processed.')
        write_to_cdf(merged_df, output_file, {'R_E': R_E}, overwrite=True, reset_index=True)


def filter_spin_data(spacecraft, proc_directories, region='msh', year=None):

    """
    The position data is timestamped separately, so keeping only data with a good flag/correct mode removes all this data
    Need to instead remove all the undesired data
    """

    sc_dir = proc_directories[spacecraft]
    data_dir   = os.path.join(sc_dir, 'combined', 'raw') # input
    region_dir = os.path.join(sc_dir, region, 'raw')   # output

    if year is not None:
        year_range = (year,)
    else:
        year_range = [int(re.search(r'\d{4}', f).group()) for f in os.listdir(data_dir) if re.search(r'\d{4}', f)]

    # Quality flags for ESA instrument
    # Could possibly remove 1 or 4
    # Quality = 0 indicates good data (-2 placeholder for no quality provided)
    qualities = (1, 4, 8, 16, 32, 64)

    bad_qualities = []
    for i in range(1, len(qualities)+1):
        for combo in it.combinations(qualities, i):
            bad_qualities.append(sum(combo))
    bad_qualities = sorted(bad_qualities)

    # Solar wind flag =1 in solar wind, =0 if not
    # So want to remove those with the wrong flag
    wrong_flag = 1
    if region=='sw':
        wrong_flag = 0

    for year in year_range:
        try:
            df_year = import_processed_data(data_dir, year=year)
        except:
            print(f'No data for {year}.')
            continue

        mask = np.ones(len(df_year),dtype=bool)
        if 'quality' in df_year:
            mask &= ~df_year['quality'].isin(bad_qualities)
            if np.sum(mask)==0:
                print(f'No {region} data of good quality for {year}.')
                continue

        if 'flag' in df_year: # solar wind flag
            mask &= (df_year['flag'].fillna(-2) != wrong_flag)
            if np.sum(mask)==0:
                print(f'No {region} data in correct mode for {year}.')
                continue

        spin_df = df_year.loc[mask]
        if spin_df.empty:
            print(f'No {region} data for {year}.')
            continue

        ###----------WRITE TO FILE----------###
        if not os.path.exists(region_dir):
            os.makedirs(region_dir)

        output_file = os.path.join(region_dir, f'C1_SPIN_{region}_{year}.cdf')
        print(f'{year} processed.')
        write_to_cdf(spin_df, output_file, {'R_E': R_E}, True, reset_index=True)

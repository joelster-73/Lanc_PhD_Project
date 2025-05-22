# -*- coding: utf-8 -*-
"""
Created on Thu May  8 17:59:45 2025

@author: richarj2
"""

def insert_bs_diff(df, df_bs, sc_key, rel=False):
    diffs = df_bs[f'r_{sc_key}'] - df_bs['r_BS']
    df['r_bs_diff'] = diffs.reindex(df.index)
    if rel:
        diffs_rel = diffs/df_bs['r_BS']
        df['r_bs_diff_rel'] = diffs_rel.reindex(df.index)
    df.attrs['units']['r_bs_diff'] = df_bs.attrs['units'].get('r_BS', None)



def find_best_buffer(df, r_diff_col, y1_col, y2_col, **kwargs):

    buff_min     = kwargs.get('buff_min',0)
    buff_max     = kwargs.get('buff_max',6)
    compression  = kwargs.get('compression',2) # Shield 1969 - wrong
    compressions = kwargs.get('compressions',None)

    buffers = np.linspace(buff_min,buff_max,501)
    num_total = np.empty(len(buffers))
    num_bad = np.empty(len(buffers))
    perc_bad = np.empty(len(buffers))

    df_sw = df[df[r_diff_col]>=0]

    if compressions is not None:
        B_imf, B_msh, _ = load_compression_ratios(compressions)

    for i, buffer in enumerate(buffers):
        df_out = df_sw[df_sw[r_diff_col]>buffer]

        num_total[i] = len(df_out)
        if compressions is None:
            num_bad[i] = np.sum(df_out[y1_col]/df_out[y2_col]>compression)
        else:
            num_bad[i] = np.sum(are_points_above_line(B_imf, B_msh, df_out[y2_col], df_out[y1_col]))

        if num_total[i] > 0:
            perc_bad[i] = num_bad[i]/num_total[i]*100
        else:
            perc_bad[i] = np.nan

    best_perc = np.nanmin(perc_bad)
    where_result = np.where(perc_bad==best_perc)[0]
    ind = int(where_result[0]) if where_result.size > 0 else where_result
    best_buff = buffers[ind]
    best_length = num_total[ind]
    print(f'Buffer: {best_buff:.2f}, {best_perc:.2g}%, {best_length:,}')


def generate_bs_df(cluster_dir, df_omni, out_dir, sc_key, variables,
                   sample_interval='1min', time_col='epoch', overwrite=True, df_sc=None):

    if df_sc is None:
        print('Processing Cluster.')
        processed_files = get_processed_files(cluster_dir)

        directory_name = os.path.basename(os.path.normpath(cluster_dir))
        log_file_path = os.path.join(out_dir, f'{directory_name}_files_not_added.txt')  # Stores not loaded files
        create_log_file(log_file_path)

        full_list = []
        # Process each year's files
        for cdf_file in processed_files:
            print(f'Processing {cdf_file}.')

            try:  # Bad data check
                data_dict = {}
                with pycdf.CDF(cdf_file) as cdf:

                    for var_name, var_code in variables.items():

                        data_dict[var_name] = cdf[var_code][...]

                yearly_df = pd.DataFrame(data_dict)
                add_df_units(yearly_df)
                full_list.append(yearly_df)

                # Samples from 5VPS to 1min
                #print('Resampled.')
                #df_sampled = resample_cluster_data(yearly_df, time_col, sample_interval)
                #full_list.append(df_sampled)

            except (AttributeError, ValueError, RuntimeError) as e:
                print('Known error.')
                log_missing_file(log_file_path, cdf_file, e)
            except Exception as e:
                print('Unknown error.')
                log_missing_file(log_file_path, cdf_file, e)

        df_sc = pd.concat(full_list, ignore_index=True)
        add_df_units(df_sc)
        set_df_indices(df_sc, time_col)

    #print(df_sc)

    # Merges data with OMNI
    # For now only need v_x_GSE
    c1_omni = merge_dataframes(df_sc, df_omni, sc_key, 'OMNI', clean=False)

    df_bs = calc_bs_pos(c1_omni, sc_key=sc_key, time_col='index')
    add_df_units(df_bs)

    #print(df_bs)

    # Write to file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_file = os.path.join(out_dir, 'C1_BS_positions.cdf')
    R_E = 6370 # Used by Cluster
    attributes = {'sample_interval': sample_interval, 'time_col': time_col, 'R_E_km': R_E}
    write_to_cdf(df_bs, output_file, attributes, overwrite=True)

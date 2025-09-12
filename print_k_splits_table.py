# Read list of filenames from given file
# For each, load the associated CSV dataframe
# Print out the LaTeX table for each

import pandas as pd
from tabulate import tabulate

def process_filename_list(ks_and_file_lists):
    output_ks_df_rows = []

    all_kf_and_files = open(ks_and_file_lists)
    for kf_str_n in all_kf_and_files:
        kf_str = kf_str_n.rstrip()
        print(f"kf_str={kf_str}")
        k,f = kf_str.split(",")
        df = pd.read_csv(f)
        r2_mean = df["r2_score_mean"]
        r2_stddev = df["r2_score_stddev"]
        output_ks_df_rows.append({"k":k, "r2_mean" : r2_mean, "r2_stddev" : r2_stddev})

    output_ks_df = pd.DataFrame(output_ks_df_rows)
    print(output_ks_df)
    #print(tabulate(output_ks_df, headers="keys"))
    return output_ks_df

process_filename_list("k_file_list")
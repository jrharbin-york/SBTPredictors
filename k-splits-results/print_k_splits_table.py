# Read list of filenames from given file
# For each, load the associated CSV dataframe
# Print out the LaTeX table for each
from functools import reduce

import pandas as pd
from tabulate import tabulate

def generate_dataframe_for_ks(all_files_filename, k):
    output_ks_df_rows = []

    all_files = open(all_files_filename)
    for name_n in all_files:
        filename = name_n.rstrip()
        # ignore results from old moved dirs
        if not "old" in filename:
            filename_after_regression = filename.split("regression-")[1]
            filename_without_k = filename_after_regression.split("k-")[0]
            df = pd.read_csv(filename)
            r2_mean = float(df["r2_score_mean"].iloc[0])
            r2_stddev = float(df["r2_score_stddev"].iloc[0])
            content_str = f"{r2_mean:.3f} ({r2_stddev:.3f})"
            header_str = f"k = {k} mean (stddev.)"
            #output_ks_df_rows.append({"file":filename, "k":k, "r2_mean" : r2_mean, "r2_stddev" : r2_stddev, header_str : content_str})
            output_ks_df_rows.append({"file_spec": filename_without_k, header_str: content_str})

    output_ks_df = pd.DataFrame(output_ks_df_rows)

    print(tabulate(output_ks_df, headers="keys"))
    return output_ks_df

def process_filename_list(k4_file_list, k5_file_list, k6_file_list, k7_file_list, tex_filename_out):
    df_4 = generate_dataframe_for_ks(k4_file_list, 4)
    df_5 = generate_dataframe_for_ks(k5_file_list, 5)
    df_6 = generate_dataframe_for_ks(k6_file_list, 6)
    df_7 = generate_dataframe_for_ks(k7_file_list, 7)
    all_dfs = [df_4, df_5, df_6, df_7]
    # Merge them all on the file_spec
    df_combined = reduce(lambda l, r: pd.merge(l, r, on="file_spec"), all_dfs)
    print(tabulate(df_combined, headers="keys"))

    latex_table = df_combined.to_latex(index=False)  # index=False removes row indices

    with open(tex_filename_out, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX summary results saved to {tex_filename_out}")

    return df_combined

filename_out = "k_latex_table.tex"
process_filename_list("k4_file_list", "k5_file_list", "k6_file_list", "k7_file_list", filename_out)
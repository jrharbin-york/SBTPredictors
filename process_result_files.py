import pandas as pd
import structlog
import tabulate

log = structlog.get_logger()

# Read multiple files for individual runs under each directory

#e.g. in each result directory

#-rw-r--r-- 1 jharbin users 33119 Jul 15 03:42 regression-Multiturtlebot-TB1TSFreshWin_GradBoost-res.csv
#-rw-r--r-- 1 jharbin users 26613 Jul 15 06:41 regression-Multiturtlebot-TB1_MiniRocket_GradBoost-res.csv
#-rw-r--r-- 1 jharbin users 19108 Jul 15 06:48 regression-Multiturtlebot-TB1_MiniRocket_Ridge-res.csv
#-rw-r--r-- 1 jharbin users 31898 Jul 15 07:56 regression-Multiturtlebot-TB1_TSForest-res.csv
#-rw-r--r-- 1 jharbin users 23924 Jul 15 04:07 regression-Multiturtlebot-TB1_TSFreshWin_Ridge-res.csv

# need to add alg_name + metric + "_"
file_names_in_result_dirs = ['TSFreshWin_GradBoost',
                             'MiniRocket_GradBoost',
                             'MiniRocket_Ridge',
                             'TSForest',
                             'TSFreshWin_Ridge']

def merge_results(use_case, metric, result_dirs):
    # Aggregate dataframe
    # For each compatible param choice:
    #
    # merge all of into one for each compatible param choice
    # compute mean and stddev of each of the mse and rmse
    # Compute the mean and stddev of the time_diff
    #stats_results = pd.DataFrame(columns=["alg", "param1", "param2", "r2_score_mean", "mse_mean", "rmse_mean", "r2_score_stddev", "mse_score_stddev", "rmse_score_stddev", "time_mean", "time_stddev"])
    results_all_algs = []

    for alg in file_names_in_result_dirs:
        dataframes_for_alg = []
        for d in result_dirs:
            # compute mean and stddev of the r2_score, mse and rmse
            alg_res_filename = f"regression-{use_case}-{metric}_{alg}-res.csv"
            log.debug(f"Loading alg_res_filename={alg_res_filename}...")
            full_path = d + "/" + alg_res_filename
            individual_results = pd.read_csv(full_path)
            log.debug(f"alg_res_filename={alg_res_filename} loaded")

            dataframes_for_alg.append(individual_results)
            combined_dfs = pd.concat(dataframes_for_alg, ignore_index=True)
            # the results need to be aggregated across params, e.g. for param1 and param2 - use group by
            results_one_alg = combined_dfs.groupby(["param1", "param2"])[['r2_score', 'mse', 'rmse']].agg(['mean', 'std'])
            results_one_alg["alg"] = alg
            print(tabulate.tabulate(results_one_alg, headers="keys"))
            results_all_algs.append(results_one_alg)

    results_all_algs_df = pd.concat(results_all_algs)
    results_all_algs_df.columns = results_all_algs_df.columns.map('_'.join).str.strip('_')
    print(tabulate.tabulate(results_all_algs_df, headers="keys"))
    sorted_res_all_algs = results_all_algs_df.sort_values(by='r2_score_mean', ascending=False, axis="index")
    resorted_cols = ['alg', 'r2_score_mean', 'r2_score_std', 'mse_mean', 'mse_std', 'rmse_mean','rmse_std' ]

    sorted_res_new_cols = sorted_res_all_algs[resorted_cols]
    sorted_res_new_cols = sorted_res_new_cols.round(2)
    sorted_res_new_cols = sorted_res_new_cols.reset_index(level=[0, 1])

    print(tabulate.tabulate(sorted_res_new_cols, headers="keys"))
    top_results = sorted_res_new_cols.head(10)
    latex = top_results.to_latex(index=False,float_format="{:0.2f}".format, escape=True)
    print(latex)

result_dirs_eterry = ["/home/jharbin/academic/soprano/SBTPredictors/for-aggregation-results/eterry/yesod_2025_07_14_20_26_54"]
merge_results("ETERRY", "Human1Dist", result_dirs_eterry)
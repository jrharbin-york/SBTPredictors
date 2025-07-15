import pandas as pd
# Read multiple files for individual runs under each directory

#e.g. in each result directory

#-rw-r--r-- 1 jharbin users 33119 Jul 15 03:42 regression-Multiturtlebot-TB1TSFreshWin_GradBoost-res.csv
#-rw-r--r-- 1 jharbin users 26613 Jul 15 06:41 regression-Multiturtlebot-TB1_MiniRocket_GradBoost-res.csv
#-rw-r--r-- 1 jharbin users 19108 Jul 15 06:48 regression-Multiturtlebot-TB1_MiniRocket_Ridge-res.csv
#-rw-r--r-- 1 jharbin users 31898 Jul 15 07:56 regression-Multiturtlebot-TB1_TSForest-res.csv
#-rw-r--r-- 1 jharbin users 23924 Jul 15 04:07 regression-Multiturtlebot-TB1_TSFreshWin_Ridge-res.csv

file_names_in_result_dirs = { 'TSFreshWin-GradBoost' : 'regression-Multiturtlebot-TB1TSFreshWin_GradBoost-res.csv',
                              'MiniRocket-GradBoost' : 'regression-Multiturtlebot-TB1_MiniRocket_GradBoost-res.csv',
                              'MiniRocket-Ridge' : 'regression-Multiturtlebot-TB1_MiniRocket_Ridge-res.csv',
                              'TSForest' : 'regression-Multiturtlebot-TB1_TSForest-res.csv',
                              'TSFreshWin-Ridge' : 'regression-Multiturtlebot-TB1_TSFreshWin_Ridge-res.csv' }

def merge_results(result_dirs):
    # Aggregate dataframe
    # For each compatible param choice:
    #
    # merge all of into one for each compatible param choice
    # compute mean and stddev of each of the mse and rmse
    # Compute the mean and stddev of the time_diff
    stats_results = pd.DataFrame(columns=["alg", "param1", "param2", "r2_score_mean", "mse_mean", "rmse_mean", "r2_score_stddev", "mse_score_stddev", "rmse_score_stddev", "time_mean", "time_stddev"])
    stats_results_rows = []

    for alg, alg_res_filename in file_names_in_result_dirs:
        dataframes_for_alg = []
        for d in result_dirs:
            # compute mean and stddev of the r2_score, mse and rmse
            full_path = d + "/" + alg_res_filename
            individual_results = pd.read_csv(full_path)

            dataframes_for_alg.append(individual_results)
            combined_dfs = pd.concat(dataframes_for_alg, ignore_index=True)
            # the results need to be aggergated across params, e.g. for param1 and param2 - use group by
            results = combined_dfs.groupby(["param1", "param2"])['r2_score', 'mse', 'rmse'].agg(['mean', 'std'])
            results.to_csv()


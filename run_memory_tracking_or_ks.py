import sys

import structlog
from numba.mviewbuf import memoryview_get_extents

import run_all_algs
import datasets
import argparse

log = structlog.get_logger()

alg_choice = sys.argv[1]
use_case_choice_metric = sys.argv[2]
memory_filename_base = sys.argv[3]

if memory_filename_base == "NONE":
    memory_filename_base = None

k_min = int(sys.argv[4])
k_max = int(sys.argv[5])

if memory_filename_base is None:
    memory_filename = None
else:
    memory_filename = memory_filename_base + "/" + use_case_choice_metric + "/" + alg_choice

print(f"Alg choice: {alg_choice}, use case choice {use_case_choice_metric}, memory_filename={memory_filename}")

params_choice = {
    "Mycobot" : {
            "TSFreshWin_GradBoost" : ([150],[4]),
            "TSFreshWin_Ridge" : ([10],[1]),
            "TSForest" : ([150],[0.5]),
            "MiniRocket_GradBoost": ([500], [150]),
            "MiniRocket_Ridge": ([500], [20.0]),
    },
    "Turtlebot_TB1": {
            "TSFreshWin_GradBoost": ([300], [0.5]),
            "TSFreshWin_Ridge": ([20], [0.5]),
            "TSForest": ([150], [1.0]),
            "MiniRocket_GradBoost": ([500], [150]),
            "MiniRocket_Ridge": ([500], [20.0]),
    },
    "Turtlebot_TB2": {
            "TSFreshWin_GradBoost": ([150], [0.5]),
            "TSFreshWin_Ridge": ([20], [0.5]),
            "TSForest": ([150], [10.0]),
            "MiniRocket_GradBoost": ([2000], [300]),
            "MiniRocket_Ridge": ([500], [20.0]),
    },
    "ETerry_Human1": {
            "TSFreshWin_GradBoost": ([50], [0.5]),
            "TSFreshWin_Ridge": ([5], [2.0]),
            "TSForest": ([300], [10.0]),
            "MiniRocket_GradBoost": ([1000], [50]),
            "MiniRocket_Ridge": ([500], [20.0]),
    },
    "ETerry_StaticHumans": {
            "TSFreshWin_GradBoost": ([150], [0.5]),
            "TSFreshWin_Ridge": ([20], [0.5]),
            "TSForest": ([300], [10.0]),
            "MiniRocket_GradBoost": ([2000], [150]),
            "MiniRocket_Ridge": ([1000], [10.0]),
    },
    "ETerry_PathCompletion": {
            "TSFreshWin_GradBoost": ([150], [0.5]),
            "TSFreshWin_Ridge": ([20], [0.5]),
            "TSForest": ([50], [1.0]),
            "MiniRocket_GradBoost": ([2000], [50]),
            "MiniRocket_Ridge": ([500], [20.0]),
    }
}

run_eterry = False
run_mycobot = False
run_turtlebot = False

alg_params1, alg_params2 = params_choice[use_case_choice_metric][alg_choice]

k_default_memtracking = 5

if k_min is None or k_max is None:
    k_range = None
else:
    k_range = range(k_min, k_max+1)


if __name__ == '__main__':
    if use_case_choice_metric == "Mycobot":
        if k_range is None:
            run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_mycobot_fourjoints_3000, memory_filename,
                                                         alg_choice, k_default_memtracking, alg_params1, alg_params2)
        else:
            for k in k_range:
                run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_mycobot_fourjoints_3000, memory_filename,
                                                         alg_choice, k, alg_params1, alg_params2)
    if use_case_choice_metric == "Turtlebot_TB1":
        if k_range is None:
            run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_turtlebot_multi_tb1_server_allops, memory_filename,
                                                         alg_choice, k_default_memtracking, alg_params1, alg_params2)
        else:
            for k in k_range:
                run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_turtlebot_multi_tb1_server_allops, memory_filename,
                                                         alg_choice, k, alg_params1, alg_params2)

    if use_case_choice_metric == "Turtlebot_TB2":
        if k_range is None:
            run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_turtlebot_multi_tb2_server_allops, memory_filename,
                                                         alg_choice, k_default_memtracking, alg_params1, alg_params2)
        else:
            for k in k_range:
                run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_turtlebot_multi_tb2_server_allops, memory_filename,
                                                         alg_choice, k, alg_params1, alg_params2)

    if use_case_choice_metric == "ETerry_Human1":
        if k_range is None:
            run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_eterry_human1_15files, memory_filename,
                                                         alg_choice, k_default_memtracking, alg_params1, alg_params2)
        else:
            for k in k_range:
                run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_eterry_human1_15files, memory_filename,
                                                         alg_choice, k, alg_params1, alg_params2)

    if use_case_choice_metric == "ETerry_StaticHumans":
        if k_range is None:
            run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_eterry_statichumans_15files, memory_filename,
                                                         alg_choice, k_default_memtracking, alg_params1, alg_params2)
        else:
            for k in k_range:
                run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_eterry_statichumans_15files, memory_filename,
                                                         alg_choice, k, alg_params1, alg_params2)

    if use_case_choice_metric == "ETerry_PathCompletion":
        if k_range is None:
            run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_eterry_pathcompletion_15files, memory_filename,
                                                         alg_choice, k_default_memtracking, alg_params1, alg_params2)
        else:
            for k in k_range:
                run_all_algs.run_filprofiler_memory_tracking(datasets.expt_config_eterry_pathcompletion_15files, memory_filename,
                                                         alg_choice, k, alg_params1, alg_params2)











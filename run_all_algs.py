import data_loader
import predictor
import settings
from tabulate import tabulate
import numpy as np

from memory_tracking import FilProfilerMemoryTracker

################################################################################
combined_results_all_tests = None
interval_results = None

################################################################################
## Fixed window size variants - for Mycobot
################################################################################

data_samples_per_second = 10.0
fixed_window_size = 30.0

def run_tsfreshwin_gradboost_fixed_window_size(name_base, results_tag, expt_config, alg_params1=settings.n_estimator_choices, alg_params2=settings.min_samples_split):
    global combined_results_all_algs
    alg_name = "TSFreshWin_GradBoost"
    combined_name = results_tag + alg_name
    combined_results_all_algs = predictor.run_test(name_base, expt_config,
                                                    combined_results_all_algs,
                                                    combined_name,
                                                    True,
                                                    lambda n_estimators, min_samples_split, params:
                                                    predictor.create_tsfresh_windowed_regression_gbr(params, n_estimators, fixed_window_size, data_samples_per_second, max_depth=4, learning_rate=0.1, min_samples_split=min_samples_split, min_samples_leaf=settings.min_samples_leaf_default),
                                                    alg_params1=alg_params1,
                                                    alg_params2=alg_params2,
                                                    param1_name = "n_estimators",
                                                    param2_name = "min_samples_split")
                                                    

def run_tsfreshwin_ridge_fixed_window_size(name_base, results_tag, expt_config, alg_params1=settings.max_alphas, alg_params2=settings.alpha_step_values):
    global combined_results_all_algs
    alg_name = "TSFreshWin_Ridge"
    combined_name = results_tag + "_" + alg_name
    combined_results_all_algs = predictor.run_test(name_base, expt_config,
                                                    combined_results_all_algs,
                                                    combined_name,
                                                    True,
                                                    lambda max_alpha, alpha_step, params:
                                                    predictor.create_tsfresh_windowed_regression_ridge(params, fixed_window_size, data_samples_per_second, max_alpha=max_alpha, alpha_step=alpha_step), 
                                                    alg_params1=alg_params1,
                                                    alg_params2=alg_params2,
                                                    param1_name = "max_alpha",
                                                    param2_name = "alpha_step")

def run_tsforest_fixed_window_size(name_base, results_tag, expt_config):
    global combined_results_all_algs
    alg_name = "TSForest"
    combined_name = results_tag + "_" + alg_name
    combined_results_all_algs = predictor.run_test(name_base, expt_config,
                                                   combined_results_all_algs,
                                                   combined_name,
                                                   True,
                                                   lambda n_estimators, window_size_fixed, params:
                                                   predictor.create_tsf_regression(n_estimators=n_estimators, min_interval=int(np.floor(data_samples_per_second * window_size_fixed))), 
                                                   alg_params1=settings.n_estimator_choices,
                                                   # min_interval is always constant for mycobot
                                                   alg_params2=[0.3],
                                                   param1_name = "n_estimators",
                                                   param2_name = "window_size")
    
################################################################################
# Variable window size variants
################################################################################

def run_tsfreshwin_gradboost(name_base, results_tag, expt_config, alg_params1=settings.n_estimator_choices, alg_params2=settings.window_size_choices_secs):
    global combined_results_all_algs
    alg_name = "TSFreshWin_GradBoost"
    combined_name = results_tag + alg_name
    combined_results_all_algs = predictor.run_test(name_base, expt_config,
                                                    combined_results_all_algs,
                                                    combined_name,
                                                    True,
                                                    lambda n_estimators, window_size, params:
                                                    predictor.create_tsfresh_windowed_regression_gbr(params, n_estimators, window_size, data_samples_per_second, max_depth=4, learning_rate=0.1, min_samples_split=settings.min_samples_split_default, min_samples_leaf=settings.min_samples_leaf_default),
                                                    alg_params1=alg_params1,
                                                    alg_params2=alg_params2,
                                                    param1_name = "n_estimators",
                                                    param2_name = "window_size")

def run_tsfreshwin_gradboost_intervals(name_base, results_tag, expt_config):
    alg_name = "TSFreshWin_GradBoost"
    combined_name = results_tag + alg_name
    combined_results_all_algs = predictor.run_test_intervals(name_base, expt_config,
                                                             combined_name,
                                                             True,
                                                             # alg_func_lower
                                                             lambda n_estimators, windowsize, params: predictor.create_tsfresh_windowed_regression_gbr(params, n_estimators, windowsize, data_samples_per_second, max_depth=4, learning_rate=0.1,
                                                                                                                                                       loss = "quantile", alpha = 0.05, min_samples_split=settings.min_samples_split_default, min_samples_leaf=settings.min_samples_leaf_default),
                                                             # alg_func_median
                                                             lambda n_estimators, windowsize, params: predictor.create_tsfresh_windowed_regression_gbr(params, n_estimators, windowsize, data_samples_per_second, max_depth=4, learning_rate=0.1,
                                                                                                                                                       loss = "quantile", alpha = 0.5, min_samples_split=settings.min_samples_split_default, min_samples_leaf=settings.min_samples_leaf_default),
                                                             # alg_func_upper
                                                             lambda n_estimators, windowsize, params: predictor.create_tsfresh_windowed_regression_gbr(params, n_estimators, windowsize, data_samples_per_second, max_depth=4, learning_rate=0.1,
                                                                                                                                                       loss = "quantile", alpha = 0.95, min_samples_split=settings.min_samples_split_default, min_samples_leaf=settings.min_samples_leaf_default),
                                                             
#                                                             lambda n_estimators, window_size, params:
#                                                             predictor.create_tsfresh_windowed_regression_gbr(params, n_estimators, window_size, data_samples_per_second, max_depth=4, learning_rate=0.1, min_samples_split=settings.min_samples_split_default, min_samples_leaf=settings.min_samples_leaf_default),
#                                                             
                                                             alg_params1=settings.n_estimator_choices,
                                                             alg_params2=settings.window_size_choices_secs,
                                                             param1_name = "n_estimators",
                                                             param2_name = "window_size")
                                                    

def run_tsfreshwin_ngboost_intervals(name_base, results_tag, expt_config):
    alg_name = "TSFreshWin_NGBoost"
    global interval_results
    combined_name = results_tag + alg_name
    interval_results = predictor.run_test_ngboost_intervals(name_base, alg_name, expt_config)
    
def run_tsfreshwin_ridge(name_base, results_tag, expt_config, alg_params1=settings.max_alphas, alg_params2=settings.window_size_choices_secs):
    global combined_results_all_algs
    alg_name = "TSFreshWin_Ridge"
    combined_name = results_tag + "_" + alg_name
    combined_results_all_algs = predictor.run_test(name_base, expt_config,
                                                    combined_results_all_algs,
                                                    combined_name,
                                                    True,
                                                    lambda max_alpha, window_size, params:
                                                    predictor.create_tsfresh_windowed_regression_ridge(params, window_size, data_samples_per_second, max_alpha=max_alpha, alpha_step=settings.alpha_step_default), 
                                                    alg_params1=alg_params1,
                                                    alg_params2=alg_params2,
                                                    param1_name = "max_alpha",
                                                    param2_name = "window_size")

def run_tsforest(name_base, results_tag, expt_config, alg_params1=settings.n_estimator_choices, alg_params2=settings.window_size_choices_secs):
    global combined_results_all_algs
    alg_name = "TSForest"
    combined_name = results_tag + "_" + alg_name
    combined_results_all_algs = predictor.run_test(name_base, expt_config,
                                                   combined_results_all_algs,
                                                   combined_name,
                                                   True,
                                                   lambda n_estimators, window_size, params:
                                                   # min_interval is data_samples_per_second * window_size
                                                   predictor.create_tsf_regression(n_estimators=n_estimators, min_interval=int(np.floor(data_samples_per_second * window_size))), 
                                                   alg_params1=alg_params1,
                                                   alg_params2=alg_params2,
                                                   param1_name = "n_estimators",
                                                   param2_name = "window_size")
   
################################################################################

def run_minirocket_ridge(name_base, results_tag, expt_config, alg_params1=settings.rocket_kernel_choices, alg_params2=settings.max_alphas ):
    global combined_results_all_algs
    alg_name = "MiniRocket_Ridge"
    combined_name = results_tag + "_" + alg_name
    combined_results_all_algs = predictor.run_test(name_base, expt_config,
                                                   combined_results_all_algs,
                                                   combined_name,
                                                   True,
                                                   lambda num_kernels, max_alpha, params:
                                                   predictor.create_minirocket_regression_ridge(params, num_kernels=num_kernels, max_alpha=max_alpha, alpha_step=settings.alpha_step_default), 
                                                   alg_params1=alg_params1,
                                                   alg_params2=alg_params2,
                                                   param1_name = "num_kernels",
                                                   param2_name = "max_alpha")

def run_inceptiontime(name_base, results_tag, expt_config):
    global combined_results_all_algs
    alg_name = "InceptionTime"
    combined_name = results_tag + "_" + alg_name
    default_n_epochs = 1500
    combined_results_all_algs = predictor.run_test(name_base, expt_config,
                                                   combined_results_all_algs,
                                                   combined_name,
                                                   True,
                                                   lambda n_epochs, kernel_size, params:
                                                   predictor.create_inceptiontime(n_epochs=n_epochs, kernel_size=kernel_size),
                                                   alg_params1=settings.inceptiontime_n_epochs,
                                                   alg_params2=settings.inceptiontime_kernel_sizes,
                                                   param1_name = "n_epochs",
                                                   param2_name = "kernel_size")

def run_minirocket_gradboost(name_base, results_tag, expt_config, alg_params1=settings.rocket_kernel_choices, alg_params2=settings.n_estimator_choices):
    global combined_results_all_algs
    alg_name = "MiniRocket_GradBoost"
    combined_name = results_tag + "_" + alg_name
    combined_results_all_algs = predictor.run_test(name_base, expt_config,
                                                    combined_results_all_algs,
                                                    combined_name,
                                                    True,
                                                    lambda num_kernels, n_estimators, params:
                                                    predictor.create_minirocket_regression_gbr(params, num_kernels, n_estimators, max_depth=4, learning_rate=0.1, min_samples_split=settings.min_samples_split_default, min_samples_leaf=settings.min_samples_leaf_default),
                                                    alg_params1=alg_params1,
                                                    alg_params2=alg_params2,
                                                    param1_name = "num_kernels",
                                                    param2_name = "n_estimators")

#def run_tsfreshwin_ngboost_intervals(name_base, results_tag, expt_config):
#    print("TSFreshWin NGBoost...")
#    global combined_results_all_algs
#    alg_name = "TSFreshWin_NGBoost_Intervals"
#    combined_name = results_tag + "_" + alg_name
#    combined_results_all_algs = predictor.run_test_ngboost_intervals(alg_name, expt_config)

def run_all_algs_on_dataset(expt_config, using_inceptiontime = True, run_intervals=False):
    global combined_results_all_algs

    combined_results_all_algs = None
    dataset_name = expt_config["dataset_name"]

    name_base = data_loader.create_directory_for_results(expt_config["dataset_name"]) + "/"

    if run_intervals:
        results_tag = "intervals"
        run_tsfreshwin_ngboost_intervals(name_base, results_tag, expt_config)
    else:
        if expt_config["use_fixed_windows"]:
            run_tsfreshwin_gradboost_fixed_window_size(name_base, dataset_name, expt_config)
            run_tsfreshwin_ridge_fixed_window_size(name_base, dataset_name, expt_config)
        else:
            run_tsfreshwin_gradboost(name_base, dataset_name, expt_config)
            run_tsfreshwin_ridge(name_base, dataset_name, expt_config)
        
    run_minirocket_gradboost(name_base, dataset_name, expt_config)
    run_minirocket_ridge(name_base, dataset_name, expt_config)
    run_tsforest(name_base, dataset_name, expt_config)

    # save results before running inceptiontime
    print("Combined results sorted by r2_score...\n")
    combined_result_file = name_base + dataset_name + "_sorted_summary_results.csv"
    combined_results_all_algs.to_csv(combined_result_file, sep=",")
    print(tabulate(combined_results_all_algs, headers="keys"))
    # Log LaTeX results
    summary_file_latex = name_base + dataset_name + "_sorted_summary_results.tex"
    predictor.log_latex_summary_results(combined_results_all_algs, sorted_by_col="r2_score_mean", limit=20, filename=summary_file_latex)

    # inceptiontime is very long running
    if using_inceptiontime:
        run_inceptiontime(dataset_name, expt_config)
        
    print("Combined results sorted by r2_score...\n")
    combined_result_file = dataset_name + "_sorted_summary_results.csv"
    combined_results_all_algs.to_csv(combined_result_file, sep=",")
    print(tabulate(combined_results_all_algs, headers="keys"))

def run_filprofiler_memory_tracking(expt_config, memory_filename, selected_alg, alg_params1, alg_params2):
    global combined_results_all_algs

    if not (memory_filename is None):
        expt_config["memory_tracking_creator"] = lambda: FilProfilerMemoryTracker(memory_filename)

    combined_results_all_algs = None
    dataset_name = expt_config["dataset_name"]

    name_base = data_loader.create_directory_for_results(expt_config["dataset_name"]) + "/"

    if expt_config["use_fixed_windows"]:
        if selected_alg == "TSFreshWin_GradBoost":
            run_tsfreshwin_gradboost_fixed_window_size(name_base, dataset_name, expt_config, alg_params1=alg_params1, alg_params2=alg_params2)
        if selected_alg == "TSFreshWin_Ridge":
            run_tsfreshwin_ridge_fixed_window_size(name_base, dataset_name, expt_config, alg_params1=alg_params1, alg_params2=alg_params2)

    else:
        if selected_alg == "TSFreshWin_GradBoost":
            run_tsfreshwin_gradboost(name_base, dataset_name, expt_config, alg_params1=alg_params1, alg_params2=alg_params2)
        if selected_alg == "TSFreshWin_Ridge":
            run_tsfreshwin_ridge(name_base, dataset_name, expt_config, alg_params1=alg_params1, alg_params2=alg_params2)

    if selected_alg == "MiniRocket_GradBoost":
        run_minirocket_gradboost(name_base, dataset_name, expt_config, alg_params1=alg_params1, alg_params2=alg_params2)
    if selected_alg == "MiniRocket_Ridge":
        run_minirocket_ridge(name_base, dataset_name, expt_config, alg_params1=alg_params1, alg_params2=alg_params2)
    if selected_alg == "TSForest":
        run_tsforest(name_base, dataset_name, expt_config, alg_params1=alg_params1, alg_params2=alg_params2)

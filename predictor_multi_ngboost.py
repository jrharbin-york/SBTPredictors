from sktime.datatypes import check_raise, convert_to

from sktime.performance_metrics.forecasting import MeanSquaredError

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from timeit import default_timer as timer

import tsfresh
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor

from scipy import stats

import numpy as np

import pandas as pd
import structlog

import data_loader

from ngboost import NGBRegressor
from ngboost.distns import Exponential, Normal, MultivariateNormal

from sktime.pipeline import make_pipeline
from tabulate import tabulate


log = structlog.get_logger()

k_fold_shuffle = True

def calc_mae_from_intervals(lower, upper, actual, is_inside_interval):
    # Hit: (UBound - Avalue) + (Avalue - LBound)
    # MissU: (Avalue-UBound) + (Avalue - LBound)
    # MissL: (UBound-Avalue) + (LBound-Avalue)

    is_above = (actual > upper)
    is_below = (actual < lower)

    if is_above:
        return (actual - upper) + (actual - lower)
    else:
        if is_below:
            return (upper - actual) + (lower - actual)
        else:
            # is inside the range
            return (upper - actual) + (actual - lower)

def calc_mae_from_intervals_all(lower, upper, actual, is_inside_interval):
    size = lower.size
    maes = np.zeros(size)
    for i in range(0,size):
        maes[i] = calc_mae_from_intervals(lower[i], upper[i], actual[i], is_inside_interval[i])
    return maes

def create_tsfresh_windowed_featuresonly(params, n_estimators, windowsize, res_samples_per_second):
    feature_name = "jrh_windowed_features_calculation_fixedsize"
    settings = {}
    log.debug("create windowsize=" + str(windowsize))
    settings[feature_name] = {"windowsize" : windowsize, "resolution_samples_per_second" : res_samples_per_second }
    col_names = params["needed_columns"]

    features_selected = list(map(lambda col_name: col_name + "__" + feature_name + "__windowsize_" + str(windowsize) + "__resolution_samples_per_second_" + str(res_samples_per_second), col_names))

    for fn in features_selected:
        settings[fn] = {"windowsize" : windowsize}
    
    print("features_selected = " + str(features_selected))
    t_features = TSFreshFeatureExtractor(default_fc_parameters=settings, kind_to_fc_parameters=features_selected, show_warnings=False)
    pipeline = make_pipeline(t_features)
    return pipeline

def plot_regression_intervals_new(regression_graph_title, predicted_vs_actual_unsorted, expt_config, filename="regression.pdf"):
    # Plot with error bars
    predicted_vs_actual = predicted_vs_actual_unsorted.sort_values(by=['actual_val'])
    
    print("length of dataframe:" + str(len(predicted_vs_actual)))
    plt.clf()

    error_above = np.abs(predicted_vs_actual["predicted_val_upper"] - predicted_vs_actual["predicted_val_median"])
    error_below = np.abs(predicted_vs_actual["predicted_val_median"] - predicted_vs_actual["predicted_val_lower"])
    test_indices = np.arange(1, len(predicted_vs_actual)+1)
    plt.errorbar(test_indices, predicted_vs_actual["predicted_val_median"], yerr=[error_below, error_above], marker='.', fmt="none", elinewidth=0.3)
    plt.scatter(test_indices, predicted_vs_actual["actual_val"], c="r", marker="x", linewidths=0.1)
    
    plt.xlabel("Test indices sorted by actual value")
    plt.ylabel("Intervals vs actual value of sensitive point")

    # ignore resetting x values
    plot_y_lower = expt_config["plot_y_lower"]
    plot_y_upper = expt_config["plot_y_upper"]
    
    #plt.xlim([plot_x_lower, plot_x_upper])
    plt.ylim([plot_y_lower, plot_y_upper])
    plt.title(regression_graph_title)
    plt.savefig(filename)

def confidence_intervals_from_ngboost(y_dist_test, num_metrics, interval_val=0.95):
    locs = y_dist_test.params["loc"]
    scales = y_dist_test.params["scale"]
    size = len(locs)
    print("size=", size)
    conf_intervals_lower = np.zeros((size, num_metrics),dtype=float)
    conf_intervals_upper = np.zeros((size, num_metrics),dtype=float)
    normal_means = np.zeros((size, num_metrics),dtype=float)

    for i in range(0, size):
        for j in range(0, num_metrics):
            # Take only the direct coefficients - not cross coefficients
            conf_int = stats.norm.interval(interval_val, loc=locs[i][j],scale=scales[i][j][j])
            conf_intervals_lower[i][j] = conf_int[0]
            conf_intervals_upper[i][j] = conf_int[1]
            normal_means[i][j] = locs[i][j]

    return conf_intervals_lower, conf_intervals_upper, normal_means

def test_regression_ngboost_intervals(id_code, alg_name, alg_func, fig_filename_func, pd_res, expt_config, summary_res, alg_param1, alg_param2, k=5):
    params = {}

    params["base_dir"] = expt_config["data_dir_base"]
    params["target_metric_name"] = expt_config["target_metric_name"]
    params["needed_columns"] = expt_config["needed_columns"]    
    mfile = expt_config["data_dir_base"] + "/allMetrics.csv"
    data_files, metrics = data_loader.read_data(expt_config["data_dir_base"], mfile)
    target_metric_name = expt_config["target_metric_name"]

    alg_func_delayed = lambda params: alg_func(alg_param1, alg_param2, params)

    base_dir = params["base_dir"]
    needed_columns = params["needed_columns"]
    alg_name = params["target_metric_name"]   
    
    k=5
    kf = KFold(n_splits=k, shuffle=k_fold_shuffle)

    # Accumulate these over all splits
    r2_score_all_splits = []
    mse_all_splits = []
    rmse_all_splits = []
    
    for i, (train_index, test_index) in enumerate(kf.split(data_files)):
        # Split the data for k_fold validation
        data_files_train = [data_files[i] for i in train_index]
        metrics_train_pandas = metrics.iloc[train_index]
        data_files_test = [data_files[i] for i in test_index]
        metrics_test_pandas = metrics.iloc[test_index]
        log.debug(f"Fold {i}:")


        time_start = timer()
        # instead of single pipeline
        # take the feature extractor and produce the features, convert them to 2D X array
        # then create the 2D Y array of metrics per test
        # fit ngboost using the ngboost example
        # performance testing using this

        train_data = data_loader.create_combined_data(base_dir, data_files_train, needed_columns)
        test_data = data_loader.create_combined_data(base_dir, data_files_test, needed_columns)

        target_metric_names = ["distanceToHuman1", "distanceToStaticHumans", "pathCompletion"]
        num_metrics = len(target_metric_names)

        metrics_train = np.array(metrics_train_pandas[target_metric_names])
        metrics_test =  np.array(metrics_test_pandas[target_metric_names])

        print(f"metrics_train={metrics_train.shape}")
        #metrics_train = np.array(metrics_train_pandas[target_metric_name])
        #metrics_test =  np.array(metrics_test_pandas[target_metric_name])
        # metrics_train needs to be a 2D array - number of columns same an number of metrics
        log.debug("Check data format train_metrics: %s",check_raise(metrics_test, mtype="np.ndarray"))
        log.debug("Check data format test_metrics: %s",check_raise(metrics_train, mtype="np.ndarray"))

        # Need a custom pipeline to just generate the features here...
        # use create_tsfresh_windowed_featuresonly
        res_samples_per_second = 10
        n_estimators = alg_param1
        window_size = alg_param2

        confidence_intervals_param = 0.95
        
        features_pipe = create_tsfresh_windowed_featuresonly(params, n_estimators, window_size, res_samples_per_second)
        features_pipe.fit(train_data, metrics_train)
        x_features_train = features_pipe.transform(train_data)
        x_features_test = features_pipe.transform(test_data)

        # this should be based on all metric lengths
        dist = MultivariateNormal(num_metrics)
        ngb = NGBRegressor(Dist=dist, verbose=True, n_estimators=n_estimators, natural_gradient=True)
        ngb.fit(x_features_train, metrics_train, early_stopping_rounds=1000)
        y_dist_test = ngb.pred_dist(x_features_test, max_iter=ngb.best_val_loss_itr)
        conf_intervals_lower, conf_intervals_upper, normal_mean = confidence_intervals_from_ngboost(y_dist_test, num_metrics, confidence_intervals_param)

        for j in range(0, num_metrics):
            metric_name = target_metric_names[j]
            actual_val = metrics_test[:,j]

            interval_width = conf_intervals_upper[:,j] - conf_intervals_lower[:,j]
            is_inside_interval = np.logical_and((actual_val >= conf_intervals_lower[:,j]), (actual_val <= conf_intervals_upper[:,j]))
            mae_intervals = calc_mae_from_intervals_all(conf_intervals_lower[:,j], conf_intervals_upper[:,j], actual_val, is_inside_interval)

            interval_width_mean = np.mean(interval_width)
            interval_width_stddev = np.std(interval_width)
            inside_interval_proportion = is_inside_interval.sum() / len(is_inside_interval)
            mae_intervals_mean = np.mean(mae_intervals)

            fig_filename = fig_filename_func(id_code, i, metric_name)

            # when doing multi-predictions need to create hash of predictions per metric
            predicted_vs_actual = pd.DataFrame({'predicted_val_lower':conf_intervals_lower[:,j],
                                                     'predicted_val_median':normal_mean[:,j],
                                                     'predicted_val_upper':conf_intervals_upper[:,j],
                                                     'actual_val':actual_val,
                                                     'interval_width':interval_width,
                                                     'inside_interval_proportion':inside_interval_proportion,
                                                     'mae_intervals':mae_intervals_mean},
                                               columns = ['predicted_val_lower',
                                                          'predicted_val_median',
                                                          'predicted_val_upper',
                                                          'actual_val',
                                                          'interval_width',
                                                          'is_inside_interval',
                                                        'mae_intervals' ])
            log.debug("Plotting regression intervals plot to %s", fig_filename)
            plot_regression_intervals_new(expt_config["regression_graph_title"], predicted_vs_actual, expt_config, fig_filename)

            time_end = timer()
            time_diff = time_end - time_start

            mse_c = MeanSquaredError()
            rmse_c = MeanSquaredError(square_root=True)

            results_this_test = {"id":id_code,
                                 "k_split":i,
                                 "param1":alg_param1,
                                 "param2":alg_param2,
                                 "interval_width_mean" : interval_width_mean,
                                 "interval_width_stddev" : interval_width_stddev,
                                 "in_interval_proportion" : inside_interval_proportion,
                                 "mae" : mae_intervals_mean,
                                 "filename_graph":fig_filename,
                                 "time_diff":time_diff
                                 }
        pd_res.loc[len(pd_res)] = results_this_test
        # change filename

    mean_r2 = np.mean(r2_score_all_splits)
    mean_mse = np.mean(mse_all_splits)
    mean_rmse = np.mean(rmse_all_splits)

    stddev_r2 = np.std(r2_score_all_splits)
    stddev_mse = np.std(mse_all_splits)
    stddev_rmse = np.std(rmse_all_splits)

    summary_this_test = {"param1":alg_param1, "param2":alg_param2, "r2_score_mean":mean_r2, "mse_mean":mean_mse, "rmse_mean":mean_rmse, "r2_score_stddev":stddev_r2, "mse_score_stddev":stddev_mse, "rmse_score_stddev":stddev_rmse}
    summary_res.loc[len(summary_res)] = summary_this_test

    log.info("Mean r2 all splits = %f, stddev r2 all splits = %f", mean_r2, stddev_r2)
    log.info("Mean MSE all splits = %f, stddev MSE all splits = %f", mean_mse, stddev_mse)
    log.info("Mean RMSE all splits = %f, stddev RMSE all splits = %f", mean_rmse, stddev_rmse)

    return pd_res, summary_res

def run_test_ngboost_intervals_multi(name_base, expt_config, combined_results_all_tests, alg_name, alg_params1, alg_params2, param1_name, param2_name):
    individual_results = pd.DataFrame(columns=["id", "param1", "param2", "k_split", "filename_graph", "time_diff", "in_interval_proportion", "interval_width_mean", "interval_width_stddev", "mae"])
    stats_results = pd.DataFrame(columns=["param1", "param2", "r2_score_mean", "mse_mean", "rmse_mean", "r2_score_stddev", "mse_score_stddev", "rmse_score_stddev"])
    id_code = 0

    results_file = name_base + alg_name + "-interval-res.csv"
    summary_file = name_base + alg_name + "-interval-summary-stats.csv"

    id_num = 0
    for param1 in alg_params1:
        for param2 in alg_params2:
            n_estimators = param1
            windowsize = param2
            fig_filename_func = lambda id_n, k_split, metric_name: name_base + metric_name + "-intervals-" + alg_name + "-ID" + str(id_n) + "-" + str(param1) + "-" + str(param2) + "-" + "k_split" + str(k_split) +".png"
            id_num+=1
            id_code = "ID" + str(id_num) + param1_name + str(param1) + "_" + param2_name + str(param2)
            alg_func = lambda min_samples_split, min_samples_leaf, params: create_tsfresh_windowed_featuresonly(params, n_estimators, windowsize, 10.0)
            individual_results, stats_results = test_regression_ngboost_intervals(id_code, alg_name, alg_func,
                                                          fig_filename_func, individual_results, expt_config, stats_results, alg_param1=param1, alg_param2=param2)
            individual_results.to_csv(results_file, sep=",")
            print(tabulate(individual_results, headers="keys"))

    individual_results.to_csv(results_file, sep=",")
    stats_results.to_csv(summary_file, sep=",")
    print(tabulate(stats_results, headers="keys"))




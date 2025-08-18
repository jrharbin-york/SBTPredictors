import keras
from ngboost.distns.lognormal import LogNormalCRPScoreCensored
from ngboost.distns.multivariate_normal import MVNLogScore
from ngboost.scores import CRPScore, MLE
from sktime.datatypes import check_raise, convert_to

from sktime.performance_metrics.forecasting import MeanSquaredError

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from timeit import default_timer as timer

from xgboostlss.model import *
from xgboostlss.distributions.MVN import *
import xgboostlss as xgb

from sklearn.model_selection import train_test_split
import pandas as pd
import multiprocessing
import plotnine
from plotnine import *

import tsfresh
from tensorflow.keras.callbacks import EarlyStopping
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

import qrnn
from predictor import conf_intervals_from_quantiles, quantile_loss, confidence_intervals_from_normal_distributions



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

def plot_regression_intervals_new(regression_graph_title, predicted_vs_actual_unsorted, expt_config, filename="regression.pdf", plot_y_lower=0.0, plot_y_upper=1.0):
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

def test_regression_xgboost_intervals(id_code, alg_name, alg_func, fig_filename_func, pd_res, expt_config, summary_res, alg_param1, alg_param2, k=5):
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

    #target_metric_names = ["distanceToHuman1", "distanceToStaticHumans", "pathCompletion"]
    #target_metric_names = ["pathCompletion", "distanceToStaticHumans", "distanceToHuman1"]
    target_metric_names = ["pathCompletion"]
    num_metrics = len(target_metric_names)

    n_estimators = alg_param1
    window_size = alg_param2
    confidence_intervals_param = 0.95
    res_samples_per_second = 10

    # Accumulate these over all splits
    interval_width_means = {}
    in_interval_proportions = {}
    mae_intervals_splits = {}
        
    for i, (train_index, test_index) in enumerate(kf.split(data_files)):
        # Split the data for k_fold validation
        data_files_train = [data_files[i] for i in train_index]
        metrics_train_pandas = metrics.iloc[train_index]
        data_files_test = [data_files[i] for i in test_index]
        metrics_test_pandas = metrics.iloc[test_index]
        log.debug(f"All metrics, fold={i}:")

        train_data = data_loader.create_combined_data(base_dir, data_files_train, needed_columns)
        test_data = data_loader.create_combined_data(base_dir, data_files_test, needed_columns)
        metrics_train = np.array(metrics_train_pandas[target_metric_names])
        metrics_test = np.array(metrics_test_pandas[target_metric_names])

        time_start = timer()
        # instead of single pipeline
        # take the feature extractor and produce the features, convert them to 2D X array
        # then create the 2D Y array of metrics per test
        features_pipe = create_tsfresh_windowed_featuresonly(params, n_estimators, window_size, res_samples_per_second)

        features_pipe.fit(train_data, metrics_train)
        x_features_train = features_pipe.transform(train_data)
        x_features_test = features_pipe.transform(test_data)

        # TODO: make this a param
        n_cpu = 12

        # Train
        # this needs to be the features!
        dtrain = xgb.DMatrix(x_features_train, label=metrics_train, nthread=n_cpu)
        n_targets = metrics_train.shape[1]

        # Validation - perform further split for this
        deval = xgb.DMatrix(x_eval, label=y_eval, nthread=n_cpu)

        # Test
        dtest = xgb.DMatrix(x_features_train, nthread=n_cpu)

        # this should be based on all metric lengths
        # Specifies a multivariate Normal distribution, using the Cholesky decompoisition. See ?MVN for details.
        xgblss = xgb.XGBoostLSS(
            xgb.MVN(D=n_targets,  # Specifies the number of targets
                stabilization="None",  # Options are "None", "MAD", "L2".
                response_fn="exp",
                # Function to transform the lower-triangular factor of the covariance, e.g., "exp" or "softplus".
                loss_fn="nll"  # Loss function, i.e., nll.
                )
        )

        # hyper-parameter selection
        param_dict = {
            "eta": ["float", {"low": 1e-5, "high": 1, "log": True}],
            "max_depth": ["int", {"low": 1, "high": 10, "log": False}],
            "gamma": ["float", {"low": 1e-8, "high": 40, "log": True}],
            "subsample": ["float", {"low": 0.2, "high": 1.0, "log": False}],
            "colsample_bytree": ["float", {"low": 0.2, "high": 1.0, "log": False}],
            "min_child_weight": ["float", {"low": 1e-8, "high": 500, "log": True}],
        }

        np.random.seed(123)
        opt_param = xgblss.hyper_opt(param_dict,
                                     dtrain,
                                     num_boost_round=100,  # Number of boosting iterations.
                                     nfold=5,  # Number of cv-folds.
                                     early_stopping_rounds=20,  # Number of early-stopping rounds
                                     max_minutes=120,
                                     # Time budget in minutes, i.e., stop study after the given number of minutes.
                                     n_trials=20,
                                     # The number of trials. If this argument is set to None, there is no limitation on the number of trials.
                                     silence=False,
                                     # Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
                                     seed=123,  # Seed used to generate cv-folds.
                                     hp_seed=None
                                     # Seed for random number generator used in the Bayesian hyperparameter search.
                                     )

        # model training
        np.random.seed(123)

        opt_params = opt_param.copy()
        n_rounds = opt_params["opt_rounds"]
        del opt_params["opt_rounds"]

        # Add evaluation set
        eval_set = [(dtrain, "train"), (deval, "evaluation")]
        eval_result = {}

        # Train Model with optimized hyperparameters
        xgblss.train(
            opt_params,
            dtrain,
            num_boost_round=n_rounds,
            evals=eval_set,
            evals_result=eval_result,
            verbose_eval=False
        )

        # Set seed for reproducibility
        #torch.manual_seed(123)

        # Number of samples to draw from predicted distribution
        n_samples = 1000
        quant_sel = [0.05, 0.95]  # Quantiles to calculate from predicted distribution

        # Sample from predicted distribution
        pred_samples = xgblss.predict(dtest,
                                      pred_type="samples",
                                      n_samples=n_samples,
                                      seed=123)

        # Calculate quantiles from predicted distribution
        pred_quantiles = xgblss.predict(dtest,
                                        pred_type="quantiles",
                                        n_samples=n_samples,
                                        quantiles=quant_sel)

        # Returns predicted distributional parameters
        pred_params = xgblss.predict(dtest,
                                     pred_type="parameters")



        for j in range(0, num_metrics):
            # may need to filter using
            metric_name = target_metric_names[j]
            pred_ql_metric_name = pred_quantiles[pred_samples.target=="metric_name"]

            conf_intervals_lower = pred_ql_metric_name["quant_0.05"]
            conf_intervals_upper = pred_ql_metric_name["quant_0.95"]

            print(f"metrics_train={metrics_train.shape}")
            # metrics_train needs to be a 2D array - number of columns same an number of metrics
            log.debug("Check data format train_metrics: %s", check_raise(metrics_test, mtype="np.ndarray"))
            log.debug("Check data format test_metrics: %s", check_raise(metrics_train, mtype="np.ndarray"))

            actual_val = metrics_test[:, j]

            interval_width = conf_intervals_upper[:,j] - conf_intervals_lower[:,j]
            is_inside_interval = np.logical_and((actual_val >= conf_intervals_lower[:,j]), (actual_val <= conf_intervals_upper[:,j]))
            mae_intervals = calc_mae_from_intervals_all(conf_intervals_lower[:,j], conf_intervals_upper[:,j], actual_val, is_inside_interval)

            interval_width_mean = np.mean(interval_width)
            interval_width_stddev = np.std(interval_width)
            inside_interval_proportion = is_inside_interval.sum() / len(is_inside_interval)
            mae_intervals_mean = np.mean(mae_intervals)

            fig_filename = fig_filename_func(id_code, i, metric_name)

            if metric_name=="pathCompletion":
                # for debugging
                print("pathCompletion")

            # when doing multi-predictions need to create hash of predictions per metric
            predicted_vs_actual = pd.DataFrame({'predicted_val_lower':conf_intervals_lower[:,j],
                                                     'predicted_val_median':normal_mean[:,j],
                                                     'predicted_val_upper':conf_intervals_upper[:,j],
                                                     'actual_val':actual_val,
                                                     'interval_width':interval_width,
                                                     'inside_interval_proportion':inside_interval_proportion,
                                                     'mae_intervals':mae_intervals_mean,
                                                     'metric_name':metric_name
                                                },
                                               columns = ['metric_name',
                                                          'predicted_val_lower',
                                                          'predicted_val_median',
                                                          'predicted_val_upper',
                                                          'actual_val',
                                                          'interval_width',
                                                          'is_inside_interval',
                                                          'mae_intervals' ])
            log.debug("Plotting regression intervals plot to %s", fig_filename)

            plot_y_min = 0.0
            plot_y_max = 1.0

            if metric_name=="distanceToHuman1":
                plot_y_min = -2.0
                plot_y_max = 14.0

            if metric_name=="distanceToStaticHumans":
                plot_y_min = 0.0
                plot_y_max = 4.0

            if metric_name=="pathCompletion":
                plot_y_min = 0.0
                plot_y_max = 1.0

            plot_regression_intervals_new(expt_config["regression_graph_title"], predicted_vs_actual, expt_config, fig_filename, plot_y_lower=plot_y_min, plot_y_upper=plot_y_max)

            time_diff = time_end - time_start

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
            print(f"results_this_test={results_this_test}")
            pd_res.loc[len(pd_res)] = results_this_test

            if not (metric_name in interval_width_means):
                interval_width_means[metric_name] = []

            if not (metric_name in in_interval_proportions):
                in_interval_proportions[metric_name] = []

            if not (metric_name in mae_intervals_splits):
                mae_intervals_splits[metric_name] = []

            interval_width_means[metric_name] = np.append(interval_width_means[metric_name], interval_width_mean)
            in_interval_proportions[metric_name] = np.append(in_interval_proportions[metric_name], inside_interval_proportion)
            mae_intervals_splits[metric_name] = np.append(mae_intervals_splits[metric_name], mae_intervals_mean)

    for j in range(0, num_metrics):
        metric_name = target_metric_names[j]
        interval_width_mean_mean = np.mean(interval_width_means[metric_name])
        in_interval_proportion_mean = np.mean(in_interval_proportions[metric_name])
        mae_intervals_mean = np.mean(mae_intervals_splits[metric_name])

        summary_this_test = { "metric_name":metric_name, "param1":alg_param1, "param2":alg_param2,
                              "interval_width_mean_mean":interval_width_mean_mean,
                              "in_interval_proportion_mean":in_interval_proportion_mean,
                              "mae_intervals_mean":mae_intervals_mean }

        summary_res.loc[len(summary_res)] = summary_this_test

        log.info("interval_width_mean_mean = %f", interval_width_mean_mean)
        log.info("in_interval_proportion_mean = %f", in_interval_proportion_mean)
        log.info("mae_intervals_mean = %f", mae_intervals_mean)

    return pd_res, summary_res

def run_test_xgboostlss(name_base, expt_config, combined_results_all_tests, alg_name, alg_params1, alg_params2, param1_name, param2_name):
    individual_results = pd.DataFrame(columns=["id", "param1", "param2", "k_split", "filename_graph", "time_diff", "in_interval_proportion", "interval_width_mean", "interval_width_stddev", "mae"])
    stats_results = pd.DataFrame(columns=["param1", "param2", "r2_score_mean", "mse_mean", "rmse_mean", "r2_score_stddev", "mse_score_stddev", "rmse_score_stddev"])
    id_code = 0

    results_file = name_base + "-" + alg_name + "-interval-res.csv"
    summary_file = name_base + "-" + alg_name + "-interval-summary-stats.csv"

    id_num = 0
    for param1 in alg_params1:
        for param2 in alg_params2:
            n_estimators = param1
            window_size = param2
            fig_filename_func = lambda id_n, k_split: name_base + "qrnn-intervals-" + alg_name + "-ID" + str(id_n) + "-" + str(param1) + "-" + str(param2) + "-" + "k_split" + str(k_split) +".png"
            id_num+=1
            id_code = "ID" + str(id_num) + param1_name + str(param1) + "_" + param2_name + str(param2)
            alg_func = lambda min_samples_split, min_samples_leaf, params: create_tsfresh_windowed_featuresonly(params, n_estimators, window_size, 10.0)
            individual_results, stats_results = test_regression_xgboost_intervals(id_code, alg_name, alg_func,
                                                          fig_filename_func, individual_results, expt_config, stats_results, alg_param1=param1, alg_param2=param2)
            individual_results.to_csv(results_file, sep=",")
            print(tabulate(individual_results, headers="keys"))

    individual_results.to_csv(results_file, sep=",")
    stats_results.to_csv(summary_file, sep=",")
    print(tabulate(stats_results, headers="keys"))

# Need a launcher directly here
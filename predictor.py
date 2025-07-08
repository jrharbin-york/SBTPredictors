from ngboost.scores import CRPScore
from sklearn.multioutput import RegressorChain
from sktime.datatypes import check_raise, convert_to
from sktime.transformations.panel.rocket import Rocket, MiniRocket, MiniRocketMultivariate, MultiRocketMultivariate
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.transformations.compose import FeatureUnion, TransformerPipeline
from sktime.transformations.series.binning import TimeBinAggregate
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.transformations.series.impute import Imputer
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.regression.interval_based import TimeSeriesForestRegressor
from sktime.regression.deep_learning import InceptionTimeRegressor

from sktime.classification.ensemble import ComposableTimeSeriesForestClassifier
from sktime.classification.deep_learning import MVTSTransformerClassifier
from sktime.transformations.series.subset import ColumnSelect

from sktime.transformations.panel.compose import ColumnConcatenator

from sktime.performance_metrics.forecasting import MeanAbsoluteError
from sktime.performance_metrics.forecasting import MeanSquaredError
from sktime.performance_metrics.forecasting import median_absolute_percentage_error

from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import r2_score

from matplotlib import pyplot as plt
from sktime.utils.plotting import plot_series
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from timeit import default_timer as timer

from sklearn.linear_model import RidgeClassifierCV, RidgeCV, SGDClassifier, SGDRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

import tsfresh
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats

import datetime as dt
import numpy as np
import glob
import sys
import pandas as pd
import os
import gc
import psutil
import tracemalloc
import structlog
import logging
import analyse_pareto_fronts

import data_loader

from ngboost import NGBRegressor
from ngboost.distns import Exponential, Normal

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from filprofiler.api import profile

from sktime.pipeline import make_pipeline
from tabulate import tabulate

log = structlog.get_logger()

k_fold_shuffle = True

class MemoryTracker:
    pass

class TraceMallocMemoryTracker(MemoryTracker):
    def __init__(self):
        pass
        
    def start_tracking(self):
        tracemalloc.start()

    def end_tracking(self):
        stats = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    def with_tracking(self, function_to_track):
        return function_to_track() 

class PSUtilMemoryTracker(MemoryTracker):
    def __init__(self):
        self.with_gc_disabled = False
        self.process = psutil.Process(os.getpid())

    def clear_gc(self):
        log.info("Running GC to clear memory")
        gc.collect()

    def enable_gc(self):
        log.info("Enabling GC")
        gc.enable()

    def disable_gc(self):
        log.info("Disabling GC")
        gc.disable()
        
    def start_tracking(self):
        if self.with_gc_disabled:
            self.clear_gc()
            self.disable_gc()
        self.mem_info_before = self.process.memory_info()
        log.info("Starting memory tracking")

    def end_tracking(self):
        self.mem_info_after = self.process.memory_info()
        if self.with_gc_disabled:
            self.clear_gc()
            self.enable_gc()
        log.info("Ending memory tracking")
        return self.mem_info_after.rss - self.mem_info_before.rss

    def with_tracking(self, function_to_track):
        return function_to_track()

class PSUtilMemoryTrackerNoGC(PSUtilMemoryTracker):
    def __init__(self):
        super().__init__()
        self.with_gc_disabled = True

class FilProfilerMemoryTracker(MemoryTracker):
    def __init__(self):
        pass

    def start_tracking(self):
        # Redundant, the tracking is done in with_tracking
        pass

    def end_tracking(self):
        # Redundant, the tracking is done in with_tracking
        pass

    def with_tracking(self, function_to_track):
        return profile(function_to_track, "/tmp/fil-result")
        

MEMORY_TRACKING_CLASS = PSUtilMemoryTracker
  
#def process_memory():
#    process = psutil.Process(os.getpid())
#    mem_info = process.memory_info()
#    return mem_info.rss

def metricval_to_class(metricval):
    return metricval

def run_regression_or_classifier(regression, pipeline_gen_func, alg_name, params):
    base_dir = params["base_dir"]
    data_files_train = params["data_files_train"]
    data_files_test = params["data_files_test"]
    metrics_train_pandas = params["metrics_train_pandas"]
    metrics_test_pandas = params["metrics_test_pandas"]
    target_metric_name = params["target_metric_name"]
    needed_columns = params["needed_columns"]
    alg_name = params["target_metric_name"]
        
    train_data = data_loader.create_combined_data(base_dir, data_files_train, needed_columns)
    test_data = data_loader.create_combined_data(base_dir, data_files_test, needed_columns)
    metrics_train = np.array(metrics_train_pandas[target_metric_name])
    metrics_test =  np.array(metrics_test_pandas[target_metric_name])
    log.debug("Check data format train_metrics: %s",check_raise(metrics_test, mtype="np.ndarray"))
    log.debug("Check data format test_metrics: %s",check_raise(metrics_train, mtype="np.ndarray"))

    log.debug("needed_columns=" + str(needed_columns))

    if regression:
        # regressor
        pipe = pipeline_gen_func(params)
        pipe.fit(train_data, metrics_train)
        if (hasattr(pipe, "regressor_")):
            if (hasattr(pipe.regressor_, "feature_importances_")):
                importance = pipe.regressor_.feature_importances_
                print("importance of features = ", importance)
        
        log.info("%s regressor fit done!", alg_name)
        r2_score = pipe.score(test_data, metrics_test)
        log.info("r2 Score on test data = %s", r2_score)
        predicted_val = pipe.predict(test_data)
        actual_val = metrics_test
        predicted_vs_actual = pd.DataFrame({'predicted_val':predicted_val, 'actual_val':actual_val}, columns = ['predicted_val', 'actual_val'])
        return [pipe, r2_score, predicted_vs_actual]
    else:
        #classifier
        class_count = params["class_count"]
        metric_max = np.max([np.max(metrics_train), np.max(metrics_test)])
        if class_count is None:
            train_class = metrics_train
            test_class = metrics_test
        else:
            metric_min = np.min([np.min(metrics_train), np.min(metrics_test)])
            train_class = metricval_to_class(metrics_train, metric_max, metric_min, class_count)
            test_class = metricval_to_class(metrics_test, metric_max, metric_min, class_count)

        pipe = pipeline_gen_func(params)
        pipe.fit(train_data, train_class)
        print("%s classifier fit done!" % alg_name)
        accuracy = pipe.score(test_data, test_class)
        print("Accuracy on test data = ", accuracy)
        predicted_class = pipe.predict(test_data)
        predicted_class_probs = pipe.predict_proba(test_data)
        class_labels = np.arange(0,len(predicted_class_probs[0]))
        actual_class = test_class
        predicted_vs_actual = pd.DataFrame({'predicted_class':predicted_class, 'actual_class':actual_class}, columns = ['predicted_class', 'actual_class'])
        return [pipe, accuracy, predicted_vs_actual, class_labels, predicted_class_probs]

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

def run_regression_intervals(pipeline_gen_func_lower, pipeline_gen_func_median, pipeline_gen_func_upper, alg_name, params):
    base_dir = params["base_dir"]
    data_files_train = params["data_files_train"]
    data_files_test = params["data_files_test"]
    metrics_train_pandas = params["metrics_train_pandas"]
    metrics_test_pandas = params["metrics_test_pandas"]
    target_metric_name = params["target_metric_name"]
    needed_columns = params["needed_columns"]
    alg_name = params["target_metric_name"]
        
    train_data = data_loader.create_combined_data(base_dir, data_files_train, needed_columns)
    test_data = data_loader.create_combined_data(base_dir, data_files_test, needed_columns)
    metrics_train = np.array(metrics_train_pandas[target_metric_name])
    metrics_test = np.array(metrics_test_pandas[target_metric_name])
    log.debug("Check data format train_metrics: %s",check_raise(metrics_test, mtype="np.ndarray"))
    log.debug("Check data format test_metrics: %s",check_raise(metrics_train, mtype="np.ndarray"))

    # regressor
    pipe_lower = pipeline_gen_func_lower(params)
    pipe_median = pipeline_gen_func_median(params)
    pipe_upper = pipeline_gen_func_upper(params)
    pipe_lower.fit(train_data, metrics_train)
    pipe_median.fit(train_data, metrics_train)
    pipe_upper.fit(train_data, metrics_train)
    importance = pipe_median.regressor_.feature_importances_
    print("importance of features = ", importance)
    
    log.info("%s regressor fit done!", alg_name)
    r2_score = pipe_median.score(test_data, metrics_test)
    log.info("r2 Score on test data = %s", r2_score)
    predicted_val_lower = pipe_lower.predict(test_data)
    predicted_val_median = pipe_median.predict(test_data)
    predicted_val_upper = pipe_upper.predict(test_data)

    actual_val = metrics_test

    interval_width = predicted_val_upper - predicted_val_lower
    is_inside_interval = np.logical_and((actual_val >= predicted_val_lower), (actual_val <= predicted_val_upper))
    mae_intervals = calc_mae_from_intervals_all(predicted_val_lower, predicted_val_upper, actual_val, is_inside_interval)
    
    predicted_vs_actual = pd.DataFrame({'predicted_val_lower':predicted_val_lower,
                                        'predicted_val_median':predicted_val_median,
                                        'predicted_val_upper':predicted_val_upper,
                                        'actual_val':actual_val,
                                        'interval_width':interval_width,                                       
                                        'is_inside_interval':is_inside_interval,
                                        'mae_intervals':mae_intervals},
                                       columns = ['predicted_val_lower',
                                                  'predicted_val_median',
                                                  'predicted_val_upper',
                                                  'actual_val',
                                                  'interval_width',
                                                  'is_inside_interval',
                                                  'mae_intervals'
                                                  ])
    
    return [pipe_lower, pipe_median, pipe_upper, r2_score, predicted_vs_actual]

def create_tsf_regression(n_estimators=200, min_interval=3):
    combiner = ColumnConcatenator()
    log.debug("Constructing TSForest regressor with n_estimators=%u", n_estimators)
    tsfr = combiner * TimeSeriesForestRegressor(n_estimators=n_estimators, n_jobs=-1)
    return tsfr

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

def create_tsfresh_windowed_regression_gbr(params, n_estimators, windowsize, res_samples_per_second, max_depth=3, learning_rate=0.1, loss="squared_error", alpha = 0.5, min_samples_leaf=1, min_samples_split=2):
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
    gregressor = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, alpha = alpha, loss=loss, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    pipeline = make_pipeline(t_features, StandardScaler(with_mean=False), gregressor)
    return pipeline

def create_tsfresh_windowed_regression_ridge(params, windowsize, res_samples_per_second, max_depth=3, learning_rate=0.1, loss="squared_error", alpha=0.5, max_alpha = 10, alpha_step=1.0):
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
    rregressor = RidgeCV(alphas=(0.1, alpha_step, max_alpha))
    pipeline = make_pipeline(t_features, StandardScaler(with_mean=False), rregressor)
    return pipeline

def create_minirocket_regression_gbr(params, num_kernels, n_estimators, max_depth=3, learning_rate=0.1, loss="squared_error", alpha = 0.5, min_samples_leaf=1, min_samples_split=2):
    t_features = MiniRocket(num_kernels=num_kernels,n_jobs=-1)
    gregressor = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, alpha=alpha, loss=loss, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    pipeline = make_pipeline(t_features, StandardScaler(with_mean=False), gregressor)
    return pipeline

def create_minirocket_regression_ridge(params, num_kernels, max_alpha = 10, alpha_step=1.0):
    t_features = MiniRocket(num_kernels=num_kernels,n_jobs=-1)
    rregressor = RidgeCV(alphas=(0.1, alpha_step, max_alpha))
    pipeline = make_pipeline(t_features, StandardScaler(with_mean=False), rregressor)
    return pipeline

def create_tsfresh_ngboost(params, n_estimators, windowsize, res_samples_per_second, max_depth=3, learning_rate=0.1, loss="squared_error", alpha = 0.5, min_samples_leaf=1, min_samples_split=2):
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
    learner_base = DecisionTreeRegressor(criterion='friedman_mse', max_depth=max_depth)
    ngboost_regressor = NGBRegressor(Dist=Normal, Score=CRPScore, verbose=False)
    #pipeline = make_pipeline(t_features, StandardScaler(with_mean=False), ngboost_regressor)
    return ngboost_regressor
#    return pipeline

def create_tsfresh_multioutput_chain(params, n_estimators, windowsize, res_samples_per_second, max_depth=3, learning_rate=0.1, loss="squared_error", alpha = 0.5, min_samples_leaf=1, min_samples_split=2):
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
    gregressor = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, alpha = alpha, loss=loss, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    # need to get order from the geometry of y
    order_TEST = [0,1,2]
    chain = RegressorChain(base_estimator=gregressor, order=order_TEST)
    pipeline = make_pipeline(t_features, StandardScaler(with_mean=False), chain)
    return pipeline
    
def create_tsfresh_windowed_classifier(params, n_estimators, windowsize, res_samples_per_second, max_depth=3):
    feature_name = "jrh_windowed_features_calculation_fixedsize"
    # TODO: try other features as well?
    settings = {}
    log.debug("create windowsize=" + str(windowsize))
    settings[feature_name] = {"windowsize" : windowsize, "resolution_samples_per_second" : res_samples_per_second }
    col_names = params["needed_columns"]

    features_selected = list(map(lambda col_name: col_name + "__" + feature_name + "__windowsize_" + str(windowsize) + "__resolution_samples_per_second_" + str(res_samples_per_second), col_names))

    for fn in features_selected:
        settings[fn] = {"windowsize" : windowsize}
    
    print("features_selected = " + str(features_selected))
    t_features = TSFreshFeatureExtractor(default_fc_parameters=settings, kind_to_fc_parameters=features_selected, show_warnings=False)
    gclassifier = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
    pipeline = make_pipeline(t_features, StandardScaler(with_mean=False), gclassifier)
    return pipeline

def create_tsfresh_windowed_classifier_rocketvars(params, n_estimators, windowsize, res_samples_per_second, num_kernels=100, max_depth=3):
    feature_name = "jrh_windowed_features_calculation_fixedsize"
    # TODO: try other features as well?
    settings = {}
    log.debug("create windowsize=" + str(windowsize))
    settings[feature_name] = {"windowsize" : windowsize, "resolution_samples_per_second" : res_samples_per_second }
    col_names_tsfresh = [c for c in params["needed_columns"] if c not in params["rocket_columns"]]
    col_names_rocket = params["rocket_columns"]
    
    features_selected = list(map(lambda col_name: col_name + "__" + feature_name + "__windowsize_" + str(windowsize) + "__resolution_samples_per_second_" + str(res_samples_per_second), col_names_tsfresh))

    for fn in features_selected:
        settings[fn] = {"windowsize" : windowsize}
    
    print("features_selected = " + str(features_selected))
    t_features = TransformerPipeline([ColumnSelect(col_names_tsfresh), TSFreshFeatureExtractor(default_fc_parameters=settings, kind_to_fc_parameters=features_selected, show_warnings=False)])
    rocket_features = TransformerPipeline([ColumnSelect(col_names_rocket), Rocket(num_kernels=num_kernels)])
    features = FeatureUnion([("TSFresh_Features", t_features), ("Rocket_Vars", rocket_features)], n_jobs=-1)
    
    gclassifier = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
    pipeline = make_pipeline(features, StandardScaler(with_mean=False), gclassifier)
    return pipeline

def create_rocket(num_kernels, max_alpha):
    rocket_pipeline = make_pipeline(Rocket(num_kernels=num_kernels, n_jobs=-1), StandardScaler(with_mean=False), RidgeCV(alphas=(0.1, 1.0, max_alpha))) 
    return rocket_pipeline

def create_rocket_with_regtrees(num_kernels, n_estimators, max_depth=3):
    gregressor = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth)
    rocket_pipeline = make_pipeline(Rocket(num_kernels=num_kernels, n_jobs=-1), StandardScaler(with_mean=False), gregressor)
    return rocket_pipeline

def create_mvts(num_heads=4, d_model=64, num_epochs=10):
    log.debug("num_heads=%u, d_model=%u" % (num_heads, d_model))
    model = MVTSTransformerClassifier(n_heads=num_heads, d_model=d_model, num_epochs=num_epochs)
    return model

def create_inceptiontime(n_epochs=30, batch_size=64, kernel_size=40):
    log.debug("batch_size=%u, kernel_size=%u" % (batch_size, kernel_size))
    model = InceptionTimeRegressor(n_epochs=n_epochs, batch_size=batch_size, kernel_size=kernel_size, verbose=True)
    return model

def plot_regression(regression_graph_title, predicted_vs_actual, expt_config, filename="regression.pdf"):
    print("length of dataframe:" + str(len(predicted_vs_actual)))
    plt.clf()
    plt.scatter(predicted_vs_actual["predicted_val"], predicted_vs_actual["actual_val"],marker='x')
    plt.axline((1,1),(2,2), marker="None", linestyle="dotted", color="Black")
    plt.xlabel("Predicted value of distance to sensitive point")
    plt.ylabel("Actual value of distance to sensitive point")

    plot_x_lower = expt_config["plot_x_lower"]
    plot_x_upper = expt_config["plot_x_upper"]
    plot_y_lower = expt_config["plot_y_lower"]
    plot_y_upper = expt_config["plot_y_upper"]
    
    plt.xlim([plot_x_lower, plot_x_upper])
    plt.ylim([plot_y_lower, plot_y_upper])
    plt.title(regression_graph_title)

    headers = ["predicted_val", "actual_val"]
    csv_filename = filename + "_rawdata.csv"
    predicted_vs_actual.to_csv(csv_filename, columns = headers)
    plt.savefig(filename)

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

    plot_x_lower = expt_config["plot_x_lower"]
    plot_x_upper = expt_config["plot_x_upper"]
    plot_y_lower = expt_config["plot_y_lower"]
    plot_y_upper = expt_config["plot_y_upper"]
    
#    plt.xlim([plot_x_lower, plot_x_upper])
#    plt.ylim([plot_y_lower, plot_y_upper])
    plt.title(regression_graph_title)
    plt.savefig(filename)

def plot_regression_intervals_original(regression_graph_title, predicted_vs_actual, expt_config, filename="regression.pdf"):
    # Plot with error bars
    print("length of dataframe:" + str(len(predicted_vs_actual)))
    plt.clf()
    error_above = np.abs(predicted_vs_actual["predicted_val_upper"] - predicted_vs_actual["predicted_val_median"])
    error_below = np.abs(predicted_vs_actual["predicted_val_median"] - predicted_vs_actual["predicted_val_lower"])
    plt.errorbar(predicted_vs_actual["predicted_val_median"], predicted_vs_actual["actual_val"], xerr=[error_below, error_above], marker='x', fmt="x", elinewidth=0.1)
    plt.axline((1,1),(2,2), marker="None", linestyle="dotted", color="Black")
    plt.xlabel("Predicted value of distance to sensitive point")
    plt.ylabel("Actual value of distance to sensitive point")

    plot_x_lower = expt_config["plot_x_lower"]
    plot_x_upper = expt_config["plot_x_upper"]
    plot_y_lower = expt_config["plot_y_lower"]
    plot_y_upper = expt_config["plot_y_upper"]
    
    plt.xlim([plot_x_lower, plot_x_upper])
    plt.ylim([plot_y_lower, plot_y_upper])
    plt.title(regression_graph_title)
    plt.savefig(filename)

def plot_confusion_matrix(predicted_vs_actual, filename="confusion.pdf", normalize=None):
    cm = confusion_matrix(predicted_vs_actual["actual_class"], predicted_vs_actual["predicted_class"], normalize=normalize)
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot()
    plt.savefig(filename)
    return None

def plot_param_variations_r2_score(data_3d, xlabel, ylabel, title, filename="r2_score_ranges.pdf"):
    plt.clf()
    ax = plt.figure().add_subplot(projection='3d')
    x = data_3d["param1"]
    y = data_3d["param2"]
    z = data_3d["r2_score_mean"]
    ax.errorbar(x, y, z, zerr=data_3d["r2_score_stddev"], linestyle="none", marker=".")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("r2 score")
    ax.set_zlim(0.0,1.0)
    ax.set_title(title)
    ax.view_init(20, 40, 0)
    headers = ["param1", "param2", "r2_score_mean", "r2_score_stddev"]
    csv_filename = filename + "_rawdata.csv"
    data_3d.to_csv(csv_filename, columns=headers)
    plt.savefig(filename)

def plot_confusion_matrix(predicted_vs_actual, filename="confusion.pdf", normalize=None):
    cm = confusion_matrix(predicted_vs_actual["actual_class"], predicted_vs_actual["predicted_class"], normalize=normalize)
    cmd = ConfusionMatrixDisplay(cm)
    cmd.plot()
    plt.savefig(filename)
    return None

def test_regression(name_base, id_code, result_desc, alg_name, alg_func, fig_filename_func, pd_res, expt_config, summary_res, alg_param1, alg_param2, k=5):
    params = {}

    params["base_dir"] = expt_config["data_dir_base"]
    params["target_metric_name"] = expt_config["target_metric_name"]
    params["needed_columns"] = expt_config["needed_columns"]

    mfile = expt_config["data_dir_base"] + "/metrics.csv"
    data_files_plus_decision_set, metrics_plus_decision_set = data_loader.read_data(expt_config["data_dir_base"], mfile)
    alg_func_delayed = lambda params: alg_func(alg_param1, alg_param2, params)

    decision_test_size = 0.2
    random_state = 23
    data_files, decision_test_files, metrics, decision_test_metrics = train_test_split(data_files_plus_decision_set, metrics_plus_decision_set, test_size = decision_test_size, random_state=random_state, shuffle=True)

    log.debug(f"data_files_plus_decision_set={len(data_files_plus_decision_set)}")
    log.debug(f"metrics_plus_decision_set={len(metrics_plus_decision_set)}")
    log.debug(f"data_files={len(data_files)}")
    log.debug(f"metrics={len(metrics)}")
    log.debug(f"decision_test_files={len(decision_test_files)}")
    log.debug(f"decision_test_metrics={len(decision_test_metrics)}")

    decision_test_metrics_filename = name_base + "-decisionTestMetrics-" + str(alg_param1) + "-" + str(alg_param2) + ".csv"
    decision_test_metrics.to_csv(decision_test_metrics_filename)

    k=5
    kf = KFold(n_splits=k, shuffle=k_fold_shuffle)

    # Accumulate these over all splits
    r2_score_all_splits = []
    mse_all_splits = []
    rmse_all_splits = []

    timediff_all_splits = []
    memoryused_all_splits = []

    
    for i, (train_index, test_index) in enumerate(kf.split(data_files)):
        # Split the data for k_fold validation
        params["data_files_train"] = [data_files[i] for i in train_index]
        params["metrics_train_pandas"] = metrics.iloc[train_index]
        params["data_files_test"] = [data_files[i] for i in test_index]
        params["metrics_test_pandas"] = metrics.iloc[test_index]
        log.debug(f"Fold {i}:")
        fig_filename = fig_filename_func(id_code, i)

        memory_tracker = MEMORY_TRACKING_CLASS()
        memory_tracker = MEMORY_TRACKING_CLASS()
        memory_tracker.start_tracking()
        time_start = timer()

#       pipeline, r2_score_from_reg, predicted_vs_actual = run_regression_or_classifier(True, alg_func_delayed, alg_name, params)
        pipeline, r2_score_from_reg, predicted_vs_actual = memory_tracker.with_tracking(lambda: run_regression_or_classifier(True, alg_func_delayed, alg_name, params))

        predictor_save_filename = name_base + expt_config["predictor_save_filename"] + "-split" + str(i) + "-" + str(alg_param1) + "-" + str(alg_param2) + ".predictor"
        data_loader.save_predictor_to_file(predictor_save_filename, pipeline)

        time_end = timer()
        time_diff = time_end - time_start
        memory_used = memory_tracker.end_tracking()
        
        mse_c = MeanSquaredError()
        rmse_c = MeanSquaredError(square_root=True)

        r2se = r2_score(predicted_vs_actual["actual_val"], predicted_vs_actual["predicted_val"], multioutput='uniform_average')

        mse = mse_c(predicted_vs_actual["actual_val"], predicted_vs_actual["predicted_val"])
        rmse = rmse_c(predicted_vs_actual["actual_val"], predicted_vs_actual["predicted_val"])

        log.debug("r2_score from regression run = %f, r2_score locally computed = %f", r2_score_from_reg, r2se)

        # Fix: needed to set multioutput='uniform_average'
        if abs(r2se - r2_score_from_reg) > 1e-6:
            log.error("Discrepancy between r2_score computed in pipeline and r2_score computed from sklearn")
            sys.exit(-1)

        r2_score_all_splits = np.append(r2_score_all_splits, r2se)
        mse_all_splits = np.append(mse_all_splits, mse)#
        rmse_all_splits = np.append(rmse_all_splits, rmse)
        timediff_all_splits = np.append(timediff_all_splits, time_diff)
        memoryused_all_splits = np.append(memoryused_all_splits, memory_used)

        results_this_test = {"id":id_code,
                             "result_desc":result_desc,
                             "k_split":i,
                             "param1":alg_param1,
                             "param2":alg_param2,
                             "r2_score":r2_score_from_reg,
                             "filename_graph":fig_filename,
                             "time_diff":time_diff,
                             "memory_used":memory_used,
                             "mse":mse,
                             "rmse":rmse }
        pd_res.loc[len(pd_res)] = results_this_test
        # change filename
        log.debug("Plotting regression plot to %s", fig_filename)
        plot_regression(expt_config["regression_graph_title"], predicted_vs_actual, expt_config, fig_filename)

    mean_r2 = np.mean(r2_score_all_splits)
    mean_mse = np.mean(mse_all_splits)
    mean_rmse = np.mean(rmse_all_splits)

    stddev_r2 = np.std(r2_score_all_splits)
    stddev_mse = np.std(mse_all_splits)
    stddev_rmse = np.std(rmse_all_splits)

    memory_used_mean = np.mean(memoryused_all_splits)
    memory_used_stddev = np.std(memoryused_all_splits)
    time_mean = np.mean(timediff_all_splits)
    time_stddev = np.std(timediff_all_splits)

    summary_this_test = {"id":id_code,
                         "result_desc":result_desc,
                         "param1":alg_param1,
                         "param2":alg_param2,
                         "r2_score_mean":mean_r2,
                         "mse_mean":mean_mse,
                         "rmse_mean":mean_rmse,
                         "r2_score_stddev":stddev_r2,
                         "mse_score_stddev":stddev_mse,
                         "rmse_score_stddev":stddev_rmse,
                         "memory_used_mean":memory_used_mean,
                         "time_mean":time_mean,
                         "memory_used_stddev":memory_used_stddev,
                         "time_stddev":time_stddev
                         }
    
    summary_res.loc[len(summary_res)] = summary_this_test

    log.info("Mean r2 all splits = %f, stddev r2 all splits = %f", mean_r2, stddev_r2)
    log.info("Mean MSE all splits = %f, stddev MSE all splits = %f", mean_mse, stddev_mse)
    log.info("Mean RMSE all splits = %f, stddev RMSE all splits = %f", mean_rmse, stddev_rmse)

    return pd_res, summary_res

def test_regression_intervals(id_code, alg_name, alg_func_lower, alg_func_median, alg_func_upper,
                              fig_filename_func, pd_res, expt_config, summary_res, alg_param1, alg_param2, k=5):
    params = {}

    params["base_dir"] = expt_config["data_dir_base"]
    params["target_metric_name"] = expt_config["target_metric_name"]
    params["needed_columns"] = expt_config["needed_columns"]    
    mfile = expt_config["data_dir_base"] + "/metrics.csv"
    data_files, metrics = data_loader.read_data(expt_config["data_dir_base"], mfile)

    alg_func_delayed_lower = lambda params: alg_func_lower(alg_param1, alg_param2, params)
    alg_func_delayed_median = lambda params: alg_func_median(alg_param1, alg_param2, params)
    alg_func_delayed_upper = lambda params: alg_func_upper(alg_param1, alg_param2, params)  
 
    k = 5
    kf = KFold(n_splits=k, shuffle=k_fold_shuffle)

    # Accumulate these over all splits
    r2_score_all_splits = []
    mse_all_splits = []
    rmse_all_splits = []
    
    for i, (train_index, test_index) in enumerate(kf.split(data_files)):
        # Split the data for k_fold validation
        params["data_files_train"] = [data_files[i] for i in train_index]
        params["metrics_train_pandas"] = metrics.iloc[train_index]
        params["data_files_test"] = [data_files[i] for i in test_index]
        params["metrics_test_pandas"] = metrics.iloc[test_index]
        log.debug(f"Fold {i}:")
        fig_filename = fig_filename_func(id_code, i)

        time_start = timer()
        pipeline_lower, pipeline_median, pipeline_upper, r2_score_from_reg, predicted_vs_actual = run_regression_intervals(alg_func_delayed_lower, alg_func_delayed_median, alg_func_delayed_upper, alg_name, params)

        time_end = timer()
        time_diff = time_end - time_start

        mse_c = MeanSquaredError()
        rmse_c = MeanSquaredError(square_root=True)

        # The proportion of values that are inside the interval
        print("is_inside_interval.sum (count inside) = ", predicted_vs_actual["is_inside_interval"].sum())
        print("count all =", len(predicted_vs_actual))
        
        in_interval_proportion = predicted_vs_actual["is_inside_interval"].sum() / len(predicted_vs_actual)
        interval_width_mean = np.mean(predicted_vs_actual["interval_width"])
        interval_width_stddev = np.std(predicted_vs_actual["interval_width"])
        mae_intervals = np.mean(predicted_vs_actual["mae_intervals"])

        results_this_test = {"id":id_code,
                             "k_split":i,
                             "param1":alg_param1,
                             "param2":alg_param2,
                             "filename_graph":fig_filename,
                             "time_diff":time_diff,
                             "in_interval_proportion":in_interval_proportion,
                             "interval_width_mean":interval_width_mean,
                             "interval_width_stddev":interval_width_stddev,
                             "mae":mae_intervals
                             }
        
        pd_res.loc[len(pd_res)] = results_this_test
        # change filename
        log.debug("Plotting regression plot to %s", fig_filename)
        plot_regression_intervals_new(expt_config["regression_graph_title"], predicted_vs_actual, expt_config, fig_filename)

    mean_r2 = np.mean(r2_score_all_splits)
    mean_mse = np.mean(mse_all_splits)
    mean_rmse = np.mean(rmse_all_splits)

    stddev_r2 = np.std(r2_score_all_splits)
    stddev_mse = np.std(mse_all_splits)
    stddev_rmse = np.std(rmse_all_splits)

    summary_this_test = {"param1":alg_param1,
                         "param2":alg_param2,
                         "r2_score_mean":mean_r2,
                         "mse_mean":mean_mse,
                         "rmse_mean":mean_rmse,
                         "r2_score_stddev":stddev_r2,
                         "mse_score_stddev":stddev_mse,
                         "rmse_score_stddev":stddev_rmse,                         
                         }
    summary_res.loc[len(summary_res)] = summary_this_test

    log.info("Mean r2 all splits = %f, stddev r2 all splits = %f", mean_r2, stddev_r2)
    log.info("Mean MSE all splits = %f, stddev MSE all splits = %f", mean_mse, stddev_mse)
    log.info("Mean RMSE all splits = %f, stddev RMSE all splits = %f", mean_rmse, stddev_rmse)

    return pd_res, summary_res


def test_classification(id_code, alg_name, alg_func, fig_filename_func, pd_res, expt_config, summary_res, alg_param1, alg_param2, k=5):
    params = {}
    
    params["base_dir"] = expt_config["data_dir_base"]
    params["target_metric_name"] = expt_config["target_metric_name"]
    params["needed_columns"] = expt_config["needed_columns"]
    params["class_count"] = None
    
    mfile = expt_config["data_dir_base"] + "/metrics.csv"
    data_files, metrics = data_loader.read_data(expt_config["data_dir_base"], mfile)

    alg_func_delayed = lambda params: alg_func(alg_param1, alg_param2, params)

    accuracy_top_limit = 2

    # Accumulate these over all splits
    accuracy_all_splits = []
    top_k_accuracy_all_splits = []
    rmse_all_splits = []

    k=5
    kf = KFold(n_splits=k, shuffle=k_fold_shuffle)
    
    for i, (train_index, test_index) in enumerate(kf.split(data_files)):
        params["data_files_train"] = [data_files[i] for i in train_index]
        params["metrics_train_pandas"] = metrics.iloc[train_index]
        params["data_files_test"] = [data_files[i] for i in test_index]
        params["metrics_test_pandas"] = metrics.iloc[test_index]

        time_start = timer()
        pipeline, accuracy_score, predicted_vs_actual, class_labels, class_probs = run_regression_or_classifier(False, alg_func_delayed, alg_name, params)
        time_end = timer()
        time_diff = time_end - time_start
        
        top_k_accuracy = 0.0
        log.debug("Classification accuracy = %f, top k accuracy this split = %f" % (accuracy_score, top_k_accuracy))
        # top k accuracy is broken because labels do not match 
#        top_k_accuracy = top_k_accuracy_score(predicted_vs_actual["actual_class"], class_probs, k=accuracy_top_limit, labels=class_labels, normalize=True)

        accuracy_all_splits = np.append(accuracy_all_splits, accuracy_score)
        top_k_accuracy_all_splits = np.append(top_k_accuracy_all_splits, top_k_accuracy)

        fig_filename = fig_filename_func(id_code, i)
                
        results_this_test = {"id":id_code, "k_split":i, "param1":alg_param1, "param2":alg_param2, "filename_graph":fig_filename, "time_diff":time_diff, "accuracy_score":accuracy_score, "top_k_accuracy_score":top_k_accuracy_score }
        pd_res.loc[len(pd_res)] = results_this_test

        #plot_regression(predicted_vs_actual, "fixedfuzzing_multimodels_twooperations_turtlebot.pdf")
        log.debug("Plotting confusion matrix to %s", fig_filename)
        plot_confusion_matrix(predicted_vs_actual, fig_filename)
        log.debug("Plot done")

    mean_accuracy = np.mean(accuracy_all_splits)
    min_accuracy = np.min(accuracy_all_splits)
    max_accuracy = np.max(accuracy_all_splits)
    
    mean_top_k_accuracy = np.mean(top_k_accuracy_all_splits)
    stddev_accuracy = np.std(accuracy_all_splits)
    stddev_top_k_accuracy = np.std(top_k_accuracy_all_splits)
    
    summary_this_test = {"param1":alg_param1, "param2":alg_param2, "mean_accuracy":mean_accuracy, "min_accuracy":min_accuracy, "max_accuracy":max_accuracy, "mean_top_k_accuracy": mean_top_k_accuracy, "stddev_accuracy": stddev_accuracy, "stddev_top_k_accuracy": stddev_top_k_accuracy}
    summary_res.loc[len(summary_res)] = summary_this_test
    return pd_res, summary_res


def log_latex_summary_results(stats_results, filename="summary-res.tex", sorted_by_col="r2_score_mean", limit=10):
    if sorted_by_col is None:
        sorted_results = stats_results
    else:
        sorted_results = stats_results.sort_values(by=sorted_by_col, ascending=False)

    if limit is None:
        selected_results = sorted_results
    else:
        selected_results = sorted_results.head(limit)
    latex_table = sorted_results.to_latex(index=False)  # index=False removes row indices
    with open(filename, 'w') as f:
        f.write(latex_table)
    log.info(f"LaTeX summary results saved to {filename}")

def run_test(name_base, expt_config, combined_results_all_tests, alg_name, regression, alg_func, alg_params1, alg_params2, param1_name, param2_name):
    # Put the directory here

    if regression:
        individual_results = pd.DataFrame(columns=["id", "param1", "param2", "k_split", "r2_score", "mse", "rmse", "filename_graph", "time_diff", "memory_used"])
        stats_results = pd.DataFrame(columns=["id", "result_desc", "param1", "param2", "r2_score_mean", "mse_mean", "rmse_mean", "r2_score_stddev", "mse_score_stddev", "rmse_score_stddev", "memory_used_mean", "time_mean", "memory_used_stddev", "time_stddev"])
        name_base = name_base + "regression"
    else:
        individual_results = pd.DataFrame(columns=["id", "param1", "param2", "k_split", "accuracy_score", "top_k_accuracy_score", "filename_graph", "time_diff"])
        stats_results = pd.DataFrame(columns=["param1", "param2", "mean_accuracy", "min_accuracy", "max_accuracy", "mean_top_k_accuracy", "stddev_accuracy", "stddev_top_k_accuracy"])
        name_base = name_base + "classification"
        
    results_file = name_base + "-" + alg_name + "-res.csv"
    summary_file = name_base + "-" + alg_name + "-summary-stats.csv"
    range_graph_file = name_base + "-" + alg_name + "-range-graph.pdf"
    range_graph_file_png = name_base + "-" + alg_name + "-range-graph.png"
    range_graph_title = expt_config["range_graph_title"]

    id_num = 0
    for param1 in alg_params1:
        for param2 in alg_params2:
            fig_filename_func = lambda id_num, k_split: name_base + "-" + alg_name + "-ID" + str(id_num) + "-" + str(param1) + "-" + str(param2) + "-" + "k_split" + str(k_split) +".pdf"
            id_num+=1
            id_code = "ID" + str(id_num) + param1_name + str(param1) + "_" + param2_name + str(param2)
            result_desc = alg_name + "_" + param1_name + "=" + str(param1) + "_" + param2_name + "_" + str(param2)
            if regression:
                individual_results, stats_results = test_regression(name_base, id_code, result_desc, alg_name, alg_func, fig_filename_func, individual_results, expt_config, stats_results, alg_param1=param1, alg_param2=param2)
            else:
                individual_results, stats_results = test_classification(id_code, result_desc, alg_name, alg_func, fig_filename_func, individual_results, expt_config, stats_results, alg_param1=param1, alg_param2=param2)
            print(tabulate(stats_results, headers="keys"))
            individual_results.to_csv(results_file, sep=",")
            print(tabulate(individual_results, headers="keys"))
            stats_results.to_csv(summary_file, sep=",")
            plot_param_variations_r2_score(stats_results, param1_name, param2_name, range_graph_title, filename=range_graph_file)
            plot_param_variations_r2_score(stats_results, param1_name, param2_name, range_graph_title, filename=range_graph_file_png)

    print(tabulate(stats_results, headers="keys"))
    individual_results.to_csv(results_file, sep=",")
    print(tabulate(individual_results, headers="keys")) 
    stats_results.to_csv(summary_file, sep=",")

    summary_file_latex = summary_file + ".tex"
    log_latex_summary_results(stats_results, sorted_by_col="r2_score_mean", limit=20, filename=summary_file_latex)

    plot_param_variations_r2_score(stats_results, param1_name, param2_name, range_graph_title, filename=range_graph_file)
    plot_param_variations_r2_score(stats_results, param1_name, param2_name, range_graph_title, filename=range_graph_file_png)

    if combined_results_all_tests is None:
        combined_results_all_tests = stats_results
    else:
        combined_results_all_tests = pd.concat([combined_results_all_tests, stats_results])
        combined_results_all_tests = combined_results_all_tests.sort_values(by=['r2_score_mean'], ascending=False)

    return combined_results_all_tests

def run_test_intervals(expt_config, alg_name, regression, alg_func_lower, alg_func_median, alg_func_upper, alg_params1, alg_params2, param1_name, param2_name):
    individual_results = pd.DataFrame(columns=["id", "param1", "param2", "k_split", "filename_graph", "time_diff", "in_interval_proportion", "interval_width_mean", "interval_width_stddev", "mae"])
    stats_results = pd.DataFrame(columns=["param1", "param2", "r2_score_mean", "mse_mean", "rmse_mean", "r2_score_stddev", "mse_score_stddev", "rmse_score_stddev"])
    name_base = "regression"

    results_file = name_base + "-" + alg_name + "-interval-res.csv"
    summary_file = name_base + "-" + alg_name + "-interval-summary-stats.csv"
    
    id_num = 0
    for param1 in alg_params1:
        for param2 in alg_params2:  
            fig_filename_func = lambda id_num, k_split: name_base + "-intervals-" + alg_name + "-ID" + str(id_num) + "-" + str(param1) + "-" + str(param2) + "-" + "k_split" + str(k_split) +".png"
            id_num+=1
            id_code = "ID" + str(id_num) + param1_name + str(param1) + "_" + param2_name + str(param2)
            individual_results, stats_results = test_regression_intervals(id_code, alg_name, alg_func_lower, alg_func_median, alg_func_upper,
                                                          fig_filename_func, individual_results, expt_config, stats_results, alg_param1=param1, alg_param2=param2)
            individual_results.to_csv(results_file, sep=",")
            print(tabulate(individual_results, headers="keys"))
    stats_results.to_csv(summary_file, sep=",")
    print(tabulate(stats_results, headers="keys"))

def confidence_intervals_from_normal_distributions(y_dist_test, interval_val=0.95):
    size = len(y_dist_test)
    print("size=", size)
    conf_intervals_lower = np.zeros(size)
    conf_intervals_upper = np.zeros(size)
    normal_means = np.zeros(size)
    for i in range(0,size):
        normal_dist_params = y_dist_test[i].params
        conf_int = stats.norm.interval(interval_val, loc=normal_dist_params["loc"], scale=normal_dist_params["scale"])
        print("first y_dist = " + str(normal_dist_params) + ",conf_int=" + str(conf_int))
        conf_intervals_lower[i] = conf_int[0]
        conf_intervals_upper[i] = conf_int[1]
        normal_means[i] = normal_dist_params["loc"]
    return conf_intervals_lower, conf_intervals_upper, normal_means

def test_regression_ngboost_intervals(id_code, alg_name, alg_func, fig_filename_func, pd_res, expt_config, summary_res, alg_param1, alg_param2, k=5):
    params = {}

    params["base_dir"] = expt_config["data_dir_base"]
    params["target_metric_name"] = expt_config["target_metric_name"]
    params["needed_columns"] = expt_config["needed_columns"]    
    mfile = expt_config["data_dir_base"] + "/metrics.csv"
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
        fig_filename = fig_filename_func(id_code, i)

        time_start = timer()
        # instead of single pipeline
        # take the feature extractor and produce the features, convert them to 2D X array
        # then create the 2D Y array of metrics per test
        # fit ngboost using the ngboost example
        # performance testing using this

        train_data = data_loader.create_combined_data(base_dir, data_files_train, needed_columns)
        test_data = data_loader.create_combined_data(base_dir, data_files_test, needed_columns)
        metrics_train = np.array(metrics_train_pandas[target_metric_name])
        metrics_test =  np.array(metrics_test_pandas[target_metric_name])
        log.debug("Check data format train_metrics: %s",check_raise(metrics_test, mtype="np.ndarray"))
        log.debug("Check data format test_metrics: %s",check_raise(metrics_train, mtype="np.ndarray"))

        # Need a custom pipeline to just generate the features here...
        # use create_tsfresh_windowed_featuresonly
        res_samples_per_second = 10
        n_estimators = 300
        window_size = 0.5

        confidence_intervals_param = 0.95
        
        features_pipe = create_tsfresh_windowed_featuresonly(params, n_estimators, window_size, res_samples_per_second)
        features_pipe.fit(train_data, metrics_train)
        x_features_train = features_pipe.transform(train_data)
        x_features_test = features_pipe.transform(test_data)

        dist = Normal
        ngb = NGBRegressor(Dist=dist, verbose=True, n_estimators=n_estimators, natural_gradient=True)
        ngb.fit(x_features_train, metrics_train, early_stopping_rounds=300)
        y_dist_test = ngb.pred_dist(x_features_test, max_iter=ngb.best_val_loss_itr)
        conf_intervals_lower, conf_intervals_upper, normal_mean = confidence_intervals_from_normal_distributions(y_dist_test, confidence_intervals_param)

        actual_val = metrics_test
        
        interval_width = conf_intervals_lower - conf_intervals_upper
        is_inside_interval = np.logical_and((actual_val >= conf_intervals_lower), (actual_val <= conf_intervals_upper))
        mae_intervals = calc_mae_from_intervals_all(conf_intervals_lower, conf_intervals_upper, actual_val, is_inside_interval)

        predicted_vs_actual = pd.DataFrame({'predicted_val_lower':conf_intervals_lower,
                                            'predicted_val_median':normal_mean,
                                            'predicted_val_upper':conf_intervals_upper,
                                            'actual_val':actual_val,
                                            'interval_width':interval_width,                                       
                                            'is_inside_interval':is_inside_interval,
                                            'mae_intervals':mae_intervals},
                                           columns = ['predicted_val_lower',
                                                      'predicted_val_median',
                                                      'predicted_val_upper',
                                                      'actual_val',
                                                      'interval_width',
                                                      'is_inside_interval',
                                                      'mae_intervals'
                                                      ])

        plot_regression_intervals_new(expt_config["regression_graph_title"], predicted_vs_actual, expt_config, fig_filename)

#        normal_dist_params = y_dist[0].params
#        conf_int = stats.norm.interval(0.95, loc=normal_dist_params["loc"], scale=normal_dist_params["scale"])
#        conf_int_lower = conf_int[0]
#        conf_int_upper = conf_int[1]
#        print("first y_dist = " + str(normal_dist_params) + ",lower=" + str(conf_int_lower) + ",upper=" + str(conf_int_upper))
#        normal_mean = y_dist[0].mean
#        normal_stddev = y_dist[0].std





        
        # how to get the interval from y_dist here?
        # https://www.reddit.com/r/MachineLearning/comments/il90ic/research_using_models_for_ngboost/

#       pipeline, r2_score_from_reg, predicted_vs_actual = run_regression_or_classifier(True, alg_func_delayed, alg_name, params)
        
        time_end = timer()
        time_diff = time_end - time_start

        mse_c = MeanSquaredError()
        rmse_c = MeanSquaredError(square_root=True)

        r2se = r2_score(predicted_vs_actual["actual_val"], predicted_vs_actual["predicted_val"], multioutput='uniform_average')

        mse = mse_c(predicted_vs_actual["actual_val"], predicted_vs_actual["predicted_val"])
        rmse = rmse_c(predicted_vs_actual["actual_val"], predicted_vs_actual["predicted_val"])

        log.debug("r2_score from regression run = %f, r2_score locally computed = %f", r2_score_from_reg, r2se)

        # Fix: needed to set multioutput='uniform_average'
        if abs(r2se - r2_score_from_reg) > 1e-6:
            log.error("Discrepancy between r2_score computed in pipeline and r2_score computed from sklearn")
            sys.exit(-1)

        r2_score_all_splits = np.append(r2_score_all_splits, r2se)
        mse_all_splits = np.append(mse_all_splits, mse)
        rmse_all_splits = np.append(rmse_all_splits, rmse)
         
        results_this_test = {"id":id_code, "k_split":i, "param1":alg_param1, "param2":alg_param2, "r2_score":r2_score_from_reg, "filename_graph":fig_filename, "time_diff":time_diff, "mse":mse, "rmse":rmse }
        pd_res.loc[len(pd_res)] = results_this_test
        # change filename
        log.debug("Plotting regression plot to %s", fig_filename)
        plot_regression(predicted_vs_actual, expt_config, fig_filename)

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

def run_test_ngboost_intervals(alg_name, expt_config):
    individual_results = pd.DataFrame(columns=["id", "param1", "param2", "k_split", "filename_graph", "time_diff", "in_interval_proportion", "interval_width_mean", "interval_width_stddev", "mae"])
    stats_results = pd.DataFrame(columns=["param1", "param2", "r2_score_mean", "mse_mean", "rmse_mean", "r2_score_stddev", "mse_score_stddev", "rmse_score_stddev"])
    name_base = "regression"
    id_code = 0

    # n_estimators
    n_estimators = 300
      # window_size
    windowsize = 20.0

    # min_samples_split
    param1 = 3
    # min_leaf_split

    fig_filename_func = lambda id_num, k_split: name_base + "-" + alg_name + "-ID" + str(id_num) + "-" + str(param1) + "-" + str(param2) + "-" + "k_split" + str(k_split) +".png"
    
    alg_func = lambda min_samples_split, min_samples_leaf, params: create_tsfresh_windowed_regression(params, n_estimators, windowsize, 10.0, max_depth=4, learning_rate=0.1,
                                                                                                            loss = "quantile", alpha = 0.05, min_samples_split=3, min_samples_leaf=min_samples_leaf)
    
    individual_results, stats_results = test_regression_ngboost_intervals(id_code, alg_name, alg_func, fig_filename_func, individual_results, expt_config, stats_results, alg_param1=param1, alg_param2=param2)
    
    results_file = name_base + "-" + alg_name + "-res.csv"
    summary_file = name_base + "-" + alg_name + "-summary-stats.csv"
    individual_results.to_csv(results_file, sep=",")
    stats_results.to_csv(summary_file, sep=",")
    print(tabulate(stats_results, headers="keys"))
    print(tabulate(individual_results, headers="keys"))



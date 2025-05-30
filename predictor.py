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

import datetime as dt
import numpy as np
import glob
import sys
import pandas as pd
import os
import psutil
import tracemalloc
import structlog 
from sktime.pipeline import make_pipeline
from tabulate import tabulate

log = structlog.get_logger()

USE_TRACEMALLOC = False
k_fold_shuffle = True

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def load_individual_instance(filename, needed_columns):
    df = pd.read_csv(filename)
    for col in needed_columns:
        if not (col in df.columns):
            df[col] = 0.0
            # Ensure all the columns are in the correct order!
    return df[needed_columns]

def create_combined_data(base_dir, filenames, needed_columns):
    combined_data_m = map(lambda file: load_individual_instance(base_dir + "/" + file, needed_columns), filenames)
    combined_data = list(combined_data_m)
    print("Check data: ",check_raise(combined_data, mtype="df-list"))
    return combined_data

def run_regression_or_classifier(regression, pipeline_gen_func, alg_name, params):
    base_dir = params["base_dir"]
    data_files_train = params["data_files_train"]
    data_files_test = params["data_files_test"]
    metrics_train_pandas = params["metrics_train_pandas"]
    metrics_test_pandas = params["metrics_test_pandas"]
    target_metric_name = params["target_metric_name"]
    needed_columns = params["needed_columns"]
    alg_name = params["target_metric_name"]
        
    train_data = create_combined_data(base_dir, data_files_train, needed_columns)
    test_data = create_combined_data(base_dir, data_files_test, needed_columns)
    metrics_train = np.array(metrics_train_pandas[target_metric_name])
    metrics_test =  np.array(metrics_test_pandas[target_metric_name])
    log.debug("Check data format train_metrics: %s",check_raise(metrics_test, mtype="np.ndarray"))
    log.debug("Check data format test_metrics: %s",check_raise(metrics_train, mtype="np.ndarray"))

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
        
    train_data = create_combined_data(base_dir, data_files_train, needed_columns)
    test_data = create_combined_data(base_dir, data_files_test, needed_columns)
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

def create_hivecote2():
    hive_cote = HIVECOTEV2()
    return hive_cote

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
#   print(predicted_vs_actual.to_markdown())
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

    # TODO: Save predicted/actual to a raw log file
    headers = ["predicted_val", "actual_val"]
    csv_filename = filename + "_rawdata.csv"
    predicted_vs_actual.to_csv(csv_filename, columns = headers)
    plt.savefig(filename)

def plot_regression_intervals(predicted_vs_actual, expt_config, filename="regression.pdf"):
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
    plt.title("Predicted vs actual object position error for Mycobot with four joints fuzzed")
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

def read_data(results_directory, mfile):
    data_files = list(map(os.path.basename, sorted(glob.glob(results_directory + "/*Test*"))))
    metrics = pd.read_csv(mfile)
    return data_files, metrics

def test_regression_intervals(id_code, alg_name, alg_func_lower, alg_func_median, alg_func_upper,
                              fig_filename_func, pd_res, expt_config, summary_res, alg_param1, alg_param2, k=5):
    params = {}

    params["base_dir"] = expt_config["data_dir_base"]
    params["target_metric_name"] = expt_config["target_metric_name"]
    params["needed_columns"] = expt_config["needed_columns"]    
    mfile = expt_config["data_dir_base"] + "/metrics.csv"
    data_files, metrics = read_data(expt_config["data_dir_base"], mfile)

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

#        r2_score_all_splits = np.append(r2_score_all_splits, r2se)
#        mse_all_splits = np.append(mse_all_splits, mse)
#        rmse_all_splits = np.append(rmse_all_splits, rmse)
         
        results_this_test = {"id":id_code,
                             "k_split":i,
                             "param1":alg_param1,
                             "param2":alg_param2,
                             "filename_graph":fig_filename,
                             "time_diff":time_diff,
#                             "mse":mse,
#                             "rmse":rmse,
                             "in_interval_proportion":in_interval_proportion,
                             "interval_width_mean":interval_width_mean,
                             "interval_width_stddev":interval_width_stddev,
                             "mae":mae_intervals
                             }
        
        pd_res.loc[len(pd_res)] = results_this_test
        # change filename
        log.debug("Plotting regression plot to %s", fig_filename)
        plot_regression_intervals(predicted_vs_actual, expt_config, fig_filename)

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

def test_regression(id_code, result_desc, alg_name, alg_func, fig_filename_func, pd_res, expt_config, summary_res, alg_param1, alg_param2, k=5):
    params = {}

    params["base_dir"] = expt_config["data_dir_base"]
    params["target_metric_name"] = expt_config["target_metric_name"]
    params["needed_columns"] = expt_config["needed_columns"]

    mfile = expt_config["data_dir_base"] + "/metrics.csv"
    data_files, metrics = read_data(expt_config["data_dir_base"], mfile)
    alg_func_delayed = lambda params: alg_func(alg_param1, alg_param2, params)
 
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

        # https://www.geeksforgeeks.org/monitoring-memory-usage-of-a-running-python-program/
        if USE_TRACEMALLOC:
            tracemalloc.start()
        
        time_start = timer()
        pipeline, r2_score_from_reg, predicted_vs_actual = run_regression_or_classifier(True, alg_func_delayed, alg_name, params)
        
        time_end = timer()
        time_diff = time_end - time_start

        if USE_TRACEMALLOC:
            memory_used = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        else:
            memory_used = 0

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
    timediff_mean = np.mean(timediff_all_splits)

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
                         "timediff_mean":timediff_mean
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
    data_files, metrics = read_data(expt_config["data_dir_base"], mfile)

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

def run_test(expt_config, combined_results_all_tests, alg_name, regression, alg_func, alg_params1, alg_params2, param1_name, param2_name):
    if regression:
        individual_results = pd.DataFrame(columns=["id", "param1", "param2", "k_split", "r2_score", "mse", "rmse", "filename_graph", "time_diff", "memory_used"])
        stats_results = pd.DataFrame(columns=["id", "result_desc", "param1", "param2", "r2_score_mean", "mse_mean", "rmse_mean", "r2_score_stddev", "mse_score_stddev", "rmse_score_stddev", "memory_used_mean", "timediff_mean"])
        name_base = "regression"
    else:
        individual_results = pd.DataFrame(columns=["id", "param1", "param2", "k_split", "accuracy_score", "top_k_accuracy_score", "filename_graph", "time_diff"])
        stats_results = pd.DataFrame(columns=["param1", "param2", "mean_accuracy", "min_accuracy", "max_accuracy", "mean_top_k_accuracy", "stddev_accuracy", "stddev_top_k_accuracy"])
        name_base = "classification"
        
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
                individual_results, stats_results = test_regression(id_code, result_desc, alg_name, alg_func, fig_filename_func, individual_results, expt_config, stats_results, alg_param1=param1, alg_param2=param2)
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
    plot_param_variations_r2_score(stats_results, param1_name, param2_name, range_graph_title, filename=range_graph_file)
    plot_param_variations_r2_score(stats_results, param1_name, param2_name, range_graph_title, filename=range_graph_file_png)

    if combined_results_all_tests is None:
        combined_results_all_tests = stats_results
    else:
        combined_results_all_tests = pd.concat([combined_results_all_tests, stats_results])
        combined_results_all_tests = combined_results_all_tests.sort_values(by=['r2_score_mean'], ascending=False)

    return combined_results_all_tests

def run_test_intervals(alg_name, expt_config):
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
    param2 = 1
    
    fig_filename_func = lambda id_num, k_split: name_base + "-" + alg_name + "-ID" + str(id_num) + "-" + str(param1) + "-" + str(param2) + "-" + "k_split" + str(k_split) +".png"
    
    alg_func_lower = lambda min_samples_split, min_samples_leaf, params: create_tsfresh_windowed_regression(params, n_estimators, windowsize, 10.0, max_depth=4, learning_rate=0.1,
                                                                                                            loss = "quantile", alpha = 0.05, min_samples_split=3, min_samples_leaf=min_samples_leaf)
    
    alg_func_median = lambda min_samples_split, min_samples_leaf, params: create_tsfresh_windowed_regression(params, n_estimators, windowsize, 10.0, max_depth=4, learning_rate=0.1,
                                                                                                  loss = "quantile", alpha = 0.5, min_samples_split=3, min_samples_leaf=min_samples_leaf)
    
    alg_func_upper = lambda min_samples_split, min_samples_leaf, params: create_tsfresh_windowed_regression(params, n_estimators, windowsize, 10.0, max_depth=4, learning_rate=0.1,
                                                                                                 loss = "quantile", alpha = 0.95, min_samples_split=3, min_samples_leaf=min_samples_leaf)
    
    individual_results, stats_results = test_regression_intervals(id_code, alg_name, alg_func_lower, alg_func_median, alg_func_upper,
                                                          fig_filename_func, individual_results, expt_config, stats_results, alg_param1=param1, alg_param2=param2)
    
    results_file = name_base + "-" + alg_name + "-res.csv"
    summary_file = name_base + "-" + alg_name + "-summary-stats.csv"
    individual_results.to_csv(results_file, sep=",")
    stats_results.to_csv(summary_file, sep=",")
    print(tabulate(stats_results, headers="keys"))
    print(tabulate(individual_results, headers="keys"))

def test_regression_ngboost(id_code, alg_name, alg_func, fig_filename_func, pd_res, expt_config, summary_res, alg_param1, alg_param2, k=5):
    params = {}

    params["base_dir"] = expt_config["data_dir_base"]
    params["target_metric_name"] = expt_config["target_metric_name"]
    params["needed_columns"] = expt_config["needed_columns"]    
    mfile = expt_config["data_dir_base"] + "/metrics.csv"
    data_files, metrics = read_data(expt_config["data_dir_base"], mfile)

    alg_func_delayed = lambda params: alg_func(alg_param1, alg_param2, params)
 
    k=5
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
        # instead of single pipeline
        # take the feature extractor and produce the features, convert them to 2D X array
        # then create the 2D Y array of metrics per test
        # fit ngboost using the ngboost example
        # performance testing using this

        train_data = create_combined_data(base_dir, data_files_train, needed_columns)
        test_data = create_combined_data(base_dir, data_files_test, needed_columns)
        metrics_train = np.array(metrics_train_pandas[target_metric_name])
        metrics_test =  np.array(metrics_test_pandas[target_metric_name])
        log.debug("Check data format train_metrics: %s",check_raise(metrics_test, mtype="np.ndarray"))
        log.debug("Check data format test_metrics: %s",check_raise(metrics_train, mtype="np.ndarray"))
        
        features_pipe = pipeline_gen_func(params)
        features_pipe.fit(train_data, metrics_train)
        features = features_pipe.transform(train_data)

        # n_estimators needs to be larger?
        dist = MultivariateNormal(2)
        ngb = NGBRegressor(Dist=dist, verbose=True, n_estimators=2000, natural_gradient=True)
        ngb.fit(X, Y, X_val=X_val, Y_val=Y_val, early_stopping_rounds=100)
        y_dist = ngb.pred_dist(X, max_iter=ngb.best_val_loss_itr)
        
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

def run_test_ngboost(alg_name, expt_config):
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
    param2 = 1
    
    fig_filename_func = lambda id_num, k_split: name_base + "-" + alg_name + "-ID" + str(id_num) + "-" + str(param1) + "-" + str(param2) + "-" + "k_split" + str(k_split) +".png"
    
    alg_func_lower = lambda min_samples_split, min_samples_leaf, params: create_tsfresh_windowed_regression(params, n_estimators, windowsize, 10.0, max_depth=4, learning_rate=0.1,
                                                                                                            loss = "quantile", alpha = 0.05, min_samples_split=3, min_samples_leaf=min_samples_leaf)
    
    alg_func_median = lambda min_samples_split, min_samples_leaf, params: create_tsfresh_windowed_regression(params, n_estimators, windowsize, 10.0, max_depth=4, learning_rate=0.1,
                                                                                                  loss = "quantile", alpha = 0.5, min_samples_split=3, min_samples_leaf=min_samples_leaf)
    
    alg_func_upper = lambda min_samples_split, min_samples_leaf, params: create_tsfresh_windowed_regression(params, n_estimators, windowsize, 10.0, max_depth=4, learning_rate=0.1,
                                                                                                 loss = "quantile", alpha = 0.95, min_samples_split=3, min_samples_leaf=min_samples_leaf)
    
    individual_results, stats_results = test_regression_intervals(id_code, alg_name, alg_func_lower, alg_func_median, alg_func_upper,
                                                          fig_filename_func, individual_results, expt_config, stats_results, alg_param1=param1, alg_param2=param2)
    
    results_file = name_base + "-" + alg_name + "-res.csv"
    summary_file = name_base + "-" + alg_name + "-summary-stats.csv"
    individual_results.to_csv(results_file, sep=",")
    stats_results.to_csv(summary_file, sep=",")
    print(tabulate(stats_results, headers="keys"))
    print(tabulate(individual_results, headers="keys"))

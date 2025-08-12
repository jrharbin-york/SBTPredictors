import analyse_pareto_fronts
import datasets

pred_eterry_base_path = "./for-aggregation-results/chosen-predictors/predictors/eterry"
pred_multiturtlebot_base_path = "./for-aggregation-results/chosen-predictors/predictors/multiturtlebot"
pred_mycobot_base_path = "./for-aggregation-results/chosen-predictors/predictors/mycobot"

eterry_file_options = [
    # top choices for approaches
    {     "Human1_Pred" :       "regressionETERRY-Human1DistTSFreshWin_GradBoost-eterry-human1-dist-split0-50-0.5.predictor",           # TSFreshWin_GradBoost_50_0.5
          "StaticHumans_Pred" : "regressionETERRY-StaticHumanDist_TSForest-eterry-statichumans-dist-split0-300-10.0.predictor",      # TSForest_300_10.0
          "PathCompletion_Pred":"regressionETERRY-PathCompletionTSFreshWin_GradBoost-eterry-pathcompletion-split0-150-0.5.predictor",     # TSFreshWin_GradBoost_150_0.5,
          "decisionMetrics" : "eterry-decisionTestMetrics.csv",
          "result_filename" : "eterry-choice1-decisions.csv"
    },

    # second choices for approaches
    {    "Human1_Pred": "regressionETERRY-Human1Dist_MiniRocket_Ridge-eterry-human1-dist-split0-500-20.predictor",                            # MiniRocket_Ridge_500_20.0
         "StaticHumans_Pred": "regressionETERRY-StaticHumanDistTSFreshWin_GradBoost-eterry-statichumans-dist-split0-150-0.5.predictor",       # TSFreshWin_GradBoost_150_0.5
         "PathCompletion_Pred": "regressionETERRY-PathCompletion_TSForest-eterry-pathcompletion-split0-50-1.0.predictor",                      # TSForest_50_1.0
         "decisionMetrics": "eterry-decisionTestMetrics.csv",
         "result_filename" : "eterry-choice2-decisions.csv"
    },

    # third choices for approaches
    {    "Human1_Pred": "regressionETERRY-Human1Dist_MiniRocket_GradBoost-eterry-human1-dist-split0-1000-50.predictor",            # MiniRocket_GradBoost_1000_50
         "StaticHumans_Pred": "regressionETERRY-StaticHumanDist_MiniRocket_GradBoost-eterry-statichumans-dist-split0-2000-150.predictor",       # MiniRocket_GradBoost_2000_150
         "PathCompletion_Pred": "regressionETERRY-PathCompletion_MiniRocket_GradBoost-eterry-pathcompletion-split0-500-20.predictor",      # MiniRocket_GradBoost_500_20.0
         "decisionMetrics": "eterry-decisionTestMetrics.csv",
         "result_filename" : "eterry-choice3-decisions.csv"
    }
]

eterry_metric_columns_direction = { "distanceToHuman1" : "min",
                                    "distanceToStaticHumans" : "min",
                                    "pathCompletion": "min" }

eterry_static_thresholds = { "distanceToHuman1" : 3.0,
                             "distanceToStaticHumans" : 2.0,
                             "pathCompletion": 0.6 }

eterry_distance_divisor_per_metric = {
        "distanceToHuman1": 10,
        "distanceToStaticHumans": 4,
        "pathCompletion": 1
    }

eterry_metric_weights = {
        "distanceToHuman1": 1.0,
        "distanceToStaticHumans": 1.0,
        "pathCompletion": 1.0
}

# TB1   1 & 500 & 150.000 & MiniRocket\_GradBoost & 0.773 & 0.024 & 0.239 & 0.026 & 0.489 & 0.027 \\
# TB2   1 & 2000 & 300.000 & MiniRocket\_GradBoost & 0.813 & 0.015 & 0.298 & 0.026 & 0.546 & 0.024 \\

multiturtlebot_file_options = [
    {
        "distanceTB1Away_Pred": "regressionMultiturtlebot-TB1_MiniRocket_GradBoost-turtlebot-tb1-dist-split0-500-150.predictor",
        "distanceTB2Away_Pred": "regressionMultiturtlebot-TB2_MiniRocket_GradBoost-turtlebot-tb2-dist-split0-2000-300.predictor",
        "decisionMetrics": "turtlebot-decisionTestMetrics.csv",
        "result_filename": "multiturtlebot-choice1-decisions.csv"
    },
]

tb_metric_columns_direction = { "distanceTB1Away" : "max",
                                "distanceTB2Away" : "max"
                                 }

tb_static_thresholds = { "distanceTB1Away" : 2.0,
                         "distanceTB2Away" : 2.0
                       }

tb_distance_divisor_per_metric = {
    "distanceTB1Away" : 1.0,
    "distanceTB2Away" : 1.0
}

tb_metric_weights = {
    "distanceTB1Away" : 1.0,
    "distanceTB2Away" : 1.0
}

######################################################################################################################################

mycobot_file_options = [
    {
        "objectFinalPositionError_Pred": "regressionMycobot-Fourjoints-ObjectPosition_TSForest-mycobot-param-mag-split4-300-0.5.predictor",
        "paramMagnitudes_Pred": "regressionMycobot-Fourjoints-ParamMagTSFreshWin_GradBoost-mycobot-param-mag-split0-300-2.predictor",
        "decisionMetrics": "mycobot-decisionTestMetrics.csv",
        "result_filename": "mycobot-choice1-decisions.csv"
    },
]

mycobot_metric_columns_direction = {
    "objectFinalPositionError" : "max",
    "paramMagnitudes" : "min"
}

mycobot_static_thresholds = {
    "objectFinalPositionError": 1.0,
    "paramMagnitudes": 2.0
}

mycobot_distance_divisor_per_metric = {
    "objectFinalPositionError" : 1.0,
    "paramMagnitudes" : 1.0
}

mycobot_metric_weights = {
    "objectFinalPositionError" : 1.0,
    "paramMagnitudes" : 1.0
}

def run_analysis_different_fronts(run_eterry=False, run_tb=False, run_mycobot=True):
    if run_mycobot:
        for pred_files in mycobot_file_options:
            analyse_pareto_fronts.test_evaluate_predictor_decisions_for_experiment(datasets.expt_config_mycobot_fourjoints_bothmetrics_objectposition,
                                                                                    pred_mycobot_base_path,
                                                                                    pred_files,
                                                                                    mycobot_metric_columns_direction,
                                                                                    mycobot_static_thresholds,
                                                                                    mycobot_distance_divisor_per_metric,
                                                                                    mycobot_metric_weights)

    if run_eterry:
        for pred_files in eterry_file_options:
            analyse_pareto_fronts.test_evaluate_predictor_decisions_for_experiment(datasets.expt_config_eterry_human1_15files,
                                                                                   pred_eterry_base_path,
                                                                                   pred_files, eterry_metric_columns_direction, eterry_static_thresholds, eterry_distance_divisor_per_metric, eterry_metric_weights)

    if run_tb:
        for pred_files in multiturtlebot_file_options:
            analyse_pareto_fronts.test_evaluate_predictor_decisions_for_experiment(datasets.expt_config_turtlebot_multi_tb1_server_allops, pred_multiturtlebot_base_path, pred_files, tb_metric_columns_direction, tb_static_thresholds, tb_distance_divisor_per_metric, tb_metric_weights)


if __name__ == '__main__':
    run_analysis_different_fronts(run_eterry=False, run_tb=True)
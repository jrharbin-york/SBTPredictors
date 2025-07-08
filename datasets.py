################################################################################
## Mycobot
################################################################################
mycobot_fourjoints_3000 = "./input_csv_data/mycobot-fourjoints-3000"

mycobot_all_ops = ["distortJoint0", "distortJoint1", "distortJoint2", "distortJoint3"]

expt_config_mycobot_fourjoints_3000 = {
    "dataset_name" : "Mycobot-Fourjoints",
    "data_dir_base" : mycobot_fourjoints_3000,
    "target_metric_name" : "objectFinalPositionError",
    "needed_columns" : mycobot_all_ops,
    "plot_x_lower" : 0.0,
    "plot_x_upper" : 1.5,
    "plot_y_lower" : 0.0,
    "plot_y_upper" : 1.5,
    "regression_graph_title" : "Predicted vs actual cylinder position error for Mycobot case",
    "range_graph_title" : "r2 score for Mycobot case - across different hyperparameters",
    "predictor_save_filename" : "mycobot-error-dist",
    "use_fixed_windows" : True
} 

################################################################################
# ETERRY new June 2025
################################################################################
data_dir_base_eterry_human1_1100 = "./input_csv_data/eterry-tenfiles-human1"
data_dir_base_eterry_statichumans_1100 = "./input_csv_data/eterry-tenfiles-statichumans"
data_dir_base_eterry_pathcompletion_1100 = "./input_csv_data/eterry-tenfiles-pathcompletion"
data_dir_base_eterry_cutdown = "./input_csv_data/eterry-cutdown-predtest"

eterry_ops = ["deletePointCloud", "delayPointCloud", "changeHumanXStep_zero", "humanPos2Offset", "humanPos3Offset"]

expt_config_eterry_human1_1100 = {
    "dataset_name" : "ETERRY-Human1Dist",
    "data_dir_base" : data_dir_base_eterry_human1_1100,
    "target_metric_name" : "distanceToHuman1", 
    "needed_columns" : eterry_ops,
    "plot_x_lower" : 0.0,
    "plot_x_upper" : 10.0,
    "plot_y_lower" : 0.0,
    "plot_y_upper" : 10.0,
    "regression_graph_title" : "Predicted vs actual human-robot distance for mobile human (human1)",
    "regression_graph_x" : "Predicted value of the ETERRY-human1 distance",
    "regression_graph_y" : "Actual value of the ETERRY-human1 distance",
    "range_graph_title" : "r2 score for ETERRY case across different hyperparameters",
    "predictor_save_filename" : "eterry-human1-dist",
    "use_fixed_windows" : False
}

expt_config_eterry_statichumans_1100 = {
    "dataset_name" : "ETERRY-StaticHumanDist",
    "data_dir_base" : data_dir_base_eterry_statichumans_1100,
    "target_metric_name" : "distanceToStaticHumans",
    "needed_columns" : eterry_ops,
    "plot_x_lower" : 0.0,
    "plot_x_upper" : 4.0,
    "plot_y_lower" : 0.0,
    "plot_y_upper" : 4.0,
    "regression_graph_title" : "Predicted vs actual human-robot distance for static humans (human2+human3)",
    "regression_graph_x": "Predicted value of the ETERRY-humans distance",
    "regression_graph_y": "Actual value of the ETERRY-humans distance",
    "range_graph_title" : "r2 score for ETERRY case across different hyperparameters",
    "predictor_save_filename" : "eterry-statichumans-dist",
    "use_fixed_windows" : False
}

expt_config_eterry_pathcompletion_1100 = {
    "dataset_name" : "ETERRY-PathCompletion",
    "data_dir_base" : data_dir_base_eterry_pathcompletion_1100,
    "target_metric_name" : "pathCompletion", 
    "needed_columns" : eterry_ops,
    "plot_x_lower" : 0.0,
    "plot_x_upper" : 1.0,
    "plot_y_lower" : 0.0,
    "plot_y_upper" : 1.0,
    "regression_graph_title" : "Predicted vs actual path completion for ETERRY robot",
    "regression_graph_x": "Predicted value of the path completion",
    "regression_graph_y": "Actual value of the path completion",
    "range_graph_title" : "r2 score for ETERRY case across different hyperparameters",
    "predictor_save_filename" : "eterry-pathcompletion",
    "use_fixed_windows" : False
}

expt_config_eterry_cutdown = {
    "dataset_name" : "ETERRY-StaticHumanDist",
    "data_dir_base" : data_dir_base_eterry_cutdown,
    "target_metric_name" : "distanceToStaticHumans",
    "needed_columns" : eterry_ops,
    "plot_x_lower" : 0.0,
    "plot_x_upper" : 4.0,
    "plot_y_lower" : 0.0,
    "plot_y_upper" : 4.0,
    "regression_graph_title" : "Predicted vs actual human-robot distance for static humans (human2+human3)",
    "regression_graph_x": "Predicted value of the ETERRY-humans distance",
    "regression_graph_y": "Actual value of the ETERRY-humans distance",
    "range_graph_title" : "r2 score for ETERRY case across different hyperparameters",
    "predictor_save_filename" : "eterry-statichumans-dist",
    "use_fixed_windows" : False
}

################################################################################
# ETERRY old
################################################################################
data_dir_base_eterry_humandist_1800 = "./input_csv_data/eterry-1800-humandist"
data_dir_base_eterry_robotdist_1800 = "./input_csv_data/eterry-1800-robotdist"

eterry_old_ops = ["deletePointCloud", "changeHumanXStep"]

##### ETERRY HUMANDIST1
expt_config_eterry_humandist_1800 = {
    "dataset_name" : "ETERRY-HumanDist",
    "data_dir_base" : data_dir_base_eterry_humandist_1800,
    "target_metric_name" : "distanceToHuman1", 
    "needed_columns" : eterry_old_ops,
    "plot_x_lower" : 0.0,
    "plot_x_upper" : 9.0,
    "plot_y_lower" : 0.0,
    "plot_y_upper" : 9.0,
    "regression_graph_title" : "Predicted vs actual cylinder position error for Mycobot case",
    "range_graph_title" : "r2 score for Mycobot case - across different hyperparameters",
    "use_fixed_windows" : False
}

##### ETERRY ROBOTDIST_AT_END
expt_config_eterry_robotdist_1800 = {
    "dataset_name" : "ETERRY-RobotDist",
    "data_dir_base" : data_dir_base_eterry_robotdist_1800,
    "target_metric_name" : "robotDistanceAtEnd", 
    "needed_columns" : eterry_old_ops,
    "plot_x_lower" : 0.0,
    "plot_x_upper" : 22.0,
    "plot_y_lower" : 0.0,
    "plot_y_upper" : 22.0,
    "regression_graph_title" : "Predicted vs actual cylinder position error for Mycobot case",
    "range_graph_title" : "r2 score for Mycobot case - across different hyperparameters",
    "use_fixed_windows" : False
}

################################################################################
# Multi-turtlebot
################################################################################
data_dir_base_turtlebot_tb1 = "./input_csv_data/turtlebot-tb1-3000"
data_dir_base_turtlebot_tb2 = "./input_csv_data/turtlebot-tb2-3000"

turtlebot_all_ops = ["distortVelocity_tb1", "delete_tb1", "delay_tb1", "distortVelocity_tb2", "delete_tb2", "delay_tb2"]

##### TURTLEBOT TB1 dist
expt_config_turtlebot_multi_tb1_server_allops = {
    "dataset_name" : "Multiturtlebot-TB1",
    "data_dir_base" : data_dir_base_turtlebot_tb1,
    "target_metric_name" : "distanceTB1Away", 
    "needed_columns" : turtlebot_all_ops,
    "plot_x_lower" : 0.0,
    "plot_x_upper" : 4.0,
    "plot_y_lower" : 0.0,
    "plot_y_upper" : 4.0,
    "regression_graph_title" : "Predicted vs actual robot position error for TB1 Turtlebot",
    "range_graph_title" : "r2 score for Turtlebot TB1",
    "use_fixed_windows" : False,
    "predictor_save_filename" : "turtlebot-tb1-dist",
}

##### TURTLEBOT TB2 dist
expt_config_turtlebot_multi_tb2_server_allops = {
    "dataset_name" : "Multiturtlebot-TB2",
    "data_dir_base" : data_dir_base_turtlebot_tb2,
    "target_metric_name" : "distanceTB1Away",
    "needed_columns" : turtlebot_all_ops,
    "plot_x_lower" : 0.0,
    "plot_x_upper" : 4.0,
    "plot_y_lower" : 0.0,
    "plot_y_upper" : 4.0,
    "regression_graph_title" : "Predicted vs actual robot position error for TB1 Turtlebot",
    "range_graph_title" : "r2 score for Turtlebot TB1",
    "use_fixed_windows" : False,
    "predictor_save_filename" : "turtlebot-tb2-dist",
}
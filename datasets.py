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
    "use_fixed_windows" : True
} 

################################################################################
# ETERRY
################################################################################
data_dir_base_eterry_humandist_1800 = "./input_csv_data/eterry-1800-humandist"
data_dir_base_eterry_robotdist_1800 = "./input_csv_data/eterry-1800-robotdist"

eterry_ops = ["deletePointCloud", "changeHumanXStep"]

##### ETERRY HUMANDIST1
expt_config_eterry_humandist_1800 = {
    "dataset_name" : "ETERRY-HumanDist",
    "data_dir_base" : data_dir_base_eterry_humandist_1800,
    "target_metric_name" : "distanceToHuman1", 
    "needed_columns" : eterry_ops,
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
    "needed_columns" : eterry_ops,
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
# data_dir_base_turtlebot_tb2 = "./input_csv_data/turtlebot-multi-multimodels-tb2-runsall"

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
    "regression_graph_title" : "Predicted vs actual robot position error for TB1 Turlebot",
    "range_graph_title" : "r2 score for Turtlebot TB1",
    "use_fixed_windows" : False
}

# #### TURTLEBOT TB2 dist
# expt_config_turtlebot_multi_tb2_server_allops = { "data_dir_base" : data_dir_base_turtlebot_multi_tb2_runsall,
#                                                   "target_metric_name" : "distanceTB2Away", 
#                                                   "needed_columns" : turtlebot_all_ops,
#                                                   "plot_x_lower" : 0.0,
#                                                   "plot_x_upper" : 4.0,
#                                                   "plot_y_lower" : 0.0,
#                                                   "plot_y_upper" : 4.0,
#                                           }

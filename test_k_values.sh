#!/bin/sh

k_min=4
k_max=7

#for use_case_metric in Turtlebot_TB1 Turtlebot_TB2 ETerry_Human1 ETerry_StaticHumans ETerry_PathCompletion
for use_case_metric in ETerry_StaticHumans ETerry_PathCompletion
do
#   for approach in TSFreshWin_GradBoost TSFreshWin_Ridge TSForest MiniRocket_GradBoost MiniRocket_Ridge
    for approach in TSForest
    do
	python3 ./run_memory_tracking_or_ks.py $approach $use_case_metric NONE $k_min $k_max
    done
done

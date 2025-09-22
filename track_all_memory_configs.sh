#!/bin/sh

for use_case_metric in Mycobot Turtlebot_TB1 Turtlebot_TB2 ETerry_Human1 ETerry_StaticHumans
do
    for approach in TSFreshWin_GradBoost TSFreshWin_Ridge TSForest MiniRocket_GradBoost MiniRocket_Ridge
    do
	fil-profile run ./run_memory_tracking_or_ks.py $approach $use_case_metric /home/jharbin/academic/soprano/SBTPredictors/memory-results/
    done
done

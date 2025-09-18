import run_all_algs
import datasets

run_eterry = False
run_mycobot = True
run_turtlebot = False

run_intervals = False

default_k = 5

if __name__ == '__main__':
    # Run multiple algs
    #if run_intervals:
    #    run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_human1_15files, run_intervals=True, run_non_intervals=False)
    #    # For running the single predictor
    #    run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_pathcompletion_15files, run_intervals=True, run_non_intervals=False)

    if run_mycobot:
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_mycobot_fourjoints_bothmetrics_objectposition, using_inceptiontime=False, run_non_intervals=True, run_intervals=False, k=default_k )
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_mycobot_fourjoints_bothmetrics_parammag, using_inceptiontime=False, run_non_intervals=True, run_intervals=False, k=default_k)

    if run_turtlebot:
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_turtlebot_multi_tb1_server_allops, using_inceptiontime = False, k=default_k)
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_turtlebot_multi_tb2_server_allops, using_inceptiontime = False, k=default_k)

    if run_eterry:
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_human1_15files, using_inceptiontime=False, k=default_k)
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_statichumans_15files, using_inceptiontime = False, k=default_k)
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_pathcompletion_15files, using_inceptiontime = False, k=default_k)

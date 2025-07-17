import run_all_algs
import datasets

run_eterry = False
run_mycobot = True
run_turtlebot = False

if __name__ == '__main__':
    if run_mycobot:
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_mycobot_fourjoints_3000, using_inceptiontime = False)

    if run_turtlebot:
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_turtlebot_multi_tb1_server_allops, using_inceptiontime = False)
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_turtlebot_multi_tb2_server_allops, using_inceptiontime = False)

    if run_eterry:
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_human1_15files, using_inceptiontime=False)
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_statichumans_15files, using_inceptiontime = False)
        run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_pathcompletion_15files, using_inceptiontime = False)

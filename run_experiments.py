import run_all_algs
import datasets

if __name__ == '__main__':
    #run_all_algs.run_all_algs_on_dataset(datasets.expt_config_mycobot_fourjoints_3000, using_inceptiontime = False)
    #run_all_algs.run_all_algs_on_dataset(datasets.expt_config_turtlebot_multi_tb1_server_allops, using_inceptiontime = False)

    # TODO: update ETERRY when new run finishes
    #run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_human1_15files, using_inceptiontime=False)
    #run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_statichumans_15files, using_inceptiontime = False)
    run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_pathcompletion_15files, using_inceptiontime = False)

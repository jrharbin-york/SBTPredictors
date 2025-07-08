import run_all_algs
import datasets

if __name__ == '__main__':
    run_all_algs.run_all_algs_on_dataset(datasets.expt_config_mycobot_fourjoints_3000, using_inceptiontime = False)
    # Turtlebot values needed
    #run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_human1_1100, using_inceptiontime=False)
    #run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_statichumans_1100, using_inceptiontime = False)
    #run_all_algs.run_all_algs_on_dataset(datasets.expt_config_eterry_pathcompletion_1100, using_inceptiontime = False)

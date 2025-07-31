# Settings for the algorithms

learning_rates=[0.1,0.05,0.03, 0.2, 0.3]

rocket_kernel_choices = [100,500,1000,2000]

full_choices = False

n_estimator_choices = [1000]
window_size_choices_secs= [0.5]
min_samples_split = [2]
max_alphas = [10]
alpha_step_values=[1.0]

if full_choices:
    n_estimator_choices = [300, 150, 50, 20]
    window_size_choices_secs= [0.5,1.0,2.0,5.0,10.0]
    tsf_min_intervals = [3]
    max_alphas = [5,10,20]
    alpha_step_default = 1.0
    alpha_step_values=[1.0,2.0,3.0]
    min_samples_split=[2,3,4]

inceptiontime_batch_sizes=[64]
#inceptiontime_kernel_sizes=[40,20,80]
inceptiontime_kernel_sizes=[40]
inceptiontime_n_epochs=[200,1500]

min_samples_split_default=2
min_samples_leaf_default=1

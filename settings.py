# Settings for the algorithms

learning_rates=[0.1,0.05,0.03, 0.2, 0.3]

rocket_kernel_choices = [100,500,1000,2000]

###### QUICK TEST SETTINGS
n_estimator_choices=[300]
window_size_choices_secs=[0.5]

full_choices = True

if full_choices:
    n_estimator_choices = [300, 150, 50, 20]
    window_size_choices_secs= [0.5,1.0,2.0,5.0,10.0]

tsf_min_intervals = [3]
max_alphas = [5,10,20]
alpha_step_default = 1.0
alpha_step_values=[1.0,2.0,3.0]

mvts_num_heads = [4,3,2,1]
mvts_d_model = [32,64,128]

inceptiontime_batch_sizes=[64]
#inceptiontime_kernel_sizes=[40,20,80]
inceptiontime_kernel_sizes=[40]
inceptiontime_n_epochs=[200,1500]

min_samples_split=[2,3,4]
min_samples_split_default=2
min_samples_leaf_default=1

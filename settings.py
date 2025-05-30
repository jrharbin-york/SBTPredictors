# Settings for the algorithms

#n_estimator_choices = [5,20,50,100,200,300]
#window_size_choices_secs = [0.5,1,2,5,10]
#max_depth_choices=[2,3,4,5,6,7,8]

# TSF also uses n_estimator
#tsf_min_intervals = [1,2,3,5,10,20] # 3 is the default

# Rocket params: number of kernels and maximum alpha
#rocket_kernels = [100,1000,5000,10000,50000]
#max_alphas = [5,10,20,50]

learning_rates=[0.1,0.05,0.03, 0.2, 0.3]

rocket_kernel_choices = [100,2000]

###### QUICK TEST SETTINGS

n_estimator_choices=[300,200,100,50,25,5]
window_size_choices_secs=[0.5,1.0,2.0,5.0,10.0,20.0]
#n_estimator_choices=[300]
#window_size_choices_secs=[0.5]

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

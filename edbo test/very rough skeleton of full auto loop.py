import numpy as np
import matplotlib.pyplot as plt
import edbo
import pandas as pd
import copy
from data_anal_tools import 
from monitor_hplc_folder import HPLC_watch
from gpytorch.priors import GammaPrior
from edbo.pd_utils import 

# init bo

domain     			 = 'a_pd_dataframe'
target 				 = 'yield' # give a name to the DataFrame column
acquisition_function = 'EI'
init_method          ='rand'
lengthscale_prior    = [GammaPrior(1.2, 1.2), 0.2] # GP prior and initial value
noise_prior 		 = None
batch_size 			 = 1 # number of exps proposed during each iteration

bo = edbo.bro.BO(domain=X, 
	             target=target, 
	             acquisition_function=acquisition_function,
	             init_method=init_method,
	             lengthscale_prior=lengthscale_prior, 
	             noise_prior=noise_prior, 
	             batch_size=batch_size, 
	             fast_comp=False, # not using gpytorch to accelerate computation
	             computational_objective=None) 

# propose the first experiment 
init_prop = bo.init_sample()

############################## 

# codes that translate this proposal 
# into commands for the flow system to excuate

##############################

watch = HPLC_watch(watch_dir='', file_type='REPORT03.CSV')
first_res = watch.run()
utils.appendResToEdbo(bo, first_res)

############################## 

# codes that tell hplc to stop sampling 
# and prepare for next exp

############################## 

# need to figure out what is the stop condition
while True: 

	bo.run()

	new_prop = bo.proposed_experiments

	############################## 

	# codes that translate this proposal 
	# into commands for the flow system to excuate

	##############################

	watch = HPLC_watch(watch_dir='', file_type='REPORT03.CSV')
	new_res = watch.run()
	utils.appendResToEdbo(bo, new_res)

	############################## 

	# codes that tell hplc to stop sampling 
	# and prepare for next exp

	############################## 
	
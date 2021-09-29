import sys
import edbo
import numpy as np
import pandas as pd
from gpytorch.priors import GammaPrior
from monitor_hplc_folder import HPLC_watch
from data_anal_tools import appendResToEdbo, propose_exp, populate_design_space, extract_data_from_csv, standardize_domain

largvs = len(sys.argv)

if largvs==1 or (largvs==2 and sys.argv[1][-3:]=='csv'): 

    # check existing results
    try: 
        resutls = sys.argv[1]
    except: 
        results = None

    # define design space
    VARIABLES = {
        # (<start>, <end>, <step>)
        'temperature': (-10, 30, 2), # C
        'dosing_duration': (0, 60, 5), # mins
        'NaBH4': (0.1, 2, 0.1), # equiv  
        'SM': (1, 10, 1) # mM
    }

    arr_list = []
    name_list = []

    for variable, (start, end, increment) in VARIABLES.items():
        # arange excludes the last value, so add the increment to it to ensure it's included
        arr_list.append(np.arange(start, end + increment, increment))
        name_list.append(variable)

    # not standardized
    domain = pd.DataFrame(populate_design_space(arr_list, name_list))

    #standardize
    std_domain = standardize_domain(domain, VARIABLES)

    # bo params
    target 				 = 'yields' # give a name to the DataFrame column
    acquisition_function = 'EI'
    init_method          = 'rand'
    batch_size 			 = 1 # number of exps proposed during each iteration
    lengthscale_prior    = [GammaPrior(1.2, 1.1), 0.2] # GP prior and initial value
    noise_prior          = [GammaPrior(1.2, 1.1), 0.2]

    # folder info
    dir_to_watch         = '.'
    files_to_watch       = ['']

    # exp info
    round_               = 0
    total_exps           = 2

    # init bo
    bo = edbo.bro.BO(results_path = results, 
                     domain=domain, 
                     target=target, 
                     acquisition_function=acquisition_function,
                     init_method=init_method,
                     lengthscale_prior=lengthscale_prior, 
                     noise_prior=noise_prior, 
                     batch_size=batch_size, 
                     fast_comp=False, # not using gpytorch to accelerate computation
                     computational_objective=None) 

    # propose the first experiment 
    _ = bo.init_sample(append=False)

else:

    # when error occurs, we restart from the previous exp
    bo = edbo.bro.BO()
    bo.load()

# main experiment loop

while True:
    # proposed exp: standardized
    bo.proposed_experiments.index.values
    # to get the original exp conditions
    domain.iloc[bo.proposed_experiments.index.values]

    # propose_exp(bo)
    # replace with lukes code to interact with kinova

    watch = HPLC_watch(watch_dir=dir_to_watch, file_type=files_to_watch)
    new_res = watch.run()
    appendResToEdbo(bo, new_res)

    # save results
    bo.obj.scaler.unstandardize_target(bo.obj.results, bo.obj.target).to_csv('results.csv')

    bo.run()
    bo.save()
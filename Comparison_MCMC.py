from simulation import MASsimulation
import numpy as np
from utils import *
import concurrent.futures
import time
from synthetis import ControlEstimationSynthesis
import pickle
import os

def run_simulation(args):
    sim = MASsimulation(args)
    # The constructor of MASsimulation starts the simulation automatically, no need to call run_sim here
    return sim.eval.get_results()

def count_completed_tasks(futures):
    # Count the number of completed tasks
    return sum(1 for future in futures if future.done())

def mcmc_simulatoin(num_simulations = 100, args = None, ctrl_type  = CtrlTypes.LQROutputFeedback):
    
    max_concurrent_processes = 10  # Define the maximum number of concurrent processes

    # Create a list of argument dictionaries for each simulation
    args_list = []
    for _ in range(num_simulations):
        
       
        args['ctrl_type'] = ctrl_type
        
        args_list.append(args)
    obj = MASsimulation(args)
    # Create a list to store the evaluation results for each run    
    results_list = []

    # Create a ThreadPool with a maximum number of threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_processes) as executor:
        # Submit each simulation job to the ThreadPool
        futures = [executor.submit(run_simulation, args) for args in args_list]

        # Periodically check the number of completed tasks
        while count_completed_tasks(futures) < num_simulations:
            print(f"Completed tasks: {count_completed_tasks(futures)} / {num_simulations}")
            time.sleep(1)  # Wait for a short interval before checking again

        # Collect the results from completed futures
        for future in futures:
            results_list.append(future.result())

    ########## Collect data from simulations ##############
    stage_costs = [result['stage_cost'] for result in results_list]
    np_stage_costs = np.stack(stage_costs).squeeze()
    avg_stage_costs = np.mean(np_stage_costs, axis=0)
    
    trajs = [result['trajs'] for result in results_list]    
    np_trajs = np.stack(trajs).squeeze()
    avg_trajs = np.mean(np_trajs, axis=0)
    
    est_trajs = [result['est_trajs'] for result in results_list]
    np_est_trajs = np.stack(est_trajs).squeeze()
   
    avg_est_trajs = np.mean(np_est_trajs, axis=0)
    
    result = {'stage_cost' : np_stage_costs, 
              'avg_trajs' : avg_trajs}
    return result
    

# for agent in range(avg_est_trajs.shape[0]):
#     plt.plot(range(avg_est_trajs.shape[1]), avg_est_trajs[agent,:,3], label=f'Agent {agent + 1}')
# plt.plot(avg_trajs[0,:,3], linestyle='--')
# # Set plot labels and title

# # Add a legend
# plt.legend()

# # Show the plot
# plt.show()

def get_fully_connected_args(args):
    new_args = args.copy()
    new_args['c'] =  np.ones([args['N'],args['N']]) # adjencency matrix 
    new_args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix             
    new_args['gain_file_name'] = 'fully' + str(args['N'])
    return new_args
    


if __name__ == "__main__":
    
    lqr_results = []
    lqg_results = []
    opt_results = []
    sub_results = []
    comlqg_results = []
    comlqg_5_results = []
    fullyconnected_synthesis_list = []
    partial_synthesis_list = []

    num_simulations = 500  # Define the number of parallel simulations
    args = {}        
    args['sim_n_step'] = 300
    args['n'] = 4
    args['p'] = 2
    args['Ts'] = 0.1   
    args['ctrl_type'] = 0
    args['gamma'] = 1
    w_std = 1.0     
    v_std = 1.0   
    
    # args['c'] = np.ones([N_agent,N_agent]) # adjencency matrix 

    num_agent_list = [5]
    for idx, N_agent in enumerate(num_agent_list):
               
        args['N'] = N_agent
        args['w_std'] = w_std  # w std for each agent 
        args['v_std'] = np.ones([N_agent,1])*v_std# v std for each agent.     
        # args['v_std'][0] = args['v_std'][0]*5            
        # args['c'] = get_chain_adj_mtx(N_agent) 
        args['c'] = get_circular_adj_mtx(N_agent) 
        args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix             
        args['Q'] = np.kron(args['L'], np.eye(args['n'])) # np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])    
        args['R'] = np.eye(N_agent)
        

        fullyconnected_args = get_fully_connected_args(args.copy())
        fullyconnected_args['gain_file_name'] = 'lqg' + str(args['N'])
        fullyconnected_args['ctrl_type'] = CtrlTypes.LQGFeedback
        fullyconnected_synthesis = ControlEstimationSynthesis(fullyconnected_args)
        fullyconnected_synthesis_list.append(fullyconnected_synthesis)


        sub_args = args.copy()
        sub_args['gain_file_name'] = 'sub' + str(args['N'])
        sub_args['ctrl_type'] = CtrlTypes.SubOutpFeedback
        sub_synthesis = ControlEstimationSynthesis(sub_args)
        
        comglqg_args = args.copy()
        comglqg_args['gain_file_name'] = 'comlqg' + str(args['N'])
        comglqg_args['ctrl_type'] = CtrlTypes.COMLQG
        comglqg_args['gamma'] = 1
        comglqg_synthesis = ControlEstimationSynthesis(comglqg_args)


        comglqg_5_args = args.copy()
        comglqg_5_args['gain_file_name'] = 'comlqg_5' + str(args['N'])
        comglqg_5_args['ctrl_type'] = CtrlTypes.COMLQG
        comglqg_5_args['gamma'] = 5
        comglqg_5_synthesis = ControlEstimationSynthesis(comglqg_5_args)


        partial_args = args.copy()
        partial_args['gain_file_name'] = 'ctrlest' +str(args['N'])    
        partial_args['ctrl_type'] = CtrlTypes.CtrlEstFeedback
        partial_synthesis = ControlEstimationSynthesis(partial_args)
        partial_synthesis_list.append(partial_synthesis)
        
        # LQROutputFeedback = 0        
        # SubOutpFeedback = 1 
        # CtrlEstFeedback = 2 
        # LQGFeedback 3   
        
        # lqr_result = mcmc_simulatoin(num_simulations, fullyconnected_args,CtrlTypes.LQROutputFeedback)
        # lqr_results.append(lqr_result) 
        # print('LQR with {} agents Done'.format(N_agent))

        lqg_result = mcmc_simulatoin(num_simulations, fullyconnected_args,CtrlTypes.LQGFeedback)
        lqg_results.append(lqg_result) 
        print('LQR with {} agents Done'.format(N_agent))

        

        sub_result = mcmc_simulatoin(num_simulations, sub_args,CtrlTypes.SubOutpFeedback)
        sub_results.append(sub_result)  
        print('Suboptimal with {} agents Done'.format(N_agent))

        comlqg_result = mcmc_simulatoin(num_simulations, comglqg_args,CtrlTypes.COMLQG)
        comlqg_results.append(comlqg_result)  
        print('Comlqg with {} agents Done'.format(N_agent))
    
        comlqg_5_result = mcmc_simulatoin(num_simulations, comglqg_5_args,CtrlTypes.COMLQG)
        comlqg_5_results.append(comlqg_5_result)  
        print('Comlqg_5 with {} agents Done'.format(N_agent))



        opt_result = mcmc_simulatoin(num_simulations, partial_args,CtrlTypes.CtrlEstFeedback)  
        opt_results.append(opt_result)
        
        print('Opt with {} agents Done'.format(N_agent))
        print('MCMCs with {} agents Done'.format(N_agent))
               
        
        
        

    save_data = {}
    save_data['args'] = args
    save_data['num_agent_list'] = num_agent_list    
    # save_data['lqr_results'] = lqr_results
    save_data['lqg_results'] = lqg_results
    save_data['opt_results'] = opt_results
    save_data['sub_results'] = sub_results
    save_data['comlqg_results'] = comlqg_results
    save_data['comlqg_5_results'] = comlqg_5_results
    # save_data['partial_synthesis_list'] = partial_synthesis_list
    # save_data['fullyconnected_synthesis_list'] = fullyconnected_synthesis_list
    
    
    save_file_name = 'mcmc_experiment_' +str(1) + str('.pkl')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')  # Create a 'data' directory in the same folder as your script
    file_path = os.path.join(data_dir, save_file_name)  # Create a 'data' directory in the same folder as your script
    if os.path.exists(file_path):
        file_path = file_path.split('.pkl')[0]+'_copy.pkl'
    with open(file_path, 'wb') as file:
         pickle.dump(save_data,file)
    

    plot_comparison_result(lqg_results, sub_results, opt_results, comlqg_results, comlqg_5_results)
    

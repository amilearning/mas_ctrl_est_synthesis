from simulation import MASsimulation
import numpy as np
from utils import *
import concurrent.futures
import time
from synthetis import ControlEstimationSynthesis
def run_simulation(args):
    sim = MASsimulation(args)
    # The constructor of MASsimulation starts the simulation automatically, no need to call run_sim here
    return sim.eval.get_results()

def count_completed_tasks(futures):
    # Count the number of completed tasks
    return sum(1 for future in futures if future.done())

def mcmc_simulatoin(args, ctrl_type : CtrlTypes):
    num_simulations = 20  # Define the number of parallel simulations
    max_concurrent_processes = 10  # Define the maximum number of concurrent processes

    # Create a list of argument dictionaries for each simulation
    args_list = []
    for _ in range(num_simulations):
        # N_agent = 5
        # args = {}        
        # args['Ts'] = 0.1       
        # args['N'] = N_agent
        # args['w_std'] = 0.1 # w std for each agent 
        # args['v_std'] = np.ones([N_agent,1])*0.1 # v std for each agent.     
        # args['v_std'][0] = 1        
        # args['c'] = np.ones([N_agent,N_agent]) # adjencency matrix 
        # args['c'] = get_chain_adj_mtx(N_agent) 
        # args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix             
        # args['n'] = 4
        # args['p'] = 2
        # args['Q'] = args['L'] # np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])
        # args['R'] = np.eye(N_agent)
        # args['sim_n_step'] = 200
       
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
 
    

if __name__ == "__main__":
    N_agent = 5
    args = {}        
    args['Ts'] = 0.1       
    args['N'] = N_agent
    args['w_std'] = 0.1 # w std for each agent 
    args['v_std'] = np.ones([N_agent,1])*0.1 # v std for each agent.     
    args['v_std'][0] = 1        
    args['c'] = np.ones([N_agent,N_agent]) # adjencency matrix 
    args['c'] = get_chain_adj_mtx(N_agent) 
    args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix             
    args['n'] = 4
    args['p'] = 2
    args['Q'] = np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])
    args['R'] = np.eye(N_agent)
    args['sim_n_step'] = 100
    args['ctrl_type'] = 0
    synthesis = ControlEstimationSynthesis(args)

    # LQROutputFeedback = 0
    # SubOutpFeedback = 1 
    # CtrlEstFeedback = 2    
   
    lqr_result = mcmc_simulatoin(args,CtrlTypes.LQROutputFeedback)
    sub_result = mcmc_simulatoin(args,CtrlTypes.SubOutpFeedback)
    opt_result = mcmc_simulatoin(args,CtrlTypes.CtrlEstFeedback)
    
    plot_comparison_result(lqr_result, sub_result, opt_result)
    

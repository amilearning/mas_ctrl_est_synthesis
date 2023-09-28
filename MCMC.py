from simulation import MASsimulation
import multiprocessing
import numpy as np
from utils import *


def run_simulation(args, results_list):
    sim = MASsimulation(args)
    # The constructor of MASsimulation starts the simulation automatically, no need to call run_sim here
    
    # Append the eval results to the results_list
    results_list.append(sim.eval.get_results())
    

if __name__ == "__main__":
    num_simulations = 5  # Define the number of parallel simulations

    # Create a list of argument dictionaries for each simulation
    args_list = []
    for _ in range(num_simulations):
        args = {}
        args['Ts'] = 0.1
        N_agent = 5
        args['N'] = N_agent
        args['w_std'] = 0.1 # w std for each agent 
        args['v_std'] = np.ones([N_agent,1])*0.1 # v std for each agent.     
        args['c'] = np.ones([N_agent,N_agent]) # adjencency matrix 
        # args['c'] = np.array([[1,1,0,0,0],
        #                       [1,1,1,0,0],
        #                       [0,1,1,1,0],
        #                       [0,0,1,1,1],
        #                       [0,0,0,1,1]])
        args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix     
        args['n'] = 4
        args['p'] = 2
        args['Q'] = np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])
        args['R'] = np.eye(N_agent)
        
        args_list.append(args)

    # Create a list to store the evaluation results for each run
    manager = multiprocessing.Manager()
    results_list = manager.list()  # Use a manager list to store results


    # Create and start separate processes for each simulation
    processes = []
    for args in args_list:
        process = multiprocessing.Process(target=run_simulation, args=(args, results_list))
        process.start()
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.join()

    stage_costs = [result['stage_cost'] for result in results_list]
    np_stage_costs = np.stack(stage_costs).squeeze()
    avg_stage_costs = np.mean(np_stage_costs,axis=0)
    plot_stage_cost(avg_stage_costs)
    trajs = [result['trajs'] for result in results_list]
    np_trajs = np.stack(trajs).squeeze()
    avg_trajs = np.mean(np_trajs,axis=0)
    plot_mas_traj(avg_trajs)
"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from pycrazyswarm import Crazyswarm
import numpy as np
from mas_ws.formation_experiment import CrazyFormation
from mas_ws.utils import *


def main():
    swarm = Crazyswarm()
    args = {}
    args['mission_duration'] = 5
    args['height'] = 1.0
    args['Hz'] = 20
    args['Ts'] = 1/ args['Hz']
    
    N_agent = 5
    args['N'] = N_agent

    args['n'] = 2
    args['p'] = 2

   
    args['w_std'] = 0.05 # w std for each agent 
    args['v_std'] = np.ones([N_agent,1])*0.1 # v std for each agent.     
    args['v_std'][0] = 0.5
    
    args['gamma'] = 1
    # LQROutputFeedback = 0
    # SubOutpFeedback = 1 
    # CtrlEstFeedback = 2
    # LQGFeedback = 3 
    # COMLQG = 4
    args['ctrl_type'] = CtrlTypes.LQROutputFeedback

    if args['ctrl_type'] == CtrlTypes.LQGFeedback:
        args['c'] = np.ones([N_agent,N_agent]) # adjencency matrix 
    else:
        args['c'] = get_chain_adj_mtx(N_agent) 
        # args['c'] = get_circular_adj_mtx(N_agent) 
    args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix     
    
    args['Q'] = np.kron(args['L'], np.eye(args['n'])) #  np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])
    args['R'] = np.eye(N_agent)

    formation_test1 = CrazyFormation(swarm,args)
   
   
if __name__ == "__main__":

    main()

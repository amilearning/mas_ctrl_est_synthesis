"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

from pycrazyswarm import Crazyswarm
import numpy as np
import numpy as np
import math
from mas_ws.agent import Agent
from mas_ws.simulation import MASsimulation
from mas_ws.utils import *
import control as ct
from mas_ws.synthetis import ControlEstimationSynthesis
import threading
import concurrent.futures

enable_debug_mode = False


TAKEOFF_DURATION = 1.5
HOVER_DURATION = 2.0


def main():


    args = {}
    args['Ts'] = 0.1
    N_agent = 5
    args['N'] = N_agent
    args['w_std'] = 0.1 # w std for each agent 
    args['v_std'] = np.ones([N_agent,1])*0.1 # v std for each agent.     
    # args['v_std'][0] = 1.0
    # args['c'] = np.ones([N_agent,N_agent]) # adjencency matrix 
    args['c'] = get_chain_adj_mtx(N_agent) 
    # args['c'] = get_circular_adj_mtx(N_agent) 
    args['gamma'] = 1
    # args['c'] = np.array([[1,1,0,0,0],
    #                       [1,1,1,0,0],
    #                       [0,1,1,1,0],
    #                       [0,0,1,1,1],
    #                       [0,0,0,1,1]])
    args['L'] = get_laplacian_mtx(args['c']) # Laplacian matrix     
    args['n'] = 4
    args['p'] = 2
    args['Q'] = np.kron(args['L'], np.eye(args['n'])) #  np.eye(N_agent)*N_agent-np.ones([N_agent,N_agent])
    # np.array([[1,0,0,0],
    #                        [0,0,0,0],
    #                        [0,0,1,0],
    #                        [0,0,0,0]])
    # args['Q'] = args['L']
    args['R'] = np.eye(N_agent)
    args['sim_n_step'] = 200
    args['gain_file_name'] = '22'

    # LQROutputFeedback = 0
    # SubOutpFeedback = 1 
    # CtrlEstFeedback = 2
    # COMLQG = 3
    args['ctrl_type'] = CtrlTypes.CtrlEstFeedback

    obj = MASsimulation(args)
    obj.eval.eval_init()


    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=0.4, duration=TAKEOFF_DURATION)
    timeHelper.sleep(5.0)
    cf.cmdVelocityWorld(np.array([0.1, 0.0, 0.0]), 0.0)
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    timeHelper.sleep(2.0)    
    timeHelper.sleep(2.0)
    # cf.takeoff(1.0, 2.5)    
    
    timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    cf.land(targetHeight=0.04, duration=2.5)
    timeHelper.sleep(TAKEOFF_DURATION)


if __name__ == "__main__":
    main()

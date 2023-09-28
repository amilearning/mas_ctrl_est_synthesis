import numpy as np
import math
from utils import *

   
class MASEval:
    # Constructor method (optional)
    def __init__(self, args):
        # Initialize instance variables here
        self.args = args

        self.stage_costs = []        
        self.trajs = []
        self.xhats = []
    def add_stage_cost(self,cost):
        self.stage_costs.append(cost.copy())
        
    def eval_init(self):
        plot_x_y(self.trajs)
        plot_stage_cost(self.stage_costs)
        
    def get_results(self):
        results = {}
        results['stage_cost'] = self.stage_costs
        results['trajs'] = self.trajs
        results['xhats'] = self.xhats
        return results
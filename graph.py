import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations,groupby 
import networkx as nx
import random
import numpy as np
import control as ctrl
from utils import * 

class GraphSynthesis:
    def __init__(self, args, n_sample = 10):
        
        self.graphs = None
        self.laplaciansMtx = None
        self.adjs = None
        self.N  = args['N']
        self.n = args['n']
        self.num_of_sample = n_sample
        
        

    def laplacian_matrix(self,graph):
        # Get the adjacency matrix
        adj_matrix = nx.adjacency_matrix(graph).todense()
        # Get the degree matrix
        degree_matrix = np.diag(np.sum(adj_matrix, axis=1))
        # Compute the Laplacian matrix
        laplacian_matrix = degree_matrix - adj_matrix
        return laplacian_matrix, adj_matrix


    def gnp_random_connected_graph(self,n, p):
        """
        Generates a random undirected graph, similarly to an Erdős-Rényi 
        graph, but enforcing that the resulting graph is conneted
        """
        edges = combinations(range(n), 2)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        if p <= 0:
            return G
        if p >= 1:
            return nx.complete_graph(n, create_using=G)
        for _, node_edges in groupby(edges, key=lambda x: x[0]):
            node_edges = list(node_edges)
            random_edge = random.choice(node_edges)
            G.add_edge(*random_edge)
            for e in node_edges:
                if random.random() < p:
                    G.add_edge(*e)
        return G

    def generate_connected_graphs(self,num_agents = None,num_of_sample = None):
        if num_agents is None:
            num_agents = self.N
        if num_of_sample is None:
            num_of_sample = self.num_of_sample
        all_graphs = []
        laplacian_matrices= []
        adj_matrices = []
        for i in range(num_of_sample):            
            nodes = num_agents
            seed = random.randint(1,100)
            probability = np.random.rand(1)
            connected_graphs= self.gnp_random_connected_graph(nodes,probability)
            
            L_tmp , adj_tmp = self.laplacian_matrix(connected_graphs)
            laplacian_matrices.append(L_tmp.copy())
            adj_matrices.append(adj_tmp.copy()+ np.eye(len(adj_tmp)))            
            all_graphs.append(connected_graphs.copy())
                
        unique_graphs, unique_laplacians, unique_adjs  = self.remove_duplicates_with_index(all_graphs, laplacian_matrices, adj_matrices)
        
        self.graphs= unique_graphs
        self.laplaciansMtx = unique_laplacians
        self.adjs = unique_adjs
        

    def remove_duplicates_with_index(self,graphs, laplacian_matrices, adj_matrices):
        unique_matrices = {}
        indices = []
        for i, matrix in enumerate(laplacian_matrices):
            # Convert the matrix to a tuple to use it as a dictionary key
            matrix_tuple = tuple(map(tuple, matrix))
            if matrix_tuple not in unique_matrices:
                # Add the matrix to the dictionary
                unique_matrices[matrix_tuple] = i
                indices.append(i)

        # Create a list of unique Laplacian matrices
        unique_laplacians = [np.array(matrix_tuple) for matrix_tuple in unique_matrices.keys()]    
        unique_graphs = [graphs[i] for i in indices]
        unique_adjs = [adj_matrices[i] for i in indices]
        return unique_graphs, unique_laplacians, unique_adjs

    def visualize_graph(self,graph, idx = None):
        if idx is None:
            plt.figure()        
        else: 
            plt.figure(idx)
        nx.draw(graph, with_labels=True, font_weight='bold')

        

    def visualize_graphs(self,graphs = None):
        if graphs is None:
            graphs = self.graphs
        for i, graph in enumerate(graphs):
            self.visualize_graph(graph, i)
                
    
            
            
    

    def get_unobservaible_adj_idx(self):    
        obs_idx = []
        for adj_idx, adj in enumerate(self.adjs):
            L_tmp = self.laplaciansMtx[adj_idx]
            every_agent_observable = True
            for id in range(self.N):
                c = adj            
                Ci = np.zeros([int(np.sum(c[id,:])), self.N])
                r_count = 0
                for j in np.where(c[id] == 1)[0]:
                    Ci[r_count :(r_count + 1), j:(j + 1) ] = 1
                    r_count += 1
                observability_matrix_tmp = ctrl.obsv(L_tmp, Ci)
                rank_obs = np.linalg.matrix_rank(observability_matrix_tmp)
                if rank_obs != L_tmp.shape[0]:                                                                                    
                    every_agent_observable = False
                    # print("The system is not observable.")  
                    # print("agent idx : {}", id)                    
            if every_agent_observable:
                obs_idx.append(adj_idx)
                print("The system is observable.")  
                self.visualize_graph(self.graphs[adj_idx])                
                plt.text(3, 1, 'Observable', color='blue')
                plt.show()
            else:
                print("The system is not observable.")  
                # self.visualize_graph(self.graphs[adj_idx])
                # plt.text(3, 1, 'Unobservable', color='red')
                # plt.show()
                

if __name__ == "__main__":
    args = {}
    args['N'] = 5
    args['n'] = 4
    num_of_sample = 100
    graphsynthesis = GraphSynthesis(args,n_sample = num_of_sample)
    graphsynthesis.generate_connected_graphs()
    
    graphsynthesis.get_unobservaible_adj_idx()
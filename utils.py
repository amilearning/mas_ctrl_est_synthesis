import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

import math
from scipy import linalg as la
from numpy.linalg import det


# Define an enumeration class
class CtrlTypes(Enum):
    LQROutputFeedback = 0
    SubOutpFeedback = 1 
    CtrlEstFeedback = 2
    LQGFeedback = 3
    COMLQG = 4
    


def matrixEquationSolver(A, B, F):
    if not isinstance(A, list) or not isinstance(B, list):
        raise ValueError("Defining matrices are not in a list.")

    if len(A) != len(B):
        raise ValueError("Ambiguous number of terms in the matrix equation.")

    nA, mA = A[0].shape
    nB, mB = B[0].shape

    if nA != mA or nB != mB:
        raise ValueError("Rectangular matrices are not allowed.")

    maxSize = 5000  # Do not form matrices above this size

    if nA * nB > maxSize:
        raise MemoryError("A very large matrix will be formed.")

    C = np.zeros((nA * nB, nA * nB), dtype=A[0].dtype)
    for j in range(len(A)):
        C += np.kron(B[j], A[j])

    x = np.linalg.solve(C, F.T.ravel())
    C_inv = np.linalg.solve(C, np.eye(C.shape[0]))
    # xx = np.dot(C_inv,F.ravel()).reshape(nA,nB)
    X = x.reshape(nB, nA).T

    return X
    
def get_laplacian_mtx(adj_mtx):
    laplacian_mtx = np.zeros(adj_mtx.shape)
    D_mtx = np.diagflat([np.sum(adj_mtx,axis=0)]) 
    laplacian_mtx =D_mtx - adj_mtx    
    return laplacian_mtx
    

def block_diagonal_matrix(matrix_list):
    """
    Create a block diagonal matrix from a list of matrices with potentially different dimensions.

    Args:
    matrix_list (list of 2D arrays): List of matrices to form the block diagonal matrix.

    Returns:
    numpy.ndarray: Block diagonal matrix.
    """
    # Determine the dimensions of the block diagonal matrix
    num_matrices = len(matrix_list)
    row_sizes = [matrix.shape[0] for matrix in matrix_list]
    col_sizes = [matrix.shape[1] for matrix in matrix_list]

    # Initialize the block diagonal matrix with zeros
    block_diag_size = (sum(row_sizes), sum(col_sizes))
    block_diag_matrix = np.zeros(block_diag_size)

    # Fill the diagonal blocks with the matrices from matrix_list
    row_idx = 0
    col_idx = 0
    for i in range(num_matrices):
        row_size = row_sizes[i]
        col_size = col_sizes[i]

        block_diag_matrix[row_idx:row_idx + row_size, col_idx:col_idx + col_size] = matrix_list[i]

        row_idx += row_size
        col_idx += col_size

    return block_diag_matrix

def display_array_in_window(data, cmap='viridis'):
    """
    Display a NumPy array in a separate window using Matplotlib.

    Args:
    data (numpy.ndarray): The array to display.
    cmap (str): The colormap to use for visualization (default: 'viridis').

    Returns:
    None
    """
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap=cmap)
    plt.colorbar(im)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')


    def close_window(event):
        if event.key or event.button:
            plt.close()

    fig.canvas.mpl_connect('key_press_event', close_window)
    fig.canvas.mpl_connect('button_press_event', close_window)

    plt.show()
    
    
def plot_x_y(traj_list):        
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_ylabel('X')
    ax1.set_title('Trajectories in X and Y Dimensions')        
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Y')
    for traj in traj_list:
        x = traj[:, 0]
        y = traj[:, 2]
        time_steps = np.arange(len(traj))  # Assuming time steps are sequential integers            
        # ax1.plot(time_steps, x, label='Agent Trajectory', linewidth=1)
        # ax2.plot(time_steps, y, label='Agent Trajectory', linewidth=1)
        ax1.plot(x, y, label='Agent Trajectory')
    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.show()

def plot_3dtraj(traj_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for traj in traj_list:
        x = traj[:, 0]
        y = traj[:, 1]
        z = np.ones([len(traj),1])  # Use z=1 for all points, assuming 2D trajectories
        ax.plot(x, y, z, label='Trajectory', linewidth=1, marker='o', markersize=1.5, markeredgecolor='blue', markeredgewidth=1, markerfacecolor='blue')
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('3D Trajectory Plot')        
    ax.legend()
    plt.show()
    
def plot_comparison_result(lqg_result, sub_result, opt_result, comlqg_result, comlqg_5_result):
    if len(lqg_result) ==1:
        lqg_result = lqg_result[0]
    if len(sub_result) ==1:
        sub_result = sub_result[0]
    if len(opt_result) ==1:
        opt_result = opt_result[0]

    if len(comlqg_result) ==1:
        comlqg_result = comlqg_result[0]

    if len(comlqg_5_result) ==1:
        comlqg_5_result = comlqg_5_result[0]

    costs = []
    costs.append(np.mean(lqg_result['stage_cost'], axis=0))
    costs.append(np.mean(sub_result['stage_cost'], axis=0))
    costs.append(np.mean(opt_result['stage_cost'], axis=0))
    costs.append(np.mean(comlqg_result['stage_cost'], axis=0))
    costs.append(np.mean(comlqg_5_result['stage_cost'], axis=0))

    plt.plot(costs[0].squeeze(), label='lqg')
    plt.plot(costs[1].squeeze(), label='suboptimal')
    plt.plot(costs[2].squeeze(), label='ctrlest')
    plt.plot(costs[3].squeeze(), label='comlqg')
    plt.plot(costs[4].squeeze(), label='comlqg5')

    # Set plot labels and title
    plt.xlabel('Time Step')
    plt.ylabel('Stage Cost')
    plt.title('Stage Costs Over Time Horizon')

    # Add a legend
    plt.legend()
    plt.show()
    print("done")

def plot_stage_cost(stage_costs):  
    stage_costs = np.array(stage_costs).squeeze()
    stage_costs = stage_costs[50:]
    time_steps = np.arange(len(stage_costs))    
    
    plt.plot(time_steps, stage_costs, marker='o', linestyle='-', markersize=5, label='Stage Costs')    
    plt.xlabel('Time Step')
    plt.ylabel('Stage Cost')
    plt.title('Stage Costs Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_mas_traj(trajs):
    num_agents, num_time_steps, state_dim = trajs.shape

    # Create a figure
    plt.figure(figsize=(8, 6))

    # Label the axes
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Agent Trajectories')

    # Create a colormap with a unique color for each agent
    cmap = plt.get_cmap('tab10')

    # Extract and plot the x and y positions for each agent with a different color
    for agent_idx in range(num_agents):
        x_positions = trajs[agent_idx, :, 0]  # Assuming x position is in the first state dimension (0)
        y_positions = trajs[agent_idx, :, 2]  # Assuming y position is in the third state dimension (2)

        # color = cmap(agent_idx / num_agents)  # Get a unique color based on agent index
        # plt.plot(x_positions, y_positions, label=f'Agent {agent_idx+1}', color=color)
        color = cmap(agent_idx / num_agents)  # Get a unique color based on agent index
        plt.plot(x_positions, y_positions, label=f'Agent {agent_idx+1}', color=color, alpha=0.5,marker='o', markersize=1.5)  # Make lines transparent

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()
    
def error_plot(error_data):

    num_agents = error_data.shape[0]
    num_time_steps = error_data.shape[1]
    # Create a figure with subplots for each agent
    fig, axs = plt.subplots(num_agents, 1, figsize=(8, 6*num_agents), sharex=True)

    # Plot error data for each agent in its own subplot
    for agent in range(num_agents):
        axs[agent].plot(range(num_time_steps), error_data[agent], label=f'Agent {agent+1}')
        axs[agent].set_ylabel('Error')
        axs[agent].set_title(f'Agent {agent+1} Error')

    # Add common X-axis label
    axs[num_agents - 1].set_xlabel('Time Step')

    # Add a legend to distinguish agents
    for agent in range(num_agents):
        axs[agent].legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    
def get_formation_offset_vector(N,n,dist = 4.0):    

    rel_pos = np.zeros((N, 2))

    for i in range(N):
        theta = 2 * np.pi / (N + 1) * (i + 1)
        rel_pos[i, :] = [dist * np.cos(theta), dist * np.sin(theta)]

    xref_tmp = np.zeros((N, N, 2))
    xref = np.zeros((n * N, N))
    offset_mtx = np.zeros((n * N, 1))

    for i in range(N):
        for j in range(N):
            xref_tmp[i, j, :] = rel_pos[i, :] - rel_pos[j, :]
            xref[j * n:j * n + n, i] = [xref_tmp[i, j, 0], 0, xref_tmp[i, j, 1], 0]

    xref = -1 * xref

    # Create offset matrix
    for id in range(N):
        offset_mtx[id * n:id * n + n, 0] = [rel_pos[id, 0], 0, rel_pos[id, 1], 0]

    return offset_mtx

def get_chain_adj_mtx(N):
    adjacency_matrix = np.zeros((N, N), dtype=int)
    # Specify the connections for the chain network
    for i in range(N - 1):
        adjacency_matrix[i, i + 1] = 1        
        adjacency_matrix[i + 1, i] = 1  # Uncomment if you want bidirectional connections
        
    for i in range(N):
        adjacency_matrix[i,i] = 1
    # np.diag(adjacency_matrix) = 1
    return adjacency_matrix

def get_circular_adj_mtx(N):
  

    # Initialize an empty adjacency matrix filled with zeros
    adj_matrix = np.zeros((N, N), dtype=int)

    # Create connections in the circular graph
    for i in range(N):
        # Connect each node to its adjacent nodes
        adj_matrix[i][(i + 1) % N] = 1  # Connect to the next node
        adj_matrix[i][(i - 1) % N] = 1  # Connect to the previous node

   
    adj_matrix = adj_matrix + np.eye(N)

    return adj_matrix


def compute_mahalanobix_dist(A,B):
    k = A.shape[0]
    e_cov_next_inv = la.solve(B, np.eye(B.shape[0]))
    tmp = np.dot(e_cov_next_inv, A)
    dKL = 0.5 * (np.trace(tmp) - k - np.log(det(tmp)))                
    return dKL

import numpy as np

from utils import *

import numpy as np
from scipy.signal import cont2discrete

N = 5
n = 4
input_size = 1
Ts = 0.1

Ac = np.array([[0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0]])

Bc = np.array([[0, 0],
              [1, 0],
              [0, 0],
              [0, 1]])

A = np.array([[1, Ts, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, Ts],
              [0, 0, 0, 1]])

B = np.array([[Ts**2/2, 0],
              [Ts, 0],
              [0, Ts**2/2],
              [0, Ts]])

Q = np.eye(n)
R = np.eye(input_size)

L = np.array([[2, -1, -1, 0, 0],
              [-1, 2, 0, -1, 0],
              [-1, 0, 2, 0, -1],
              [0, -1, 0, 1, 0],
              [0, 0, -1, 0, 1]])

sample = 100

Ac = np.kron(np.eye(5), Ac)
Bc = np.kron(np.eye(5), Bc)

eigvalue = np.sort(np.linalg.eigvals(L))
epsilon = 0.0001
c = 2 / (eigvalue[1] + eigvalue[-1])
ct = (c**2) * eigvalue[-1]**2 - 2 * c * eigvalue[-1]

G = np.dot(np.dot(Bc, np.linalg.inv(R)), Bc.T) * ct
Qbar = eigvalue[-1] * Q + epsilon * np.eye(n)

from scipy.linalg import solve_discrete_are
P = solve_discrete_are(Ac, None, Qbar, None, None, None, G)
K1 = -np.dot(np.dot(np.linalg.inv(R), Bc.T), P)
K = -c * np.dot(np.linalg.inv(R), Bc.T).dot(P)
gain_c = np.kron(L, K)

Ad, Bd, _, _, _ = cont2discrete((Ac, Bc, np.eye(4), np.eye(2)), Ts, method='zoh')

A_tilde_c = Ac + np.dot(Bc, K)
sys = cont2discrete((A_tilde_c, None, np.eye(4), np.eye(2)), Ts, method='zoh')
A_tilde_d = sys[0]

Kd = np.dot(np.linalg.pinv(Bd), A_tilde_d - Ad)
gain_d = np.kron(L, Kd)

print("gain_c:\n", gain_c)
print("\ngain_d:\n", gain_d)


# A = np.array([[1, 2], [3, 4]])  # Replace with your values
# B = np.array([[5, 6], [7, 8]])  # Replace with your values
# C = np.array([[9, 10], [11, 12]])  # Replace with your values
# D = np.array([[2, 10], [11, 12]])  # Replace with your values

# G = np.array([[9, 2], [11, 12]])  # Replace with your values
# E = np.array([[13, 14], [15, 16]])  # Replace with your values
# # P = sp.Matrix([[17, 18], [19, 20]])  # Replace with your values
# F_sol = np.array([[17, 18], [19, 20]])  # Replace with your values
# P = np.dot(np.dot(A, F_sol), B.T) +  np.dot(np.dot(C, F_sol), D.T) +  np.dot(np.dot(E, F_sol), G.T) 

# a = [] 
# b = []
# a.append(A)
# a.append(C)
# a.append(E)
# b.append(B)
# b.append(D)
# b.append(G)
# F = matrixEquationSolver(a,b,P)
# print(1)

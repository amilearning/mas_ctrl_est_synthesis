
import numpy as np

from utils import *

A = np.array([[1, 2], [3, 4]])  # Replace with your values
B = np.array([[5, 6], [7, 8]])  # Replace with your values
C = np.array([[9, 10], [11, 12]])  # Replace with your values
D = np.array([[2, 10], [11, 12]])  # Replace with your values

G = np.array([[9, 2], [11, 12]])  # Replace with your values
E = np.array([[13, 14], [15, 16]])  # Replace with your values
# P = sp.Matrix([[17, 18], [19, 20]])  # Replace with your values
F_sol = np.array([[17, 18], [19, 20]])  # Replace with your values
P = np.dot(np.dot(A, F_sol), B.T) +  np.dot(np.dot(C, F_sol), D.T) +  np.dot(np.dot(E, F_sol), G.T) 

a = [] 
b = []
a.append(A)
a.append(C)
a.append(E)
b.append(B)
b.append(D)
b.append(G)
F = matrixEquationSolver(a,b,P)
print(1)

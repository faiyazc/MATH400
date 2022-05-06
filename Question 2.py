
import numpy as np
from numpy import linalg as la
from numpy import inf
import copy
"""
Question 2:
Use the Jacobi method and the Gauss-Seidel method to solve the indicated linear system of equations. Your code should 
efficiently use the sparseness of the coefficient matrix. Take x_0 = [0, ..., 0], and terminate the iteration when 
||e_{k+1}||_{inf} falls below 5e-6. Record the iterations required to achieve convergence.
"""
tol = 5e-6
A = [[4, -1, 0, -2, 0, 0],
     [-1, 4, -1, 0, -2, 0],
     [0, -1, 4, 0, 0, -2],
     [-1, 0, 0, 4, -1, 0],
     [0, -1, 0, -1, 4, -1],
     [0, 0, -1, 0, -1, 1]]

b = [[-1],
     [0],
     [1],
     [-2],
     [1],
     [2]]


def jacobiMethod(A, b):
    x = np.zeros(len(b))
    error = 1  # Set a dummy tolerance to start iteration
    iteration = 0
    x0 = copy.copy(x)
    m = len(b)
    while error > tol:
        print("nextX = \n", x0, "\n")
        print("error = \n", error, "\n")

        # Use a for loop to figure this out/write a function to do it.
        # nextX[0] = 1/A[0][0] * (b[0] - (A[0][1] * x[1] + ... + A[0][len(A)] * x[len(A)]))
        row_sum = 0  # sum of a_{ij}*x_{j} for all j
        for i in range(m):          # iterate through rows
            for j in range(len(A[0])):   # iterate through columns
                if A[i][j] == 0:
                    continue
                row_sum = row_sum + A[i][j] * x[j]
            x0[i] = (b[i] - row_sum) / A[i][i]

        error = la.norm(np.abs(np.subtract(x0, x)), inf)
        x = copy.copy(x0)
        iteration = iteration + 1
    print(f" Number of iterations for Jacobi = {iteration}")
    print(f"approximate x = {x0}")


jacobiMethod(A, b)

def gaussSeidelMethod(A, b):
    x = np.zeros(len(b))
    error = 1  # Set a dummy tolerance to start iteration
    iteration = 0
    x0 = copy.copy(x)
    m = len(b)
    while error > tol:
        print("nextX = \n", x0, "\n")
        print("error = \n", error, "\n")

        # Use a for loop to figure this out/write a function to do it.
        # nextX[0] = 1/A[0][0] * (b[0] - (A[0][1] * x[1] + ... + A[0][len(A)] * x[len(A)]))
        row_sum = 0  # sum of a_{ij}*x_{j} for all j
        for i in range(len(b)):  # Iterate through rows
            current_column = 0  # Keep track of which element we are on
            for index in range(current_column): # Iterate 0->current_column
                print(A[i][index])
                row_sum = row_sum + A[i][index] * x[index]
            for index in range(current_column, len(b)):  # Iterate current_column->len(b)
                print(A[i][index])
                row_sum = row_sum + A[i][index] * x0[index]
            x0[i] = (b[i] - row_sum) / A[i][i]

        error = la.norm(np.abs(np.subtract(x0, x)), inf)
        x = copy.copy(x0)
        iteration = iteration + 1
    print(f" Number of iterations for Gauss-Seidel = {iteration}")
    print(f"approximate x = {x0}")


gaussSeidelMethod(A, b)


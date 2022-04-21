"""
Created on 4/20/2022 (LOL)

Author: @faiyazc
"""
from importlib import util

import numpy as np
from numpy import linalg as la
from numpy import inf

"""
Question 1:
"""

"""
Question 2:
Use the Jacobi method and the Gauss-Seidel method to solve the indicated linear system of equations. Your code should 
efficiently use the sparseness of the coefficient matrix. Take x_0 = [0, ..., 0], and terminate the iteration when 
||e_{k+1}||_{inf} falls below 5e-6. Record the iterations required to achieve convergence.
"""
tol = 5e-6
A = [
    [4, -1, 0, -2, 0, 0],
    [-1, 4, -1, 0, -2, 0],
    [0, -1, 4, 0, 0, -2],
    [-1, 0, 0, 4, -1, 0],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 1]
]
b = [
    [-1],
    [0],
    [1],
    [-2],
    [1],
    [2]
]


def jacobiMethod(A, b):
    L, D, U = matrixSplitter(A)
    x = np.zeros(len(b))
    error = 1  # Set a dummy tolerance to start iteration
    iteration = 0
    nextX = x
    while error > tol:
        # Use a for loop to figure this out/write a function to do it.
        # nextX[0] = 1/A[0][0] * (b[0] - (A[0][1] * x[1] + ... + A[0][len(A)]x[len(A)]))
        error = la.norm(nextX, inf)


def gaussSeidelMethod(A, b):
    L, D, U = matrixSplitter(A)
    x = np.zeros(len(b))


def matrixSplitter(A):
    L = np.zeros((len(A), len(A)))
    D = np.zeros((len(A), len(A)))
    U = np.zeros((len(A), len(A)))
    " Iterate through A and return ternary of L, D, U"
    for i in range(len(A)):
        """print("A[i]: ", A[i])"""
        for j in range(len(A[i])):
            """print("A[i][j]", A[i][j])"""
            if i > j:
                U[i][j] = A[i][j]
            if i == j:
                D[i][j] = A[i][j]
            if i < j:
                L[i][j] = A[i][j]
    return L, D, U


matrixSplitter(A)

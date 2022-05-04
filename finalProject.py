"""
Created on 4/20/2022 (LOL)

Author: @faiyazc
"""
from importlib import util

import numpy as np
from numpy import array
from numpy import linalg as la
from numpy import inf
import copy

"""
Question 1:
# Solve nonlinear system using Newton's method with a given intitial vector. 
# Terminating the process when the max norm of difference between
# successive iterates is less than 5 * 10^-6. 
        
#Implements Gaussian elimination with partial pivoting to solve the linear
#system in each step. Uses numerical differentiation to compute the Jacobian
#matrix at each step.
"""
   
#Section 1
def A(x):
  x1 = x[0]
  x2 = x[1]
  x3 = x[2]
  output = np.zeros(3)
  output[0] = x[0]**2 + x[1]**2 + x[2]**2 - 1
  output[1] = x[0]**2 + x[2]**2 - 0.25
  output[2] = x[0]**2 + x[1]**2 - 4*x[2]
  return output

b = array([1, 0.25, 0])

m = len(b)

x = array([1.0, 1, 1]) # initial vector

v = [0,0,0]
w = [0,0,0]
w = [[0],[0],[0]]


n = 0 #iteration counter
x0 = copy.copy(x)

xn = 1 # dummy initial tolerance
h = 1e-2
tol = 5 * 10**-6



#approximate the jacobian matrix using central finite differences
def jacobian(A, x, h):
    """
        Computes the Jacobian matrix of the nonlinear system using central finite differences.
    """
    J = np.zeros_like(A)
    
    nrow = len(A(x))
    ncol = len(x)
    J = np.zeros(nrow*ncol)
    J = J.reshape(nrow,ncol)
    for i in range(nrow):
        for j in range(ncol):
            ej = np.zeros(ncol)
            ej[j] = 1
            dij = (A(x + h * ej)[i] - A(x - h * ej)[i])/(2*h)
            J[i,j] = dij
    return J
            


def newton_nonlinear (A, b, x, h, tol, xn, x0):
    """
        Solves nonlinear system using Newton's method with a given intitial vector. 
        Terminating the process when the max norm of difference between
        successive iterates is less than 5 * 10^-6. 
        
        Implements Gaussian elimination with partial pivoting to solve the linear
        system in each step. Uses numerical differentiation to compute the Jacobian
        matrix at each step.
        
        Parameters
        ----------
        A : 
        
        b :
        
        x : 
        
        tol :
    """

    while xn > tol:
        # compute the Jacobian matrix
        J = jacobian(A, x, h)
        # # compute the residual vector
        # r = b - A.dot(x)
        # compute the Newton step
        delta_x = LA.solve(J, b)
        # update the solution
        x = x + delta_x
        # compute the norm of the difference between the current and previous iterates
        xn = LA.norm(x - x0)
        # update the previous iterate
        x0 = copy.copy(x)
    
    return x
       

#1
#Test Jacobian Method
print(jacobian(A, x, h))

#Call Newton's Method
print(newton_nonlinear(A, b, x, h, tol, xn, x0))






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

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
    
A1 = array([[1, 0, 1, 0, 1, 0], [1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 0, -4]])
A2 = array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 2, 0, 0], [0, 1, 0, 0, 0, 3]])
A3 = array([[1, 50, 1, 0, 1, 0], [1, 0, 0, 20, 1, 0], [-1, 0, -1, 0, 0, 40]])

b1 = array([1, 0.25, 0])
b2 = array([10, 2, 9])
b3 = array([200, 50, -75])
  
m1 = len(b1)
m2 = len(b2)
m3 = len(b3)

x1 = array([1, 1, 1]) # initial vector
x2 = array([2, 0, 2]) # initial vector
x3 = array([2, 2, 2]) # initial vector


n = 0 #iteration counter
x01 = copy.copy(x1)
x02 = copy.copy(x2)
x03 = copy.copy(x3)

xn = 1 #dummy initial tolerance
tol = 5 * 10**-6

def jacobian(A, b, x, tol, xn, x0):
    
    while xn > tol:
    
        x[0] = (b[0] - ( A[0,1]*x0[1] ) )/A[0,0]
    
        for i in range(1,m-1):
            x[i] = ( b[i]- ( A[i, i-1]*x0[i-1] + A[i,i+1]*x0[i+1] ) )/A[i,i]
            
        x[m-1] = (b[m-1] - ( A[m-1,m-2]*x0[m-2] ) )/A[m-1,m-1]
    
        #xn = LA.norm(x-x0,1) 
        #xn = LA.norm(x-x0,2) 
        xn = LA.norm(x-x0,inf)
        x0 = copy.copy(x)
        n = n + 1

def newton_nonlinear (A, b, x, tol, xn, x0):
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
        J = jacobian(A, b, x, tol, xn, x0)
        # compute the residual vector
        r = b - A.dot(x)
        # compute the Newton step
        delta_x = LA.solve(J, r)
        # update the solution
        x = x + delta_x
        # compute the norm of the difference between the current and previous iterates
        xn = LA.norm(x - x0)
        # update the previous iterate
        x0 = copy.copy(x)
        # increment the iteration counter
        n = n + 1

#1
x = x1
A = A1
b = b1
m = m1
x0 = x01

#Test Jacobi Method
jacobian(A, b, x, tol, xn, x0)

#Call Newton's Method
newton_nonlinear(A, b, x, tol, xn, x0)

#2
x = x2
A = A2
b = b2
m = m2
x0 = x02

#Call Newton's Method
newton_nonlinear(A, b, x, tol, xn, x0)

#3
x = x3
A = A3
b = b3
m = m3
x0 = x03

#Call Newton's Method
newton_nonlinear(A, b, x, tol, xn, x0)


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

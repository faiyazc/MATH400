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
    
f = lambda x,y,z: x**2 + y**2 + z**2 - 1
g = lambda x,y,z: x**2 + z**2 - 0.25
h = lambda x,y,z: x**2 + y**2 - 4*z  


#Section 1
A = array([f, g, h])

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
def jacobian(A, b, x, h):
    """
        Computes the Jacobian matrix of the nonlinear system using central finite differences.
        
        Parameters
        ----------
        A : 
        
        b :
        
        x : 
        
        h :
    """
    J = np.zeros_like(A)
    
    while xn > tol:
        for i in range(0,len(x)):
            v[i] = x + h[i]
            w[i] = x - h[i]
       
        x = v[0] 
        y = v[1]
        z = v[2]
        
        l = w[0]
        m = w[1]
        n = w[2]

        final_vector = (array([f(x,y,z), g(x,y,z), h(x,y,z)]) - array([f(l,m,n), g(l,m,n), h(l,m,n)])) / (2 * h)
        
        J = [[final_vector[0,0], final_vector[1,0], final_vector[2,0]], 
             [final_vector[0,1], final_vector[1,1], final_vector[2,1]],
             [final_vector[0,2], final_vector[1,2], final_vector[2,2]]]

    return J
    


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
#Test Jacobian Method
jacobian(A, b, x, h)


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

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


# Section 1
def A(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    output = np.zeros(3)
    output[0] = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 1
    output[1] = x[0] ** 2 + x[2] ** 2 - 0.25
    output[2] = x[0] ** 2 + x[1] ** 2 - 4 * x[2]
    return output


b = array([1, 0.25, 0])

m = len(b)

x = array([1.0, 1, 1])  # initial vector

v = [0, 0, 0]
w = [0, 0, 0]
w = [[0],
     [0],
     [0]]

n = 0  # iteration counter
x0 = copy.copy(x)

xn = 1  # dummy initial tolerance
h = 1e-2
tol = 5 * 10 ** -6


# approximate the jacobian matrix using central finite differences
def jacobian(A, x, h):
    """
        Computes the Jacobian matrix of the nonlinear system using central finite differences.
    """
    J = np.zeros_like(A)

    nrow = len(A(x))
    ncol = len(x)
    J = np.zeros(nrow * ncol)
    J = J.reshape(nrow, ncol)
    for i in range(nrow):
        for j in range(ncol):
            ej = np.zeros(ncol)
            ej[j] = 1
            dij = (A(x + h * ej)[i] - A(x - h * ej)[i]) / (2 * h)
            J[i, j] = dij
    return J


def newton_nonlinear(A, b, x, h, tol, error, x0):
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

    while error > tol:
        # compute the Jacobian matrix
        J = jacobian(A, x, h)
        # # compute the residual vector
        # r = b - A.dot(x)
        # compute the Newton step
        delta_x = la.solve(J, b)
        # update the solution
        x = x + delta_x
        print(x)
        # compute the norm of the difference between the current and previous iterates
        error = la.norm(x0 - x, inf)
        # update the previous iterate
        x0 = copy.copy(x)

    return x


# 1
# Test Jacobian Method
print("jacobian: ", jacobian(A, x, h))

# Call Newton's Method
print("newton_nonlinear" ,newton_nonlinear(A, b, x, h, tol, xn, x0))

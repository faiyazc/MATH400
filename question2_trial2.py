
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



def jacobi(a, b, x0, tol, iter_max):
    """Jacobi method: solve Ax = b given an initial approximation x0.

    Args:
        a: matrix A from system Ax=b.
        b: an array containing b values.
        x0: initial approximation of the solution.
        tol: tolerance.
        iter_max: maximum number of iterations.

    Returns:
        x: solution of linear the system.
        iter: used iterations.
    """
    # D and M matrices
    d = np.diag(np.diag(a))
    m = a - d

    # Iterative process
    i = 1
    x = None
    for i in range(1, iter_max + 1):
        x = np.linalg.solve(d, (b - np.dot(m, x0)))

        if np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf) <= tol:
            break
        x0 = x.copy()

    # print(f" Number of iterations for Jacobi = {iter}")
    print(f"approximate x = {x0}")


    jacobi(A, b, 0, 5e-6, 1000)
    return [x, i]

    


def gauss_seidel(a, b, x0, tol, iter_max):
    """Gauss-Seidel method: solve Ax = b given an initial approximation x0.

    Args:
        a: matrix A from system Ax=b.
        b: an array containing b values.
        x0: initial approximation of the solution.
        tol: tolerance.
        iter_max: maximum number of iterations.

    Returns:
        x: solution of linear the system.
        iter: used iterations.
    """
    # L and U matrices
    lower = np.tril(a)
    upper = a - lower

    # Iterative process
    i = 1
    x = None
    for i in range(1, iter_max + 1):
        x = np.linalg.solve(lower, (b - np.dot(upper, x0)))

        if np.linalg.norm(x - x0, np.inf) / np.linalg.norm(x, np.inf) <= tol:
            break
        x0 = x.copy()

    return [x, i]
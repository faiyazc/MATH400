import copy
import numpy as np 
import numpy.linalg as la

def A1(x):
  output = np.zeros(3)
  output[0] = x[0]**2 + x[1]**2 + x[2]**2 - 1
  output[1] = x[0]**2 + x[2]**2 - 0.25
  output[2] = x[0]**2 + x[1]**2 - 4*x[2]
  return output

def A2(x):
    output = np.zeros(3)
    output[0] = x[0]**2 + x[1]**2 + x[2]**2 - 10
    output[1] = x[0] + 2*x[1] - 2
    output[2] = x[0] +  3*x[2] - 9
    return output

def A3(x):
    output = np.zeros(3)
    output[0] = x[0]**2 + 50*x[0] + x[1]**2 + x[2]**2 - 200
    output[1] = x[0]**2 + 20*x[1] + x[2]**2 - 50
    output[2] = -x[0]**2 - x[1]**2 - 40*x[2] + 75
    return output

""" Compute the Jacobian matrix of the nonlinear system using forward finite differences """
def jacobian(A, x, h):
    nrow = len(A(x))
    ncol = len(x)
    J = np.zeros(nrow*ncol)
    J = J.reshape(nrow,ncol)

    for i in range(nrow):
        for j in range(ncol):
            x_h = copy.deepcopy(x)
            x_h[j] = x_h[j] + h
            J[i,j] = (A(x_h)[i] - A(x)[i])/h
    return J


""" applying Newton method  """
def Newton(A, x, h, tol, n):
    xn = copy.deepcopy(x)
    while True:
        J = jacobian(A, xn, h)
        f = A(xn)
        delta = la.solve(J, -f)
        xn = xn + delta
        if la.norm(delta) < tol:
            break
        n += 1
        print (n)
    return xn

x = [1,1,1]
n = 0

print("A1:")
print(Newton(A1, x, 0.001, 0.001, n))

x = [2,0,2]
n = 0

print("A2:")
print(Newton(A2, x, 0.001, 0.001, n))


x = [2,2,2]
n = 0

print("A3:")
print(Newton(A3, x, 0.001, 0.001, n))

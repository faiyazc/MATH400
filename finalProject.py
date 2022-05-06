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
"""
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

""" Compute Gaussian Elimination with partial pivoting """
def part_piv_ge(A,b):

    nrow = A.shape[0]
    ncol = A.shape[1]
    
    #    Gaussian elimination

    for i in range(ncol-1): #loop over columns
        col_i = abs(A[i:nrow,i]) #isolate column i
        pivot=max(col_i)
        t = np.where(col_i == pivot)[0]; #find pivot i and its index  
        t = t + i 
        #interchanging i^th row with pivot row
        temp = A[i,:].copy()
        tb=b[i].copy();  
        A[i,:] = A[t,:]
        b[i]=b[t]
        A[t,:] = temp
        b[t]=tb
        
        aii=A[i,i]
         
        for j in range(i+1, nrow): #loop over rows below column i 
            m = -A[j,i] / aii #multiplier
            A[j,i] = 0
            A[j, i+1:nrow] = A[j, i+1:nrow] + m*A[i, i+1:nrow]
            b[j] = b[j] + m*b[i]
            
    #  back substitution
    
    x = np.zeros((nrow)); #initialize vector of zeros 
    
    nrow = nrow - 1
    x[nrow] = b[nrow] / A[nrow, nrow]
    
    for i in range(nrow -1, -1, -1):    
        #dot = np.dot(x[i+1:nrow+1], A[i, i+1:nrow+1])  
        dot = A[i, i+1:nrow+1] @ x[i+1:nrow+1]
        x[i] = (b[i] - dot ) / A[i,i]
    
    return x


""" applying Newton method  """
def Newton(A, x, h, tol, n):
    xn = copy.deepcopy(x)
    while True:
        J = jacobian(A, xn, h)
        f = A(xn)
        #find delta using partial pivoting
        #delta = part_piv_ge(J, -f)#Partial pivoting call
        delta = la.solve(J, -f) #linalg method used because part_piv_ge is breaking
        xn = xn + delta
        if la.norm(delta) < tol:
            break
        n += 1
    print (n, "iterations")
    return xn

x = [1,1,1]
n = 0

print("A1:")
print(Newton(A1, x, 5e-6, 0.001, n))

x = [2,0,2]
n = 0

print("A2:")
print(Newton(A2, x, 5e-6, 0.001, n))


x = [2,2,2]
n = 0

print("A3:")
print(Newton(A3, x, 5e-6, 0.001, n))

print(jacobian(A3, [1,0,1], 0.001))


"""
Question 2
"""
# resources used: professor's code and python library pages to find infinity norm directly
import numpy as np
from numpy import linalg as LA
import copy

A = [
    [4, -1, 0, -2, 0, 0],
    [-1, 4, -1, 0, -2, 0],
    [0, -1, 4, 0, 0, -2],
    [-1, 0, 0, 4, -1, 0],
    [0, -1, 0, -1, 4, -1],
    [0, 0, -1, 0, -1, 4]
]

b = [-1, 0, 1, -2, 1, 2]


def jacobi(A, xk0, b, tol):
    
    number_of_iterations = 0
    xk1 = xk0.copy()  # making a copy of k0 to keep  track 
    e = 1  #  e = ||E||
    relative_error = [1,1,1,1,1,1]  # x^(k+1) - x^k
    
    #setting the limit of error that it shouldn't be greater than the tolerance
    while (e > tol):
        # going thru x1 to x6
        xk1[0] = (b[0] - (A[0][1]*xk0[1] + A[0][3]*xk0[3])) / A[0][0]
      
        xk1[1] = (b[1] - (A[1][0]*xk0[0] + A[1][2] *
                  xk0[2] + A[1][4]*xk0[4])) / A[1][1]
        
        xk1[2] = (b[2] - (A[2][1]*xk0[1] + A[2][5]*xk0[5])) / A[2][2]
        
        xk1[3] = (b[3] - (A[3][0]*xk0[0] + A[3][4]*xk0[4])) / A[3][3]
        
        xk1[4] = (b[4] - (A[4][1]*xk0[1] + A[4][3] *
                  xk0[3] + A[4][5]*xk0[5])) / A[4][4]
        
        xk1[5] = (b[5] - (A[5][2]*xk0[2] + A[5][4]*xk0[4])) / A[5][5]

        # Looping through length of relative error to find the error bound
        for i in range(0, len(relative_error)):
            relative_error[i] = xk1[i] - xk0[i]

        # calculating the infinity norm of error
        e = LA.norm(relative_error, np.inf)

        # assigning the new value of k1 to k0
        xk0 = copy.copy(xk1)

        
        print(f"  k = {number_of_iterations} ")
        print(f"approx x:\n{xk0}\n")

        #incrementing the iteration counter by 1 each time
        number_of_iterations = number_of_iterations + 1

        #returning the values of xk1 and number of iterations
    return xk1, number_of_iterations


def gaussSeidel(A, xk0, b, tol):
   
    number_of_iterations = 0
    xk1 = xk0.copy()  # making a copy of k0 to keep  track 
    e = 1  #  e = ||E||
    relative_error = [1,1,1,1,1,1]  # x^(k+1) - x^k

    #setting the limit of error that it shouldn't be greater than the tolerance
    while (e > tol):
        
        
        # going thru x1 to x6
        xk1[0] = (b[0] - (A[0][1]*xk1[1] + A[0][3]*xk1[3])) / A[0][0]
        # x2
        xk1[1] = (b[1] - (A[1][0]*xk1[0] + A[1][2] *
                  xk1[2] + A[1][4]*xk1[4])) / A[1][1]
        # x3
        xk1[2] = (b[2] - (A[2][1]*xk1[1] + A[2][5]*xk1[5])) / A[2][2]
        # x4
        xk1[3] = (b[3] - (A[3][0]*xk1[0] + A[3][4]*xk1[4])) / A[3][3]
        # x5
        xk1[4] = (b[4] - (A[4][1]*xk1[1] + A[4][3] *
                  xk1[3] + A[4][5]*xk1[5])) / A[4][4]
        # x6
        xk1[5] = (b[5] - (A[5][2]*xk1[2] + A[5][4]*xk1[4])) / A[5][5]

        # Looping through length of relative error to find the error bound
        for i in range(0, len(relative_error)):
            relative_error[i] = xk1[i] - xk0[i]

        # calculating the infinity norm of error
        e = LA.norm(relative_error, np.inf)
        
        # assigning the new value of k1 to k0
        xk0 = copy.copy(xk1)

        
        print(f" k = {number_of_iterations}")
        print(f"approx x:\n{xk1}\n")

        
        #incrementing the iteration counter by 1 each time
        number_of_iterations = number_of_iterations + 1

    #returning the values of xk1 and number of iterations
    return xk1, number_of_iterations



print(" Iterations with the Jacobi Method: \n")
x0 = [0, 0, 0, 0, 0, 0]
k = 0
x0, k = jacobi(A, x0, b, 5e-6)

print("number of iteration:", k)
print(f"approximate x = {x0}\n")
print()

print("Iterations with the Gauss-Seidel Method: \n")
x0 = [0, 0, 0, 0, 0, 0]
k = 0
x0, k = gaussSeidel(A, x0, b, 5e-6)

print("number of iteration:", k)
print("approximate x = ", x0)
print()

"""
Question 3
"""
#Resources used: online python libraries, youtube videos

import numpy as np
import sympy as sp
import scipy.linalg as la
from scipy.linalg import lu, lu_factor, lu_solve



a00, a10, a20, a30 = sp.symbols('a00, a10, a20, a30')
a01, a11, a21, a31 = sp.symbols('a01, a11, a21, a31')
a02, a12, a22, a32 = sp.symbols('a02, a12, a22, a32')
a03, a13, a23, a33 = sp.symbols('a03, a13, a23, a33')

a = [a00, a10, a20, a30,
     a01, a02, a03, a11,
     a21, a31, a12, a22,
     a32, a13, a23, a33]
def partA():
    x, y = sp.symbols('x, y')
    for i in 0, 1:
        for j in 0, 1:
            print("p(", i, j, "): ", interpolate(i, j))

    for i in 0, 1:
        for j in 0, 1:
            x, y = i, j
            print("px(", i, j, "): ", interpolate_x(x, y))

    for i in 0, 1:
        for j in 0, 1:
            x, y = i, j
            print("py(", i, j, "): ", interpolate_y(x, y))

    for i in 0, 1:
        for j in 0, 1:
            x, y = i, j
            print("pxy(", i, j, "): ", interpolate_xy(x, y))

def interpolate(x, y):
    p = np.dot(a, v(x, y))
    return p

def interpolate_x(x, y):
    p = np.dot(a, vx(x, y))
    return p

def interpolate_y(x, y):
    p = np.dot(a, vy(x, y))
    return p

def interpolate_xy(x, y):
    p = np.dot(a, vxy(x, y))
    return p

def v(x, y):
    return [1,
            x,
            x ** 2,
            x ** 3,
            y,
            y ** 2,
            y ** 3,
            x * y,
            x ** 2 * y,
            x ** 3 * y,
            x * y ** 2,
            x ** 2 * y ** 2,
            x ** 3 * y ** 2,
            x * y ** 3,
            x ** 2 * y ** 3,
            x ** 3 * y ** 3]

def vx(x, y):
    return [
        0,                      # 1
        1,                      # x
        2 * x,                  # x ** 2
        3 * x ** 2,             # x ** 3
        0,                      # y
        0,                      # y ** 2
        0,                      # y ** 3
        y,                      # x * y
        2 * x * y,              # x ** 2 * y
        3 * x ** 2 * y,         # x ** 3 * y
        y ** 2,                 # x * y ** 2
        2 * x * y ** 2,         # x ** 2 * y ** 2
        3 * x ** 2 * y ** 2,    # x ** 3 * y ** 2
        y ** 3,                 # x * y ** 3,
        2 * x * y ** 3,         # x ** 2 * y ** 3
        3 * x ** 2 * y ** 3     # x ** 3 * y ** 3
    ]

def vy(x, y):
    return [0,
            0,
            0,
            0,
            1,
            2 * y,
            3 * y ** 2,
            x,
            x ** 2,
            x ** 3,
            2 * x * y,
            2 * x ** 2 * y,
            2 * x ** 3 * y,
            3 * x * y ** 2,
            3 * x ** 2 * y ** 2,
            3 * x ** 3 * y ** 2]

def vxy(x, y):
    return [0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            2 * x,
            3 * x ** 2,
            2 * y,
            4 * x * y,
            6 * x ** 2 * y,
            3 * y ** 2,
            6 * x * y ** 2,
            9 * x ** 2 * y ** 2]

def partB():
    # 0: a00, 1: a10, 2: a20, 3: a30,
    # 4: a01, 5: a02, 6: a03, 7: a11,
    # 8: a21, 9: a31, 10: a12, 11: a22,
    # 12: a32, 13:a13, 14: a23, 15: a33
    b = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # p(0, 0)
         [1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # p(0, 1)
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # p(1, 0)
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # p(1, 1)
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # px(0, 0)
         [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # px(0, 1)
         [0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # px(1, 0)
         [0, 1, 2, 3, 0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3],  # px(1, 1)
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # py(0, 0)
         [0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # py(0, 1)
         [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # py(1, 0)
         [0, 0, 0, 0, 1, 2, 3, 1, 1, 1, 2, 2, 2, 3, 3, 3],  # py(1, 1)
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # pxy(0, 0)
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0],  # pxy(0, 1)
         [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0],  # pxy(1, 0)
         [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 2, 4, 6, 3, 6, 9]  # pxy(1, 1)
         ]

    print(np.array(b))
    # piv, lu = lu_factor(np.array(b))
    P, L, U = la.lu(np.array(b))
    np.dot(P.T, b)
    np.dot(L, U)
    print("P:")
    print(P)
    print("L:")
    print(np.array(L))
    print("U:")
    print(U)

   


print("Part A:")
partA()
print()
print("Part B:")
partB()


def Bicubic_Interpolation(f, interpolation_of_x, interpolation_of_y, lu,piv, h):
    def fx(x, y):
        return (f(x+h, y) - f(x-h, y))/(2*h)

    def fy(x, y):
        return (f(x, y+h) - f(x, y-h))/(2*h)

    def fxy(x, y):
        return (f(x+h, y+h) - f(x+h, y-h)-f(x-h, y+h) + f(x-h, y-h))/(4*h**2)

    right_side_f_vec = np.array([
        f(0, 0),
        f(1, 0),
        f(0, 1),
        f(1, 1),
        fx(0, 0),
        fx(1, 0),
        fx(0, 1),
        fx(1, 1),
        fy(0, 0),
        fy(1, 0),
        fy(0, 1),
        fy(1, 1),
        fxy(0, 0),
        fxy(1, 0),
        fxy(0, 1),
        fxy(1, 1),
    ])  

    alpha = lu_solve((lu, piv), right_side_f_vec)
    p = np.dot(alpha.T, np.array)([
        1,
        interpolation_of_x,
        interpolation_of_x ** 2,
        interpolation_of_x ** 3,
        interpolation_of_y,
        interpolation_of_y ** 2,
        interpolation_of_y ** 3,
        interpolation_of_x * interpolation_of_y,
        interpolation_of_x ** 2 * interpolation_of_y,
        interpolation_of_x ** 3 * interpolation_of_y,
        interpolation_of_x * interpolation_of_y ** 2,
        interpolation_of_x ** 2 * interpolation_of_y ** 2,
        interpolation_of_x ** 3 * interpolation_of_y ** 2,
        interpolation_of_x * interpolation_of_y ** 3,
        interpolation_of_x ** 2 * interpolation_of_y ** 3,
        interpolation_of_x ** 3 * interpolation_of_y ** 3
    ])

    px = np.dot(alpha.T, np.array)([
        0,
        1,
        2 * interpolation_of_x,
        3 * interpolation_of_x ** 2,
        0,
        0,
        0,
        interpolation_of_y,
        2 * interpolation_of_x * interpolation_of_y,
        3 * interpolation_of_x ** 2 * interpolation_of_y,
        interpolation_of_y ** 2,
        2 * interpolation_of_x * interpolation_of_y ** 2,
        3 * interpolation_of_x ** 2 * interpolation_of_y ** 2,
        interpolation_of_y ** 3,
        2 * interpolation_of_x * interpolation_of_y ** 3,
        3 * interpolation_of_x ** 2 * interpolation_of_y ** 3
    ])

    py = np.dot(alpha.T,np.array)([
        0,
        0,
        0,
        0,
        1,
        2 * interpolation_of_y,
        3 * interpolation_of_y ** 2,
        interpolation_of_x,
        interpolation_of_x ** 2,
        interpolation_of_x ** 3,
        2 * interpolation_of_x * interpolation_of_y,
        2 * interpolation_of_x ** 2 * interpolation_of_y,
        2 * interpolation_of_x ** 3 * interpolation_of_y,
        3 * interpolation_of_x * interpolation_of_y ** 2,
        3 * interpolation_of_x ** 2 * interpolation_of_y ** 2,
        3 * interpolation_of_x ** 3 * interpolation_of_y ** 2
    ])

    pxy = np.dot(alpha.T, np.array)([
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        2 * interpolation_of_x,
        3 * interpolation_of_x ** 2,
        2 * interpolation_of_y,
        4 * interpolation_of_x * interpolation_of_y,
        6 * interpolation_of_x ** 2 * interpolation_of_y,
        3 * interpolation_of_y ** 2,
        6 * interpolation_of_x * interpolation_of_y ** 2,
        9 * interpolation_of_x ** 2 * interpolation_of_y ** 2

    ])
    interpolation = [p, px, py, pxy]

   
   #Calculating the error

    approx_centered_difference = [f(interpolation_of_x, interpolation_of_y), fx(interpolation_of_x, interpolation_of_y), fy(
        interpolation_of_x, interpolation_of_y), fxy(interpolation_of_x, interpolation_of_y)]
    error = np.subtract(approx_centered_difference, interpolation)
    inf_norm = abs[max(error)]
    one_norm = sum([abs(x) for x in error])
    two_norm = np.sqrt(sum([x**2 for x in error]))
    norms = [inf_norm, one_norm, two_norm]

    return interpolation, error, norms

    # interpolation for part e
# interpolation_j, error_j, norms_j = Bicubic_Interpolation(lambda x,y: (y**2*(np.e**(-(x**2 + y**2)))),0.5, 0.5, lu, piv, 0.01)
# print("Interpolation:", Interpolation)
# print("Error: ", error)
# print("norms:", norms)


# interpolation for part f
# interpolation_g, error_g, norms_g = Bicubic_Interpolation(lambda x,y: (x**2*(np.tanh(x*y))),0.5, 0.5, lu, piv, 0.01)
# print("Interpolation:", Interpolation)
# print("Error: ", error)
# print("norms:", norms)

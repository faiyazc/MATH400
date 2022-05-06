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

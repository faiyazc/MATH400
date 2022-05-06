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
        # delta = part_piv_ge(J, -f)
        delta = la.solve(J, -f) #faster method
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

#Test Calls
# print(abs(jacobian(A1, [1,1,1], 0.001)[0:3,0]))
# print(jacobian(A1, [1,1,1], 0.001).shape[0])
# print(range(0,2))
# print(max(abs(jacobian(A1, [1,1,1], 0.001)[0:3,0])))

x = [2,0,2]
n = 0

print("A2:")
print(Newton(A2, x, 5e-6, 0.001, n))


x = [2,2,2]
n = 0

print("A3:")
print(Newton(A3, x, 5e-6, 0.001, n))

# print(jacobian(A2, [1,1,1], 0.001))

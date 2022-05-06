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
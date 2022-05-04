import numpy as np
import sympy as sp

def bicubic_interpolation():
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
    a00, a10, a20, a30 = sp.symbols('a00, a10, a20, a30')
    a01, a11, a21, a31 = sp.symbols('a01, a11, a21, a31')
    a02, a12, a22, a32 = sp.symbols('a02, a12, a22, a32')
    a03, a13, a23, a33 = sp.symbols('a03, a13, a23, a33')

    a = [a00, a10, a20, a30,
         a01, a02, a03, a11,
         a21, a31, a12, a22,
         a32, a13, a23, a33]

    p = np.dot(a, v(x, y))
    return p


def interpolate_x(x, y):
    a00, a10, a20, a30 = sp.symbols('a00, a10, a20, a30')
    a01, a11, a21, a31 = sp.symbols('a01, a11, a21, a31')
    a02, a12, a22, a32 = sp.symbols('a02, a12, a22, a32')
    a03, a13, a23, a33 = sp.symbols('a03, a13, a23, a33')

    a = [a00, a10, a20, a30,
         a01, a02, a03, a11,
         a21, a31, a12, a22,
         a32, a13, a23, a33]

    p = np.dot(a, vx(x, y))
    return p

def interpolate_y(x, y):
    a00, a10, a20, a30 = sp.symbols('a00, a10, a20, a30')
    a01, a11, a21, a31 = sp.symbols('a01, a11, a21, a31')
    a02, a12, a22, a32 = sp.symbols('a02, a12, a22, a32')
    a03, a13, a23, a33 = sp.symbols('a03, a13, a23, a33')

    a = [a00, a10, a20, a30,
         a01, a02, a03, a11,
         a21, a31, a12, a22,
         a32, a13, a23, a33]

    p = np.dot(a, vy(x, y))
    return p

def interpolate_xy(x, y):
    a00, a10, a20, a30 = sp.symbols('a00, a10, a20, a30')
    a01, a11, a21, a31 = sp.symbols('a01, a11, a21, a31')
    a02, a12, a22, a32 = sp.symbols('a02, a12, a22, a32')
    a03, a13, a23, a33 = sp.symbols('a03, a13, a23, a33')

    a = [a00, a10, a20, a30,
         a01, a02, a03, a11,
         a21, a31, a12, a22,
         a32, a13, a23, a33]

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

bicubic_interpolation()

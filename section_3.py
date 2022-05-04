# Refrence used : github and wikipedia

import sympy as syp
import numpy as np
import pandas as pd

x = syp.Symbol('x')
y = syp.Symbol('y')
a00, a01, a02, a03, a10, a11, a12, a13 = syp.symbols('a00 a01 a02 a03 a10 a11 a12 a13')
a20, a21, a22, a23, a30, a31, a32, a33 = syp.symbols('a20 a21 a22 a23 a30 a31 a32 a33')

p = a00 + a01*y + a02*y**2 + a03*y**3\
+ a10*x + a11*x*y + a12*x*y**2 + a13*x*y**3\
+ a20*x**2 + a21*x**2*y + a22*x**2*y**2 + a23*x**2*y**3\
+ a30*x**3 + a31*x**3*y + a32*x**3*y**2 + a33*x**3*y**3 

px = syp.diff(p, x)
py = syp.diff(p, y)
pxy = syp.diff(p, x, y)

df = pd.DataFrame(columns=['function', 'evaluation'])

for i in range(2):
    for j in range(2):
        function = 'p({}, {})'.format(j,i)
        df.loc[len(df)] = [function, p.subs({x:j, y:i})]
for i in range(2):
    for j in range(2):
        function = 'px({}, {})'.format(j,i)
        df.loc[len(df)] = [function, px.subs({x:j, y:i})]
for i in range(2):
    for j in range(2):
        function = 'py({}, {})'.format(j,i)
        df.loc[len(df)] = [function, py.subs({x:j, y:i})]
for i in range(2):
    for j in range(2):
        function = 'pxy({}, {})'.format(j,i)
        df.loc[len(df)] = [function, pxy.subs({x:j, y:i})]

eqns = df['evaluation'].tolist()
symbols = [a00,a01,a02,a03,a10,a11,a12,a13,a20,a21,a22,a23,a30,a31,a32,a33]
A = (syp.linear_eq_to_matrix(eqns, *symbols)[0])
Inverse = np.array(A.inv()).astype(np.float64)

# print(df)
# print (A)
print (Inverse)


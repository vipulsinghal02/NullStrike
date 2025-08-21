import sympy as sym

""" 
The model states, outputs, inputs, parameters,
and dynamic equations must be defined as vectors of symbolic variables; the names of
these vectors must follow the specific convention shown in Table 1. Important: x, p, u,
w, f, h are reserved names, which must be used for the variables and/or functions listed
in Table 1 and cannot be used to name any other variables. However, it is possible to
use variants of them, e.g. x1, x2, p1, p2, u1, u2, w1, w2, f1, f2, h1, h2.




"""
# C2M
x1 = sym.Symbol('x1')  # define the symbolic variable x1
x2 = sym.Symbol('x2')  # define the symbolic variable x2
x3 = sym.Symbol('x3')  # define the symbolic variable x3
x4 = sym.Symbol('x4')  # define the symbolic variable x4
x5 = sym.Symbol('x5')  # define the symbolic variable x5
x6 = sym.Symbol('x6')  # define the symbolic variable x6
x = [[x1], [x2]]

# 1 output
h = [x1]
# 1 known input
u1 = sym.Symbol('u1')  # define the symbolic variable u1
u = [u1]
# 0 unknown inputs
w = []
# 4 unknown parameters
p = [[x3], [x4], [x5], [x6]]
# dynamic equations
f = [[-(x3+x4)*x1+x5*x2+x6*u1], [x4*x1-x5*x2]]
variables_locales = locals().copy()

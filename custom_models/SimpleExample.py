import sympy as sym

# Simple two-state model
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2') 
x = [[x1], [x2]]

# One output (measure first state)
h = [x1]

# One input
u1 = sym.Symbol('u1')
u = [u1]

# No unknown inputs
w = []

# Two parameters
k1 = sym.Symbol('k1')
k2 = sym.Symbol('k2')
p = [[k1], [k2]]

# Simple dynamics: x1' = -k1*x1 + u1, x2' = k2*x1
f = [[-k1*x1 + u1], [k2*x1]]

variables_locales = locals().copy()

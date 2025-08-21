import sympy as sym

# calibration model, just one extract. 
x1A = sym.Symbol('x1A')  
x2A = sym.Symbol('x2A') 
x3A = sym.Symbol('x3A')  
x4A = sym.Symbol('x4A')  
x5A = sym.Symbol('x5A')  
x16A = sym.Symbol('x16A')  
x1B = sym.Symbol('x1B')  
x2B = sym.Symbol('x2B') 
x3B = sym.Symbol('x3B')  
x4B = sym.Symbol('x4B')  
x5B = sym.Symbol('x5B')  
x16B = sym.Symbol('x16B')

x = [[x1A], [x2A], [x3A], [x4A], [x5A], [x16A], [x1B], [x2B], [x3B], [x4B], [x5B], [x16B]]

# 2 output
h = [[x16A], [x16B]]

# 0 input
u = []

# 0 unknown inputs
w = []

x6 = sym.Symbol('x6')
x7 = sym.Symbol('x7')
x8 = sym.Symbol('x8')
x9 = sym.Symbol('x9')
x10 = sym.Symbol('x10')
x11 = sym.Symbol('x11')
x12 = sym.Symbol('x12')
x13 = sym.Symbol('x13')
x14 = sym.Symbol('x14')
x15 = sym.Symbol('x15')

# 10 unknown parameters
p = [[x6], [x7], [x8], [x9], [x10], [x11], [x12], [x13], [x14], [x15]]
dTA = 1
dTB = 10
dG=5
# dynamic equations 
f = [[-x6*(x15-dTA+x1A-dG+x2A+x5A)*x1A+(x7+x8)*(dTA-x1A)],
    [x9*(x15-dTA+x1A -dG+x2A+x5A)*x2A+(x10+x8)*(dG-x2A-x5A)-x11*x2A*x4A+x12*x5A],
    [x8*(dTA-x1A)-2*x13*x3A**2+2*x14*x4A],
    [-x11*x2A*x4A+x12*x5A+2*x13*x3A**2-2*x14*x4A], 
    [x11*x2A*x4A-x12*x5A], 
    [x8*(dG-x2A-x5A)],
    [-x6*(x15-dTB+x1B-dG+x2B+x5B)*x1B+(x7+x8)*(dTB-x1B)],
    [x9*(x15-dTB+x1B -dG+x2B+x5B)*x2B+(x10+x8)*(dG-x2B-x5B)-x11*x2B*x4B+x12*x5B],
    [x8*(dTB-x1B)-2*x13*x3B**2+2*x14*x4B],
    [-x11*x2B*x4B+x12*x5B+2*x13*x3B**2-2*x14*x4B], 
    [x11*x2B*x4B-x12*x5B], 
    [x8*(dG-x2B-x5B)]]

variables_locales = locals().copy()

import sympy as sym

# calibration model, just one extract. 
x1A1 = sym.Symbol('x1A1')  
x2A1 = sym.Symbol('x2A1')  
x1B1 = sym.Symbol('x1B1')  
x2B1 = sym.Symbol('x2B1')  
x1C1 = sym.Symbol('x1C1')  
x2C1 = sym.Symbol('x2C1')  
x1A2 = sym.Symbol('x1A2')  
x2A2 = sym.Symbol('x2A2')  
x1B2 = sym.Symbol('x1B2')  
x2B2 = sym.Symbol('x2B2')  
x1C2 = sym.Symbol('x1C2')  
x2C2 = sym.Symbol('x2C2') 



x3 = sym.Symbol('x3')  
x4 = sym.Symbol('x4')  
x51 = sym.Symbol('x51') 
x52 = sym.Symbol('x52') 
e1 = sym.Symbol('e1')  
e2 = sym.Symbol('e2')  

x = [[x1A1], [x2A1], [x1B1], [x2B1], [x1C1], [x2C1], 
    [x1A2], [x2A2], [x1B2], [x2B2], [x1C2], [x2C2]]

# 1 output
h = [[x2A1], [x2B1],[x2C1], [x2A2], [x2B2],[x2C2]]

# 0 input
u = []

# 0 unknown inputs
w = []

# 4 unknown parameters
p = [[e1],[e2], [x3], [x4], [x51], [x52]]

# dynamic equations 
f = [
    [-x3*(e1-1+x1A1)*x1A1+(x4+x51)*(1-x1A1)], 
    [x51*(1-x1A1)],
    [-x3*(e1-5+x1B1)*x1A1+(x4+x51)*(5-x1B1)], [x51*(5-x1B1)],
    [-x3*(e1-20+x1C1)*x1C1+(x4+x51)*(20-x1C1)], [x51*(20-x1C1)],
    [-x3*(e2-1+x1A2)*x1A2+(x4+x52)*(1-x1A2)], [x52*(1-x1A2)],
    [-x3*(e2-5+x1B2)*x1A2+(x4+x52)*(5-x1B2)], [x52*(5-x1B2)],
    [-x3*(e2-20+x1C2)*x1C2+(x4+x52)*(20-x1C2)], [x51*(20-x1C2)]
    ]

variables_locales = locals().copy()

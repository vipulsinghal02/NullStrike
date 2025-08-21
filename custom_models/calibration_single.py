import sympy as sym

# calibration model, just one extract. 
DNA = sym.Symbol('DNA')  # define the symbolic variable x1
GFP = sym.Symbol('GFP')  # define the symbolic variable x2
kf = sym.Symbol('kf')  # define the symbolic variable x3
kr = sym.Symbol('kr')  # define the symbolic variable x4
kcat = sym.Symbol('kcat')  # define the symbolic variable x5
Etot = sym.Symbol('Etot')  # define the symbolic variable x5

x = [[DNA], [GFP]]

# 1 output
h = [GFP]

# 0 input
u = []

# 0 unknown inputs
w = []

# 4 unknown parameters
p = [[Etot], [kf], [kr], [kcat]]
# p = [[e], [x3], [x4], [x5]]
# dynamic equations 
f = [[-kf*(Etot-10+DNA)*DNA+(kr+kcat)*(10-DNA)], [kcat*(10-DNA)]]

variables_locales = locals().copy()

from math import inf

modelname = 'C2M_unknown_input_w'
checkObser = 1
maxLietime = inf
nnzDerU = []  # No known inputs
nnzDerW = [inf]  # Unknown input can vary arbitrarily
prev_ident_pars = []

MANIFOLD_PLOTTING = {
    'enabled': True,
    'max_2d_plots': 10,
    'max_3d_plots': 5,
    'resolution': 50
}
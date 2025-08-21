#==============================================================================
#THE USER CAN DEFINE THE PROBLEM AND SET OPTIONS IN THE FOLLOWING LINES:
#==============================================================================
import sympy as sym
from math import inf
###############################################################################
# (1) NAME OF THE MODEL TO BE STUDIED:
modelname = 'calibration_single'
##############################################################################
# (2) FISPO ANALYSIS OPTIONS:
checkObser = 1    # check state observability, i.e. identifiability of initial conditions (1 = yes; 0 = no).
maxLietime = inf  # max. time allowed for calculating 1 Lie derivative (seconds)
nnzDerU = [0] # Number of non-zero known input derivatives in each experiment (Rows=inputs;Columns=experiments)
nnzDerW    = [inf] # numbers of nonzero derivatives of the unmeasured inputs (w); may be 'inf'
###############################################################################
# (3) KNOWN/IDENTIFIABLE PARAMETERS (parameters assumed known, or already classified as identifiable):
prev_ident_pars = []
# # An example of use would be:
    # x2 = sym.Symbol('x2')
    # x5 = sym.Symbol('x5')
    # prev_ident_pars = [x2, x5]

# --- Manifold plotting config ---
MANIFOLD_PLOTTING = {
    # Per-symbol axis ranges. Use either a tuple or a dict.
    # Tuple: (min, max, num) — linear by default
    # Dict:  {"min":..., "max":..., "num":..., "scale":"linear"|"log"}
    "var_ranges": {
        # examples:
        # "q2": (-5.0, 5.0, 100),
        # "p2": {"min": 1e-3, "max": 10.0, "num": 120, "scale": "log"},
        # "p4": {"min": 1e-3, "max": 10.0, "num": 120, "scale": "log"},
    },

    # If a symbol is listed here and no explicit var_ranges entry exists,
    # we’ll use `default_positive_range` (and log if enabled).
    "positive_symbols": [
        # "p2", "p4", "k1", ...
    ],

    # Defaults when no per-symbol range is given
    # "default_var_range": (-5.0, 5.0, 100),
    # "default_positive_range": (1e-3, 10.0, 120),
    # "log_for_positive": True,   # use logspace for positive-only if no explicit scale is provided

    "default_positive_range": (1e-3, 10.0, 120),
    "log_for_positive": True,
    "default_var_range": (-5.0, 5.0, 100),
    "use_negative_fallback": False,  # Only True if you really want -5 to 5


    # How many z-slices to draw in implicit 3D fallback
    "z_slices": 15,

    # Add *extra* 3D triplets to plot (in addition to auto-selected ones)
    # Symbol names as strings; these will be resolved to the model's existing SymPy symbols
    "extra_triplets_3d": [
        # ("q2", "p2", "p4"),
        # ("x1", "x3", "p4"),
    ],

    # Add *extra* 2D pairs to plot (in addition to auto-selected ones)
    "extra_pairs_2d": [
        # ("q2", "p2"),
    ],

    # Remove/raise any plotting caps. None means no cap.
    "max_triplets_3d": None,
    "max_pairs_2d": None,

    # Custom numeric values for *non-plotted* parameters/states/inputs
    # used when substituting constants during plotting.
    # Keys are symbol names; values are floats.
    "param_overrides": {
        # "p4": 2.0,
        # "k1": 0.5,
    },
}



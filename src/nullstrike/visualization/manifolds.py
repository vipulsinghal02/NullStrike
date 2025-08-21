"""
3D manifold visualization for nullspace analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sympy as sym
from sympy import Eq
from datetime import datetime
import os

# Import utilities
from ..utils import get_results_dir

# Move your visualize_nullspace_manifolds function and related helper functions here
# from enhanced_subspace.py


def _range_from_cfg(sym_obj, cfg):
    """Return a 1D numpy array for the symbol based on cfg ranges."""
    name = str(sym_obj)

    # 1. Explicit per-symbol range
    if name in cfg.get("var_ranges", {}):
        spec = cfg["var_ranges"][name]
        if isinstance(spec, tuple) and len(spec) == 3:
            lo, hi, num = spec
            return np.linspace(lo, hi, int(num))
        elif isinstance(spec, dict):
            lo = spec["min"]; hi = spec["max"]; num = int(spec.get("num", 100))
            scale = spec.get("scale", "linear").lower()
            if scale == "log":
                lo = max(lo, 1e-12)
                return np.logspace(np.log10(lo), np.log10(hi), num)
            return np.linspace(lo, hi, num)

    # 2. Decide whether to use "positive" as default
    positive = (
        name in set(cfg.get("positive_symbols", []))
        or not cfg.get("use_negative_fallback", False)  # default to positive if not told otherwise
    )

    if positive:
        lo, hi, num = cfg.get("default_positive_range", (1e-3, 10.0, 120))
        if cfg.get("log_for_positive", True):
            lo = max(lo, 1e-12)
            return np.logspace(np.log10(lo), np.log10(hi), int(num))
        return np.linspace(lo, hi, int(num))

    # 3. Explicit negative fallback
    lo, hi, num = cfg.get("default_var_range", (-5.0, 5.0, 100))
    return np.linspace(lo, hi, int(num))


def _z_slices_from_cfg(z_sym, cfg):
    """Return z-slice positions for implicit 3D plotting."""
    Z = _range_from_cfg(z_sym, cfg)
    # Pick ~cfg['z_slices'] evenly spaced indices from Z
    n_slices = int(cfg.get("z_slices", 15))
    if len(Z) <= n_slices:
        return Z
    idx = np.linspace(0, len(Z)-1, n_slices).astype(int)
    return Z[idx]


def _value_from_cfg(sym_obj, cfg):
    """Numeric default for non-plotted symbols; prefer overrides; else midpoint of range; else 1.0."""
    name = str(sym_obj)
    overrides = cfg.get("param_overrides", {})
    if name in overrides:
        return float(overrides[name])

    # try midpoint of configured range, geometric mean if log
    r = cfg.get("var_ranges", {}).get(name, None)
    if isinstance(r, tuple) and len(r) == 3:
        lo, hi, _ = r
        return 0.5 * (float(lo) + float(hi))
    elif isinstance(r, dict):
        lo = float(r["min"]); hi = float(r["max"])
        scale = r.get("scale", "linear").lower()
        if scale == "log":
            lo = max(lo, 1e-12)
            return float(np.sqrt(lo * hi))  # geometric mean
        return 0.5 * (lo + hi)

    # fall back to defaults
    if name in set(cfg.get("positive_symbols", [])):
        lo, hi, _ = cfg.get("default_positive_range", (1e-3, 10.0, 120))
        if cfg.get("log_for_positive", True):
            lo = max(lo, 1e-12)
            return float(np.sqrt(lo * hi))
        return 0.5 * (float(lo) + float(hi))

    # final fallback
    return 1.0


def _resolve_symbol(name, universe):
    """Return the *existing* SymPy symbol with this string name from universe (states+params+inputs)."""
    for s in universe:
        if str(s) == name:
            return s
    raise KeyError(f"Symbol '{name}' not found in symbol universe.")

    
def visualize_nullspace_manifolds(
    nullspace_results,
    state_symbols,
    param_symbols,
    symbol_universe,
    plot_cfg=None,
    input_symbols=None,
    model_name="Model",
    shared_timestamp=None):
    """
    Visualize the unidentifiable manifolds as 3D surfaces.
    
    For each nullspace vector, find all combinations of 3 variables that are involved,
    and plot the constraint surface in 3D.
    """
    import matplotlib.pyplot as plt

    import numpy as np
    from datetime import datetime
    plot_cfg = plot_cfg or {}

    def _create_results_structure(model_name, timestamp):
        """Create organized directory structure for results."""
        import os
        
        base_dir = get_results_dir() / model_name / timestamp
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (base_dir / "manifolds_3d").mkdir(exist_ok=True)
        (base_dir / "manifolds_2d").mkdir(exist_ok=True)
        (base_dir / "graphs").mkdir(exist_ok=True)
        
        return base_dir

    # Resolve extra triplets/pairs by name to symbols
    extras_3d = []
    for t in plot_cfg.get("extra_triplets_3d", []):
        a,b,c = t
        extras_3d.append((_resolve_symbol(a, symbol_universe),
                          _resolve_symbol(b, symbol_universe),
                          _resolve_symbol(c, symbol_universe)))

    extras_2d = []
    for p in plot_cfg.get("extra_pairs_2d", []):
        a,b = p
        extras_2d.append((_resolve_symbol(a, symbol_universe),
                          _resolve_symbol(b, symbol_universe)))

    # Map every symbol to its global index in all_symbols (to pull the correct coeff)
    if input_symbols is None:
        input_symbols = []
    all_symbols = state_symbols + param_symbols + input_symbols
    sym_to_idx_global = {str(s): i for i, s in enumerate(all_symbols)}

    # Combine + dedup for the multiplets to plot (preserve order)
    def _uniq(seq):
        seen = set()
        out = []
        for item in seq:
            key = tuple(map(str, item))
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out
    
    

    if shared_timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    else:
        timestamp = shared_timestamp
    results_base = _create_results_structure(model_name, timestamp)
    
    # Configurable caps (None => no cap)
    max_triplets_3d = plot_cfg.get("max_triplets_3d", None)
    max_pairs_2d    = plot_cfg.get("max_pairs_2d", None)
    
    print(f"\nVisualizing nullspace manifolds for {model_name}...")
    
    plot_count = 0
    max_plots = 10  # Limit total number of plots
    
    for vec_idx, null_vec in enumerate(nullspace_results['nullspace_basis']):
        print(f"\n--- Nullspace Vector {vec_idx + 1} ---")
        
        # Find variables involved in this nullspace vector
        involved_indices = [i for i in range(len(null_vec)) if null_vec[i] != 0]
        involved_symbols = [all_symbols[i] for i in involved_indices if i < len(all_symbols)]
        involved_coeffs = [null_vec[i] for i in involved_indices if i < len(all_symbols)]
        
        print(f"Involved variables: {involved_symbols}")
        print(f"Coefficients: {involved_coeffs}")
        
        from itertools import combinations

        # Build a dict from symbol -> coeff for this vector
        sym_to_coeff = {str(s): c for s, c in zip(involved_symbols, involved_coeffs)}

        # --- 3D TRIPLETS ---
        triples_auto = [tuple(involved_symbols[i] for i in idxs)
                        for idxs in combinations(range(len(involved_symbols)), 3)]

        # Filter extras_3d to those fully present in this vector
        triples_extras = []
        for (a, b, c) in extras_3d:
            names = {str(a), str(b), str(c)}
            if names.issubset({str(s) for s in involved_symbols}):
                # use the existing SymPy symbols from involved_symbols for exact identity
                remapped = tuple(next(s for s in involved_symbols if str(s) == n) for n in [str(a), str(b), str(c)])
                triples_extras.append(remapped)

        # De-duplicate while preserving order
        triples_merged = _uniq(triples_auto + triples_extras)

        # Apply cap if configured
        if isinstance(max_triplets_3d, int):
            triples_merged = triples_merged[:max_triplets_3d]

        # Plot each triple
        for combo_idx, triple_symbols in enumerate(triples_merged):
            triple_coeffs = [sym_to_coeff[str(s)] for s in triple_symbols]
            try:
                plot_3d_constraint_surface(
                    triple_symbols, triple_coeffs, null_vec, involved_indices,
                    vec_idx, combo_idx, model_name, timestamp,
                    plot_cfg=plot_cfg
                )
            except Exception as e:
                print(f"Could not plot {triple_symbols}: {e}")

        # --- 2D PAIRS ---
        pairs_auto = [tuple(involved_symbols[i] for i in idxs)
                    for idxs in combinations(range(len(involved_symbols)), 2)]

        pairs_extras = []
        for (a, b) in extras_2d:
            names = {str(a), str(b)}
            if names.issubset({str(s) for s in involved_symbols}):
                remapped = tuple(next(s for s in involved_symbols if str(s) == n) for n in [str(a), str(b)])
                pairs_extras.append(remapped)

        pairs_merged = _uniq(pairs_auto + pairs_extras)

        if isinstance(max_pairs_2d, int):
            pairs_merged = pairs_merged[:max_pairs_2d]

        for pair_symbols in pairs_merged:
            pair_coeffs = [sym_to_coeff[str(s)] for s in pair_symbols]
            try:
                plot_2d_constraint_line(
                    pair_symbols, pair_coeffs,
                    vec_idx, model_name, timestamp,
                    plot_cfg=plot_cfg
                )
            except Exception as e:
                print(f"Could not plot 2D constraint for {pair_symbols}: {e}")

def _get_axis_label(sym_obj, plot_cfg):
    """Return axis label, with log() prefix if symbol uses log scale."""
    name = str(sym_obj)
    
    # Check if this symbol is configured for log scale
    ranges = plot_cfg.get("var_ranges", {})
    if name in ranges and isinstance(ranges[name], dict):
        if ranges[name].get("scale", "linear").lower() == "log":
            return f"log({name})"
    
    # Check if it's a positive symbol with log_for_positive=True
    if (name in set(plot_cfg.get("positive_symbols", [])) and 
        plot_cfg.get("log_for_positive", True)):
        return f"log({name})"
    
    return name
    
def plot_3d_constraint_surface(symbols, coeffs, full_null_vec, involved_indices, 
                               vec_idx, combo_idx, model_name, timestamp,
                               plot_cfg=None):
    """
    Plot the 3D constraint surface for three variables.
    
    """
    plot_cfg = plot_cfg or {}
    if len(symbols) != 3 or len(coeffs) != 3:
        return
    
    # Create symbolic constraint equation
    constraint_expr = sum(c * s for c, s in zip(coeffs, symbols))   
    print(f"  Plotting 3D surface for {symbols}")
    print(f"  Constraint: {constraint_expr} = 0")
    
    free_syms = set(constraint_expr.free_symbols)
    param_syms = sorted(free_syms - set(symbols), key=str)
    param_values = {p: _value_from_cfg(p, plot_cfg) for p in param_syms}
    expr_sub = sym.simplify(constraint_expr.subs(param_values))

    
    expr_simpl = sym.simplify(expr_sub)
    if expr_simpl == 0:
        print("    Constraint simplifies to 0=0; manifold is entire R^3 for chosen params. Skipping plot.")
        return
    if not (set(expr_simpl.free_symbols) & set(symbols)):
        # No dependence on the plotted variables; constant ≠ 0 means no solutions
        if sym.simplify(expr_simpl) != 0:
            print("    Constraint is constant and nonzero for chosen params; no real solutions. Skipping plot.")
            return


    # Try to solve for one variable to get a parametric surface Z(X,Y)
    vars3 = list(symbols)
    solutions = None
    solve_idx = None
    for idx_try in [2, 1, 0]:  # prefer solving for the last symbol; fall back if needed
        try:
            sols = sym.solve(Eq(expr_sub, 0), vars3[idx_try], dict=True)
            if sols:
                solutions = sols
                solve_idx = idx_try
                break
        except Exception:
            pass

# -------------------- # 
    # Plot settings - create 2x2 subplot for 4 orientations
    fig = plt.figure(figsize=(16, 12))

    # Define 4 different viewing angles
    view_angles = [
        (30, 45),   # default
        (60, 135),  # rotated
        (45, 225),  # back view
        (75, 315)   # top-angled
    ]

    view_names = ['Default', 'Side', 'Back', 'Top']

    def mask_invalid(A):
        return np.where(np.isfinite(A), A, np.nan)
    
    for subplot_idx, ((elev, azim), view_name) in enumerate(zip(view_angles, view_names)):
        ax = fig.add_subplot(2, 2, subplot_idx + 1, projection='3d')
        
        ax.set_xlabel(_get_axis_label(symbols[0], plot_cfg))
        ax.set_ylabel(_get_axis_label(symbols[1], plot_cfg))
        ax.set_zlabel(_get_axis_label(symbols[2], plot_cfg))
        
        # Set the viewing angle
        ax.view_init(elev=elev, azim=azim)
        
        # [Insert the same plotting logic here - either parametric or implicit]
        # This is the same plotting code you had before, just repeated for each subplot
        
        if solutions:            
            idx_other = [i for i in [0,1,2] if i != solve_idx]
            u_sym, v_sym = vars3[idx_other[0]], vars3[idx_other[1]]
            
            for sol in solutions:
                sol_expr = sym.simplify(sol[vars3[solve_idx]])
                f = sym.lambdify((u_sym, v_sym), sol_expr, 'numpy')
                
                # mask singularities in the solved expression
                try:
                    num_p, den_p = sym.together(sol_expr).as_numer_denom()
                    den_p = sym.simplify(den_p)
                    Duv = None if den_p == 1 else sym.lambdify((u_sym, v_sym), den_p, 'numpy')
                except Exception:
                    Duv = None

                var_range_u = _range_from_cfg(u_sym, plot_cfg)
                var_range_v = _range_from_cfg(v_sym, plot_cfg)
                U, V = np.meshgrid(var_range_u, var_range_v)
                
                try:
                    W = f(U, V)
                    if Duv is not None:
                        den_vals = Duv(U, V)
                        mask = np.isfinite(den_vals) & (np.abs(den_vals) > 1e-12)
                        W = np.where(mask, W, np.nan) 

                    if np.iscomplexobj(W):
                        W = np.where(np.abs(np.imag(W)) < 1e-9, np.real(W), np.nan)
                    W = mask_invalid(W)
                except Exception:
                    continue
                
                # Place (U,V,W) onto the correct axes based on which var was solved
                if solve_idx == 2:
                    X, Y, Z = U, V, W
                elif solve_idx == 1:
                    X, Y, Z = U, W, V
                else:
                    X, Y, Z = W, U, V
                
                surf = ax.plot_surface(X, Y, Z, alpha=0.7, cmap='viridis', linewidth=0)
        else:
            # [Same implicit plotting code as before]
            # ... implicit contour plotting code

            # Fallback: implicit contours by z-slices (draw contour lines where F(x,y,z0)=0 at multiple z0)
            # Choose z as the 3rd symbol for slicing
            x_sym, y_sym, z_sym = symbols
            Fxyz = sym.lambdify((x_sym, y_sym, z_sym), expr_sub, 'numpy')

            Xv = _range_from_cfg(x_sym, plot_cfg)
            Yv = _range_from_cfg(y_sym, plot_cfg)
            X, Y = np.meshgrid(Xv, Yv)

            z_slices = _z_slices_from_cfg(z_sym, plot_cfg)
            ax.set_zlim(float(np.min(z_slices)), float(np.max(z_slices)))

            
            try:
                num, den = sym.together(expr_sub).as_numer_denom()
                den = sym.simplify(den)
                Dxyz = None if den == 1 else sym.lambdify((x_sym, y_sym, z_sym), den, 'numpy')
            except Exception:
                Dxyz = None
                

            for z0 in z_slices:
                try:
                    F_vals = Fxyz(X, Y, z0)
                    if Dxyz is not None:
                        den_vals = Dxyz(X, Y, z0)
                        F_vals = np.where(np.isfinite(den_vals) & (np.abs(den_vals) > 1e-12), F_vals, np.nan)
                                    
                    F_vals = mask_invalid(F_vals)
                    # Contour at level 0, positioned at z=z0
                    # ax.contour(X, Y, z0*np.ones_like(X), F_vals, levels=[0], zdir='z')
                    ax.contour(X, Y, F_vals, levels=[0], zdir='z', offset=z0)
                except Exception:
                    continue
                
            if Dxyz is not None:
                for z0 in z_slices:
                    try:
                        den_vals = Dxyz(X, Y, z0)
                        den_vals = np.where(np.isfinite(den_vals), den_vals, np.nan)
                        ax.contour(X, Y, den_vals, levels=[0], zdir='z', offset=z0, linewidths=1)
                    except Exception:
                        pass


        ax.set_title(f'{view_name} View')
        
        try:
            ax.set_box_aspect((1,1,1))
        except Exception:
            pass
    
    # Main title for the entire figure
    constraint_str = sym.sstr(expr_sub)
    fig.suptitle(f'Unidentifiable Manifold: {constraint_str} = 0\n'
                f'Nullspace Vector {vec_idx + 1}, Combination {combo_idx + 1}',
                fontsize=14)

    plt.tight_layout()
    
    constrained_layout=True
    
    # Get the base directory from the calling function
    results_base = get_results_dir() / model_name / timestamp
    filename = results_base / "manifolds_3d" / f"manifold_3d_vec{vec_idx+1}_combo{combo_idx+1}.png"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=1)
    print(f"    3D manifold saved: {filename}")

    # Also save as vector format
    svg_filename = results_base / "manifolds_3d" / f"manifold_3d_vec{vec_idx+1}_combo{combo_idx+1}.svg"
    plt.savefig(svg_filename, dpi=300, bbox_inches='tight', pad_inches=1, format='svg')
    print(f"    3D manifold saved: {svg_filename}")

    plt.close()


# def plot_2d_constraint_line(symbols, coeffs, vec_idx, model_name, timestamp):
def plot_2d_constraint_line(symbols, coeffs, vec_idx, model_name, timestamp, plot_cfg=None):
    """Plot 2D implicit curve F(x,y)=0 where F = sum(c_i * x_i) with symbolic coefficients.

    - Preserves symbolic structure (no collapsing coefficients to 1.0 for plotted vars).
    - Substitutes only *non-plotted* symbols with defaults (here 1.0; swap in model defaults if available).
    - Masks singularities from rational denominators.
    - Overlays the singular set (denominator == 0) as dashed contours for interpretability.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import sympy as sym
    plot_cfg = plot_cfg or {}
    
    if len(symbols) != 2 or len(coeffs) != 2:
        return

    x_sym, y_sym = symbols

    print(f"  Plotting 2D implicit curve for {symbols}")

    # Full symbolic constraint
    constraint_expr = sym.simplify(sum(c * s for c, s in zip(coeffs, symbols)))
    print(f"  Constraint: {constraint_expr} = 0")

    # Fix *other* symbols (parameters/other states/inputs); keep the plotted vars free
    free_syms = set(constraint_expr.free_symbols)
    keep = {x_sym, y_sym}

    params = sorted(free_syms - keep, key=str)
    param_values = {p: _value_from_cfg(p, plot_cfg) for p in params}
    expr_sub = sym.simplify(constraint_expr.subs(param_values))


    expr_simpl = sym.simplify(expr_sub)

    # Degenerate cases
    if expr_simpl == 0:
        print("    Constraint simplifies to 0=0; manifold is entire R^2 for chosen params. Skipping plot.")
        return
    if not (set(expr_simpl.free_symbols) & keep):
        # No dependence on x,y; if constant != 0 then no solutions
        if sym.simplify(expr_simpl) != 0:
            print("    Constraint is constant and nonzero for chosen params; no real solutions. Skipping plot.")
            return

    # Extract denominator to mask singularities
    try:
        num, den = sym.together(expr_sub).as_numer_denom()
        den = sym.simplify(den)
        Dxy = None if den == 1 else sym.lambdify((x_sym, y_sym), den, 'numpy')
    except Exception:
        Dxy = None

    # Numeric evaluator for F(x,y)
    Fxy = sym.lambdify((x_sym, y_sym), expr_sub, 'numpy')

    # Grid
    Xv = _range_from_cfg(x_sym, plot_cfg)
    Yv = _range_from_cfg(y_sym, plot_cfg)
    X, Y = np.meshgrid(Xv, Yv)

    # Evaluate safely and mask invalids/singularities
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        try:
            Z = Fxy(X, Y)
            if Dxy is not None:
                den_vals = Dxy(X, Y)
                mask = np.isfinite(den_vals) & (np.abs(den_vals) > 1e-12)
                Z = np.where(mask, Z, np.nan)
            if np.iscomplexobj(Z):
                Z = np.where(np.abs(np.imag(Z)) < 1e-9, np.real(Z), np.nan)
            Z = np.where(np.isfinite(Z), Z, np.nan)
        except Exception as e:
            print(f"    Could not evaluate implicit curve: {e}")
            return

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    CS = ax.contour(X, Y, Z, levels=[0], linewidths=2)
    ax.clabel(CS, inline=1, fontsize=8, fmt={0: 'F=0'})

    # Overlay singular set (denominator == 0), if present
    if Dxy is not None:
        try:
            den_vals = Dxy(X, Y)
            den_vals = np.where(np.isfinite(den_vals), den_vals, np.nan)
            ax.contour(X, Y, den_vals, levels=[0], linewidths=1, linestyles='dashed')
        except Exception:
            pass
        
    ax.set_xlabel(_get_axis_label(x_sym, plot_cfg))
    ax.set_ylabel(_get_axis_label(y_sym, plot_cfg))
    
    # ax.set_xlabel(str(x_sym))
    # ax.set_ylabel(str(y_sym))
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Unidentifiable Curve: {sym.sstr(expr_sub)} = 0')

    ax.grid(True, alpha=0.3)

    results_base = get_results_dir() / model_name / timestamp
    filename = results_base / "manifolds_2d" / f"manifold_2d_vec{vec_idx+1}.png"    

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=1)
    print(f"    2D manifold saved: {filename}")

    # Also save as vector format
    svg_filename = results_base / "manifolds_2d" / f"manifold_2d_vec{vec_idx+1}.svg"
    plt.savefig(svg_filename, dpi=300, bbox_inches='tight', pad_inches=1, format='svg')
    print(f"    2D manifold saved: {svg_filename}")

    plt.close(fig)

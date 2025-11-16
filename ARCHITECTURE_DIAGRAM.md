# NullStrike Code Architecture

## High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CLI: nullstrike [model] [options] [--parameters-only]             │
│       └─> src/nullstrike/cli/complete_analysis.py                  │
│                          │                                          │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ANALYSIS ORCHESTRATION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  main() - 3 Method Fallback Strategy:                              │
│                                                                     │
│  Method 1: run_integrated_analysis()                               │
│  ├─> src/nullstrike/analysis/integrated_analysis.py               │
│  │    ├─> Runs StrikePy                                            │
│  │    ├─> Checks checkpoint cache                                  │
│  │    ├─> Performs nullspace analysis                              │
│  │    └─> Generates visualizations                                 │
│  │                                                                  │
│  Method 2: run_step_by_step_analysis()                             │
│  ├─> Falls back if Method 1 fails                                  │
│  │    ├─> Step 1: Run StrikePy separately                          │
│  │    └─> Step 2: Analyze results                                  │
│  │                                                                  │
│  Method 3: analyze_strikepy_results()                              │
│  └─> Most robust - just analyze existing output                    │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌──────────────┐  ┌──────────────────┐
│  CORE ENGINE  │  │   ANALYSIS   │  │  VISUALIZATION   │
└───────────────┘  └──────────────┘  └──────────────────┘
```

## Detailed Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  custom_models/                    custom_options/                 │
│  ├─ C2M.py                        ├─ options_C2M.py                │
│  ├─ Bolie.py                      ├─ options_Bolie.py              │
│  └─ calibration_single.py         └─ options_default.py            │
│       │                                    │                        │
│       │ Defines:                           │ Defines:               │
│       │ • States (x)                       │ • modelname            │
│       │ • Parameters (p)                   │ • checkObser           │
│       │ • Outputs (h)                      │ • maxLietime           │
│       │ • Dynamics (f)                     │ • nnzDerU/nnzDerW      │
│       │ • Inputs (u, w)                    │ • MANIFOLD_PLOTTING    │
│       │                                    │                        │
└───────┼────────────────────────────────────┼─────────────────────────┘
        │                                    │
        └────────────────┬───────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CORE: StrikePy Engine                          │
│                  src/nullstrike/core/                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  strike_goldd.py - Main STRIKE-GOLDD Algorithm                     │
│  ────────────────────────────────────────────────────────           │
│                                                                     │
│  INPUT: Model + Options                                            │
│    ↓                                                                │
│  1. Extract symbols (states, parameters, inputs)                   │
│    ↓                                                                │
│  2. Compute Lie Derivatives iteratively:                           │
│     • L⁰_f(h) = h                                                  │
│     • L¹_f(h) = ∂h/∂x · f                                          │
│     • L²_f(h) = ∂(L_f h)/∂x · f                                    │
│     • ... continue until rank stops increasing                     │
│    ↓                                                                │
│  3. Build Observability-Identifiability Matrix:                    │
│     ┌                    ┐                                         │
│     │  L⁰_f(h)          │                                          │
│     │  L¹_f(h)          │                                          │
│     │  L²_f(h)          │  ← Rows = Lie derivatives                │
│     │    ...            │                                          │
│     │  Lⁿ_f(h)          │                                          │
│     └                    ┘                                         │
│          Columns = [states, parameters, unknown_inputs]            │
│    ↓                                                                │
│  4. Compute Jacobian: ∂(Lie derivatives)/∂(states, params)         │
│    ↓                                                                │
│  5. Identify individually identifiable parameters                  │
│    ↓                                                                │
│  OUTPUT: Observability-Identifiability Matrix (O)                  │
│          Saved to: results/obs_ident_matrix_{model}_{timestamp}.txt│
│                                                                     │
│  functions/                                                         │
│  ├─ elim_and_recalc.py - Eliminate identified vars & recalculate   │
│  └─ rationalize.py - Simplify symbolic expressions                 │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  ANALYSIS: Nullspace Analysis                       │
│                  src/nullstrike/analysis/                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  enhanced_subspace.py - Core Mathematical Analysis                 │
│  ──────────────────────────────────────────────────────             │
│                                                                     │
│  analyze_identifiable_combinations()                               │
│  ────────────────────────────────────                               │
│                                                                     │
│  INPUT: O matrix from StrikePy                                     │
│    ↓                                                                │
│  1. Compute nullspace of O:                                        │
│     N = {v : O·v = 0}  ← Unidentifiable directions                │
│    ↓                                                                │
│  2. Build nullspace matrix (rows = nullspace vectors):             │
│     ┌              ┐                                               │
│     │  n₁ᵀ        │ ← Each row is a nullspace vector              │
│     │  n₂ᵀ        │                                                │
│     │  ...        │                                                │
│     │  nₖᵀ        │                                                │
│     └              ┘                                               │
│    ↓                                                                │
│  3. Compute nullspace of N:                                        │
│     I = nullspace(N) ← Identifiable combinations                   │
│    ↓                                                                │
│  4. Interpret results:                                             │
│     • If nullspace_dim = 0 → Fully identifiable                   │
│     • If nullspace_dim = k → (total_vars - k) combinations         │
│                              are identifiable                      │
│    ↓                                                                │
│  5. Extract parameter relationships:                               │
│     • Parse nullspace vectors                                      │
│     • Find patterns (sums, products, ratios)                       │
│     • Generate human-readable interpretations                      │
│    ↓                                                                │
│  OUTPUT: {                                                          │
│    'nullspace_dimension': k,                                       │
│    'fully_identifiable': bool,                                     │
│    'unidentifiable_patterns': [...],                               │
│    'identifiable_info': {...}                                      │
│  }                                                                  │
│                                                                     │
│  ────────────────────────────────────────────────────────           │
│                                                                     │
│  integrated_analysis.py - Orchestration                            │
│  ├─ Runs StrikePy                                                  │
│  ├─ Checks checkpoints                                             │
│  ├─ Calls enhanced_subspace                                        │
│  └─ Triggers visualizations                                        │
│                                                                     │
│  checkpointing.py - Caching System                                 │
│  ├─ compute_model_hash() - Hash model + options                   │
│  ├─ save_checkpoint() - Cache results                              │
│  └─ load_checkpoint() - Retrieve cached results                    │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  VISUALIZATION: Output Generation                   │
│                  src/nullstrike/visualization/                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  graphs.py - Network Graphs                                        │
│  ───────────────────────────                                        │
│  ├─ build_identifiability_graph() - Create NetworkX graph          │
│  │   • Nodes = parameters/states                                   │
│  │   • Edges = relationships from nullspace                        │
│  │                                                                  │
│  └─ visualize_identifiability_graph() - Plot graph                 │
│      • Full graph (all variables)                                  │
│      • Parameters-only graph                                       │
│                                                                     │
│  manifolds.py - 3D/2D Manifolds                                    │
│  ───────────────────────────────                                    │
│  └─ visualize_nullspace_manifolds() - Plot constraint surfaces     │
│      • 3D surface plots (parameter combinations)                   │
│      • 2D projections (pairwise relationships)                     │
│                                                                     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       OUTPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  results/{model_name}/{timestamp}/                                 │
│  ├─ detailed_analysis.txt - Text report                            │
│  ├─ graphs/                                                         │
│  │  ├─ identifiability_graph_full.png                              │
│  │  └─ identifiability_graph_params.png                            │
│  ├─ manifolds_2d/                                                   │
│  │  └─ param_pair_*.png                                            │
│  └─ manifolds_3d/                                                   │
│     └─ param_triple_*.png                                           │
│                                                                     │
│  checkpoints/{model_name}/                                         │
│  └─ checkpoint_{hash}.pkl - Cached analysis                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Diagram

```
┌─────────────┐
│    USER     │
│ runs CLI    │
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  cli/complete_analysis.py                   │
│  • Parse arguments                           │
│  • Determine model/options                   │
└──────┬───────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  Import model & options                      │
│  • custom_models/{model}.py                  │
│  • custom_options/{options}.py               │
└──────┬───────────────────────────────────────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌─────────────┐      ┌────────────────┐
│ Checkpoint? │──Yes─>│ Load cached    │
│             │      │ Skip StrikePy  │
└─────┬───────┘      └────────┬───────┘
      │ No                    │
      ▼                       │
┌─────────────────────────────┼───────┐
│  core/strike_goldd.py       │       │
│  • Compute Lie derivatives  │       │
│  • Build O matrix           │       │
│  • Identify individual vars │       │
└─────┬───────────────────────┼───────┘
      │                       │
      └───────────┬───────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  analysis/enhanced_subspace.py             │
│  • Compute nullspace(O)                     │
│  • Compute nullspace(N) = identifiable      │
│  • Parse relationships                      │
│  • Generate interpretations                 │
└─────┬───────────────────────────────────────┘
      │
      ├──────────────┬──────────────┐
      │              │              │
      ▼              ▼              ▼
┌─────────┐  ┌──────────┐  ┌───────────────┐
│ Save    │  │ Generate │  │ Generate      │
│ Report  │  │ Graphs   │  │ Manifolds     │
└─────────┘  └──────────┘  └───────────────┘
      │              │              │
      └──────────────┼──────────────┘
                     │
                     ▼
            ┌────────────────┐
            │ results/ dir   │
            │ User reviews   │
            └────────────────┘
```

## Key Mathematical Flow

```
Model Definition (custom_models/)
         ↓
    States: x = [x₁, x₂, ..., xₙ]
    Parameters: p = [p₁, p₂, ..., pₘ]
    Outputs: h = [h₁, h₂, ..., hₖ]
    Dynamics: dx/dt = f(x, p, u)
         ↓
─────────────────────────────────────────────
StrikePy: Compute Lie Derivatives
─────────────────────────────────────────────
         ↓
    L⁰_f(h) = h
    L¹_f(h) = ∂h/∂x · f
    L²_f(h) = ∂(L_f h)/∂x · f
    ...
    Lⁿ_f(h) = continues until rank saturates
         ↓
─────────────────────────────────────────────
Build Observability-Identifiability Matrix
─────────────────────────────────────────────
         ↓
         ┌                    ┐
         │  L⁰_f(h)          │
    O =  │  L¹_f(h)          │
         │     ...           │
         │  Lⁿ_f(h)          │
         └                    ┘

    Jacobian: J = ∂O/∂[x, p, w]
         ↓
─────────────────────────────────────────────
Nullspace Analysis (NullStrike Enhancement)
─────────────────────────────────────────────
         ↓
    Nullspace: N = {v : O·v = 0}

    If N = {0} → Fully identifiable ✓
    If dim(N) = k → k combinations unidentifiable
         ↓
    Nullspace Matrix:
         ┌          ┐
    N =  │  n₁ᵀ    │  ← Each row = unidentifiable direction
         │  n₂ᵀ    │
         │  ...    │
         └          ┘
         ↓
    Identifiable Combinations:
    I = nullspace(N) = row space of O

    dim(I) = total_variables - dim(N)
         ↓
─────────────────────────────────────────────
Interpretation & Visualization
─────────────────────────────────────────────
         ↓
    Parse nullspace vectors:
    • n₁ = [0, 0, 1, -1, 0] → p₃ - p₄ = 0 (unidentifiable)
    • Means: p₃ + p₄ IS identifiable ✓
         ↓
    Generate outputs:
    • Text reports
    • Network graphs
    • 3D manifolds
    • 2D projections
```

## Module Dependencies

```
nullstrike/
│
├── cli/
│   └── complete_analysis.py
│       └── imports: core.strike_goldd
│                   analysis.integrated_analysis
│                   analysis.enhanced_subspace
│
├── core/
│   ├── strike_goldd.py
│   │   └── imports: functions.rationalize
│   │               functions.elim_and_recalc
│   │               configs.default_options
│   └── functions/
│       ├── rationalize.py (no deps)
│       └── elim_and_recalc.py (no deps)
│
├── analysis/
│   ├── integrated_analysis.py
│   │   └── imports: core.strike_goldd
│   │               analysis.enhanced_subspace
│   │               analysis.checkpointing
│   │               visualization.graphs
│   │               visualization.manifolds
│   │
│   ├── enhanced_subspace.py
│   │   └── imports: visualization.graphs (conditional)
│   │               visualization.manifolds (conditional)
│   │
│   └── checkpointing.py (no deps)
│
├── visualization/
│   ├── graphs.py (no internal deps)
│   └── manifolds.py (no internal deps)
│
├── configs/
│   └── default_options.py (no deps)
│
└── utils.py (no deps)
```

## Execution Paths

### Path 1: Successful Integrated Analysis
```
CLI → run_integrated_analysis()
   → strike_goldd() [computes O matrix]
   → load_checkpoint() [cache miss]
   → analyze_identifiable_combinations() [nullspace analysis]
   → save_checkpoint() [cache for next time]
   → visualize_graphs() [network plots]
   → visualize_manifolds() [3D/2D surfaces]
   → save_detailed_results() [text report]
   → display_final_summary()
```

### Path 2: Checkpoint Hit (Fast Path)
```
CLI → run_integrated_analysis()
   → strike_goldd() [computes O matrix]
   → load_checkpoint() [cache HIT! ✓]
   → visualize_graphs() [from cached results]
   → visualize_manifolds() [from cached results]
   → display_final_summary()
```

### Path 3: Fallback to Step-by-Step
```
CLI → run_integrated_analysis() [FAILS]
   → run_step_by_step_analysis()
   → strike_goldd() [separate call]
   → analyze_strikepy_results() [loads from file]
   → display_final_summary()
```

### Path 4: Analyze Existing Results Only
```
CLI → run_integrated_analysis() [FAILS]
   → run_step_by_step_analysis() [FAILS]
   → analyze_strikepy_results()
      → load_oic_matrix_from_file()
      → analyze_identifiable_combinations()
      → display_final_summary()
```

## Performance Optimizations

```
┌─────────────────────────────────────────┐
│  Checkpointing System                   │
├─────────────────────────────────────────┤
│                                         │
│  Hash = hash(model + options)          │
│      ↓                                  │
│  First run:                             │
│  • Compute O matrix (slow, symbolic)   │
│  • Perform nullspace analysis          │
│  • Save to checkpoints/{hash}.pkl      │
│      ↓                                  │
│  Subsequent runs:                       │
│  • Check if checkpoint exists           │
│  • Load cached results (fast!)         │
│  • Skip expensive computation          │
│                                         │
│  Speed improvement: 10-100x faster     │
│                                         │
└─────────────────────────────────────────┘
```

---

**This architecture emphasizes:**
1. **Modularity**: Each component has a single responsibility
2. **Robustness**: 3-method fallback strategy ensures analysis completes
3. **Performance**: Intelligent checkpointing avoids recomputation
4. **Extensibility**: Easy to add new visualization or analysis methods

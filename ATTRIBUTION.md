# Attribution and Credits

## NullStrike: Enhanced Structural Identifiability Analysis

NullStrike builds upon and extends the StrikePy package, which is a Python implementation of the STRIKE-GOLDD algorithm.

### Original Work Credits

**StrikePy** (Python Implementation)
- **Author**: David Rey Rostro (davidreyrostro@gmail.com)  
- **License**: GPL-3.0
- **Description**: Python implementation of STRIKE-GOLDD for structural identifiability analysis
- **Original Repository**: [Link to original StrikePy if available]

**STRIKE-GOLDD** (Original Algorithm)
- **Author**: Alejandro Fernandez Villaverde (afvillaverde@uvigo.gal)
- **Institution**: University of Vigo, Spain
- **Description**: Original MATLAB toolbox for structural identifiability analysis using Lie derivatives
- **Reference**: Villaverde, A.F., Barreiro, A., & Papachristodoulou, A. (2016). Structural identifiability of dynamic systems biology models. PLoS computational biology, 12(10), e1005153.

### NullStrike Extensions

**Enhanced Nullspace Analysis** 
- **Authors**: [Your Name(s)]
- **License**: GPL-3.0 (to comply with StrikePy licensing)
- **Description**: Advanced nullspace analysis to identify parameter combinations that are structurally identifiable

**Key Enhancements**:
- Nullspace analysis of observability-identifiability matrices
- Identification of structurally identifiable parameter combinations
- Advanced visualization tools (3D manifold plots, constraint graphs)
- Comprehensive reporting with mathematical interpretations
- Checkpointing system for efficient reanalysis
- Enhanced command-line interface

### Directory Structure Attribution

```
src/nullstrike/core/          # Contains original StrikePy code (GPL-3.0)
src/nullstrike/analysis/      # NullStrike extensions (GPL-3.0)
src/nullstrike/visualization/ # NullStrike extensions (GPL-3.0)
```

### Usage Citation

If you use NullStrike in your research, please cite both the original works and this extension:

```bibtex
@software{nullstrike2024,
  title={NullStrike: Enhanced Structural Identifiability Analysis with Nullspace Parameter Combinations},
  author={[Your Name]},
  year={2024},
  url={https://github.com/yourusername/nullstrike},
  note={Built upon StrikePy by David Rey Rostro and STRIKE-GOLDD by Alejandro Fernandez Villaverde}
}

@article{villaverde2016structural,
  title={Structural identifiability of dynamic systems biology models},
  author={Villaverde, Alejandro F and Barreiro, Antonio and Papachristodoulou, Antonis},
  journal={PLoS computational biology},
  volume={12},
  number={10},
  pages={e1005153},
  year={2016},
  publisher={Public Library of Science}
}
```

### License Compliance

This project is licensed under GPL-3.0 to maintain compatibility with the original StrikePy implementation. All modifications and extensions are also released under GPL-3.0, ensuring the continued open-source nature of this work.
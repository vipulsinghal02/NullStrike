import pickle
import hashlib
import json
from pathlib import Path
import sympy as sym
from sympy import Matrix

# Import utilities
from ..utils import get_checkpoints_dir

def compute_model_hash(model, options):
    """Compute hash of model and options for checkpoint validation."""
    # Get model content
    model_content = {
        'x': str(model.x),
        'p': str(model.p), 
        'h': str(model.h),
        'f': str(model.f),
        'u': str(getattr(model, 'u', [])),
        'w': str(getattr(model, 'w', []))
    }
    
    # Get relevant options (exclude plotting config)
    options_content = {
        'modelname': options.modelname,
        'checkObser': options.checkObser,
        'maxLietime': options.maxLietime,
        'nnzDerU': options.nnzDerU,
        'nnzDerW': options.nnzDerW,
        'prev_ident_pars': str(options.prev_ident_pars)
    }
    
    combined = json.dumps([model_content, options_content], sort_keys=True)
    return hashlib.md5(combined.encode()).hexdigest()

def save_checkpoint(model_name, options_file, model_hash, oic_matrix, nullspace_results, identifiable_results):
    """Save checkpoint data."""
    checkpoint_dir = get_checkpoints_dir()
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_name = f"{model_name}_{options_file or 'default'}.pkl"
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    checkpoint_data = {
        'model_hash': model_hash,
        'oic_matrix': oic_matrix,
        'nullspace_results': nullspace_results,
        'identifiable_results': identifiable_results,
        'model_name': model_name,
        'options_file': options_file
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"Checkpoint saved: {checkpoint_path}")

def load_checkpoint(model_name, options_file, expected_hash):
    """Load checkpoint if it exists and hash matches."""
    checkpoint_dir = get_checkpoints_dir()
    checkpoint_name = f"{model_name}_{options_file or 'default'}.pkl"
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    if not checkpoint_path.exists():
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        if checkpoint_data['model_hash'] == expected_hash:
            print(f"✓ Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
        else:
            print(f"⚠ Checkpoint hash mismatch - model/options changed")
            return None
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
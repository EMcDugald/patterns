from pathlib import Path
import numpy as np

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_npz(path, **arrays):
    path = Path(path)
    np.savez_compressed(path, **arrays)
    return path

def save_npy(path, obj):
    path = Path(path)
    np.save(path, obj)
    return path

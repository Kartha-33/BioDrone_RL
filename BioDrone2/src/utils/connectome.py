import torch
import numpy as np
import os

def load_adjacency_matrix(path):
    """
    Placeholder: Load real connectome data (e.g., from generated CSV).
    For now, returns a random sparse mask.
    """
    if not os.path.exists(path):
        return None
    return np.loadtxt(path, delimiter=',')

def generate_bio_mask(in_features, out_features, connection_type='random', density=0.2):
    """
    Generates a binary mask based on biological principles.
    
    Types:
    - 'random': Erdős–Rényi graph (Standard sparse).
    - 'local': Neurons only connect to neighbors (Retinotopic).
    - 'small_world': Watts-Strogatz (Local + few long-range).
    """
    mask = torch.zeros(out_features, in_features)
    
    if connection_type == 'random':
        # Pure random sparsity
        mask = (torch.rand(out_features, in_features) < density).float()
        
    elif connection_type == 'local':
        # Retinotopy: Input i connects to Hidden i-k...i+k
        # This mimics the optic lobe (Lamina -> Medulla) where spatial layout is preserved.
        
        # Scale indices to match dimensions
        scale = in_features / out_features
        window_size = max(1, int(in_features * density))
        
        for i in range(out_features):
            center = int(i * scale)
            start = max(0, center - window_size // 2)
            end = min(in_features, center + window_size // 2)
            mask[i, start:end] = 1.0
            
    elif connection_type == 'small_world':
        # 1. Start with Local Structure (Retinotopy)
        # Recursive call to get the base local mask
        mask = generate_bio_mask(in_features, out_features, 'local', density)
        
        # 2. Add Long-Range "Shortcuts" (Rewiring)
        # We want to add random connections without increasing density too much.
        # Let's add 5% randomness.
        
        # Creates a random mask of same size
        random_shortcuts = (torch.rand(out_features, in_features) < 0.05).float()
        
        # Combine: Local OR Random -> Clamped to 1.0
        mask = torch.clamp(mask + random_shortcuts, 0.0, 1.0)
            
    return mask

class ConnectomeRegistry:
    """
    Manages the masks for different layers of the policy.
    """
    def __init__(self):
        self.masks = {}
    
    def register(self, layer_name, mask):
        self.masks[layer_name] = mask
        
    def get(self, layer_name):
        return self.masks.get(layer_name)
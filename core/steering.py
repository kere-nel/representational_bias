"""Steering vector generation and application functionality."""

import torch
import os
from .models import get_mean_representation


def _get_steering_vector_cache_path(expert_file, novice_file, layer, max_samples, model_name):
    """Generate cache file path for steering vector.
    
    Args:
        expert_file: Path to expert examples file
        novice_file: Path to novice examples file
        layer: Layer to extract representations from
        max_samples: Maximum number of samples used
        model_name: Name of the model
        
    Returns:
        Path to cache file
    """
    # Create a unique identifier based on all parameters
    expert_basename = os.path.splitext(os.path.basename(expert_file))[0]
    novice_basename = os.path.splitext(os.path.basename(novice_file))[0]
    
    # Extract steering vector type (e.g., "expertise" from "expertise_expert" and "expertise_novice")
    expert_parts = expert_basename.split('_')
    novice_parts = novice_basename.split('_')
    
    steering_type = "unknown"
    if len(expert_parts) > 1 and len(novice_parts) > 1:
        if expert_parts[0] == novice_parts[0]:
            steering_type = expert_parts[0]
    
    # Clean model name for filename
    model_clean = model_name.replace('/', '_').replace(':', '_')
    
    # Create cache filename
    cache_filename = f"{steering_type}_layer{layer}_samples{max_samples}_{model_clean}.pt"
    
    # Get the base directory (where the script is run from)
    base_dir = os.getcwd()
    cache_dir = os.path.join(base_dir, "data", "steering_vectors")
    
    return os.path.join(cache_dir, cache_filename)


def create_steering_vector(model, tokenizer, expert_file, novice_file, layer, max_samples=10, force_recompute=False):
    """Create steering vector from expert vs novice text files with caching.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        expert_file: Path to file containing expert example texts
        novice_file: Path to file containing novice example texts  
        layer: Layer to extract representations from
        max_samples: Maximum number of samples to use from each file (default: 10)
        force_recompute: If True, recompute even if cached version exists
        
    Returns:
        Steering vector tensor
    """
    # Get model name for cache path
    model_name = getattr(model, 'name_or_path', 'unknown_model')
    if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
        model_name = model.config._name_or_path
    
    # Check cache first
    cache_path = _get_steering_vector_cache_path(expert_file, novice_file, layer, max_samples, model_name)
    
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached steering vector from: {cache_path}")
        try:
            steering_vector = torch.load(cache_path, map_location=model.device)
            return steering_vector
        except Exception as e:
            print(f"Failed to load cached steering vector: {e}")
            print("Recomputing steering vector...")
    
    print(f"Computing steering vector and saving to: {cache_path}")
    
    # Load expert examples
    with open(expert_file, 'r') as f:
        expert_texts = [line.strip() for line in f if line.strip()][:max_samples]
    
    # Load novice examples  
    with open(novice_file, 'r') as f:
        novice_texts = [line.strip() for line in f if line.strip()][:max_samples]
    
    novice_vec = get_mean_representation(model, tokenizer, novice_texts, layer=layer)
    expert_vec = get_mean_representation(model, tokenizer, expert_texts, layer=layer)

    print(f"DEBUG: novice_vec shape: {novice_vec.shape}")
    print(f"DEBUG: expert_vec shape: {expert_vec.shape}")

    steering_vector = expert_vec.mean(dim=0) - novice_vec.mean(dim=0)

    print(f"DEBUG: steering_vector shape: {steering_vector.shape}")
    print(f"DEBUG: Expected hidden dimension: {model.config.hidden_size if hasattr(model.config, 'hidden_size') else 'unknown'}")
    
    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        torch.save(steering_vector, cache_path)
        print(f"Saved steering vector to cache: {cache_path}")
    except Exception as e:
        print(f"Failed to save steering vector to cache: {e}")
    
    return steering_vector


def apply_steering(residual, steering_vector, alpha):
    """Apply steering vector to residual stream.
    
    Args:
        residual: Residual stream tensor
        steering_vector: Vector to steer with
        alpha: Steering strength
        
    Returns:
        Modified residual stream
    """
    return residual + alpha * steering_vector
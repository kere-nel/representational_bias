"""Steering vector generation and application functionality."""

import torch
from .models import get_mean_representation


def create_steering_vector(model, tokenizer, expert_file, novice_file, layer, max_samples=10):
    """Create steering vector from expert vs novice text files.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        expert_file: Path to file containing expert example texts
        novice_file: Path to file containing novice example texts  
        layer: Layer to extract representations from
        max_samples: Maximum number of samples to use from each file (default: 10)
        
    Returns:
        Steering vector tensor
    """
    # Load expert examples
    with open(expert_file, 'r') as f:
        expert_texts = [line.strip() for line in f if line.strip()][:max_samples]
    
    # Load novice examples  
    with open(novice_file, 'r') as f:
        novice_texts = [line.strip() for line in f if line.strip()][:max_samples]
    
    novice_vec = get_mean_representation(model, tokenizer, novice_texts, layer=layer)
    expert_vec = get_mean_representation(model, tokenizer, expert_texts, layer=layer)
    expert_vector = expert_vec.mean(dim=0) - novice_vec.mean(dim=0)
    return expert_vector


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
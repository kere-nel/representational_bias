"""Utility functions for file I/O, caching, and environment setup."""

import os
import pickle
import json
import time
from typing import Any, Dict


def setup_environment(cache_dir="/projectnb/buinlp/kerenf"):
    """Set up environment variables for HuggingFace caching.
    
    Args:
        cache_dir: Directory for caching models and data
    """
    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_HUB_CACHE"] = cache_dir


def save_results(data, filepath, create_dirs=True):
    """Save results to pickle file.
    
    Args:
        data: Data to save
        filepath: Path to save file
        create_dirs: Whether to create directories if they don't exist
    """
    if create_dirs:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_results(filepath):
    """Load results from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded data
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_json(data, filepath, create_dirs=True):
    """Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save file  
        create_dirs: Whether to create directories if they don't exist
    """
    if create_dirs:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, "r") as f:
        return json.load(f)


def get_steering_vector_name(expert_file, novice_file):
    """Extract a descriptive name from steering vector files.
    
    Args:
        expert_file: Path to expert examples file
        novice_file: Path to novice examples file
        
    Returns:
        String representing the steering vector name
    """
    expert_basename = os.path.splitext(os.path.basename(expert_file))[0]
    novice_basename = os.path.splitext(os.path.basename(novice_file))[0]
    
    # Try to extract common prefix (e.g., "expertise" from "expertise_expert" and "expertise_novice")
    expert_parts = expert_basename.split('_')
    novice_parts = novice_basename.split('_')
    
    # Find common prefix
    common_prefix = None
    if len(expert_parts) > 1 and len(novice_parts) > 1:
        if expert_parts[0] == novice_parts[0]:
            common_prefix = expert_parts[0]
    
    # Return common prefix if found, otherwise use expert basename
    return common_prefix if common_prefix else expert_basename


def create_output_filepath(model_name, occupation, args, suffix="", expert_file=None, novice_file=None):
    """Create standardized output filepath.
    
    Args:
        model_name: Name of the model
        occupation: Occupation being analyzed
        args: Argument namespace with experiment parameters
        suffix: Optional suffix for filename
        expert_file: Optional path to expert steering vector file
        novice_file: Optional path to novice steering vector file
        
    Returns:
        Complete filepath for output
    """
    model_clean = model_name.replace('/', '_')
    
    # Include steering vector name in path if provided
    if expert_file and novice_file:
        steering_name = get_steering_vector_name(expert_file, novice_file)
        folder_path = f"/project/buinlp/data/outputs/{model_clean}/{steering_name}/temp{args.temperature}/topp{args.top_p}"
    else:
        folder_path = f"/project/buinlp/data/outputs/{model_clean}/temp{args.temperature}/topp{args.top_p}"
    
    debug_str = '_DEBUG' if args.debug else ''
    
    filename = f"{occupation}{debug_str}_{args.part}{suffix}.pkl"
    return os.path.join(folder_path, filename)


def time_execution(func, *args, **kwargs):
    """Time function execution.
    
    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, execution_time)
    """
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    
    return result, end - start


def print_execution_time(start_time, end_time=None):
    """Print execution time.
    
    Args:
        start_time: Start time
        end_time: End time (current time if None)
    """
    if end_time is None:
        end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")
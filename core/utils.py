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


def create_output_filepath(model_name, occupation, args, suffix=""):
    """Create standardized output filepath.
    
    Args:
        model_name: Name of the model
        occupation: Occupation being analyzed
        args: Argument namespace with experiment parameters
        suffix: Optional suffix for filename
        
    Returns:
        Complete filepath for output
    """
    model_clean = model_name.replace('/', '_')
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
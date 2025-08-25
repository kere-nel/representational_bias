"""Core module for SAE gender bias analysis.

This module provides the core functionality for analyzing gender bias 
in large language models using Sparse Autoencoders and steering vectors.
"""

from .models import load_model, setup_model_hooks
from .data import load_questions, load_demographics, get_demographic_prefixes, create_name_based_prefixes, partition_questions, get_random_profession_prefixes
from .experiments import run_experiment, sample_model, run_name_based_experiment, run_baseline_experiment
from .steering import create_steering_vector, apply_steering
from .utils import save_results, load_results, setup_environment, create_output_filepath
from .analysis import (
    remove_context_post, 
    readability_score
)

__all__ = [
    'load_model',
    'setup_model_hooks', 
    'load_questions',
    'load_demographics',
    'get_demographic_prefixes',
    'create_name_based_prefixes',
    'partition_questions',
    'get_random_profession_prefixes',
    'run_experiment',
    'sample_model',
    'run_name_based_experiment',
    'run_baseline_experiment',
    'create_steering_vector',
    'apply_steering',
    'save_results',
    'load_results',
    'setup_environment',
    'create_output_filepath',
    'remove_context_post',
    'readability_score'
]
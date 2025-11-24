#!/usr/bin/env python3
"""
Generate steering vectors for all contrastive pairs across all models.

This script creates steering vectors for all available expert/novice pairs
in the data/contrastive_pairs directory for each model tested in the hiring
bias experiments.

Output naming format: {vector_type}_layer{layer}_samples{n}_{model_name}.pt
"""

import os
import sys
import torch as t
from nnsight import LanguageModel
from huggingface_hub import login
import argparse
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.abspath('.')
sys.path.append(parent_dir)

from core.steering import create_steering_vector

# Configuration
CONTRASTIVE_PAIRS_DIR = "data/contrastive_pairs"
STEERING_VECTORS_DIR = "data/steering_vectors"
SAVE_DIR = 'cache/models'

# Model configurations: model_name -> (layer, samples)
MODEL_CONFIGS = {
    'google/gemma-2-2b-it': {'layer': 10, 'samples': 50},
    'google/gemma-2-9b-it': {'layer': 20, 'samples': 50},
    'meta-llama/Llama-3.1-8B-Instruct': {'layer': 13, 'samples': 50},
}

# Vectors to skip (already created as 'expertise' and 'it')
SKIP_VECTORS = {'expertise', 'it_resume'}

# Special mappings for vector names
VECTOR_NAME_MAPPING = {
    'it_resume': 'it',
    'expertise_expert_corrected': 'expertise',
}


def find_contrastive_pairs(root_dir):
    """
    Find all expert/novice pairs in the contrastive pairs directory.

    Returns:
        dict: axis_name -> (expert_path, novice_path)
    """
    files = {f.lower(): os.path.join(root_dir, f) for f in os.listdir(root_dir)}
    pairs = {}

    # Auto-discover *_expert.txt vs *_novice.txt pairs
    expert_files = [f for f in files if f.endswith("_expert.txt")]
    for ef in expert_files:
        axis = ef[:-len("_expert.txt")]
        nf = axis + "_novice.txt"
        if nf in files:
            pairs[axis] = (files[ef], files[nf])

    return pairs


def get_model_name_for_filename(model_name):
    """Convert model path to filename-safe format."""
    return model_name.replace('/', '_')


def build_vector(model, expert_path, novice_path, layer, n_samples, seed=1234):
    """
    Build a steering vector with specified parameters.

    Args:
        model: LanguageModel instance
        expert_path: Path to expert examples
        novice_path: Path to novice examples
        layer: Layer to extract representations from
        n_samples: Number of samples to use
        seed: Random seed for reproducibility

    Returns:
        torch.Tensor: Steering vector
    """
    if seed is not None:
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)

    vec = create_steering_vector(
        model, model.tokenizer, expert_path, novice_path, layer, n_samples
    ).detach().cpu().to(dtype=t.float32)

    return vec.view(-1)


def main():
    parser = argparse.ArgumentParser(description='Generate steering vectors for all models and contrastive pairs')
    parser.add_argument('--model', type=str, default=None,
                        help='Specific model to process (if not specified, processes all)')
    parser.add_argument('--vector', type=str, default=None,
                        help='Specific vector type to process (if not specified, processes all)')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace access token')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, cpu)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        help='Data type (bfloat16, float16, float32)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Override number of samples (uses model config if not specified)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be generated without actually generating')

    args = parser.parse_args()

    # Login to HuggingFace
    if args.hf_token:
        login(args.hf_token)

    # Parse dtype
    dtype_map = {
        'bfloat16': t.bfloat16,
        'float16': t.float16,
        'float32': t.float32,
    }
    dtype = dtype_map.get(args.dtype, t.bfloat16)

    # Find all contrastive pairs
    pairs = find_contrastive_pairs(CONTRASTIVE_PAIRS_DIR)
    print(f"Found {len(pairs)} contrastive pairs:")
    for axis in sorted(pairs.keys()):
        print(f"  - {axis}")
    print()

    # Create output directory if it doesn't exist
    os.makedirs(STEERING_VECTORS_DIR, exist_ok=True)

    # Filter models if specified
    models_to_process = {args.model: MODEL_CONFIGS[args.model]} if args.model else MODEL_CONFIGS

    # Filter vectors if specified
    if args.vector:
        if args.vector not in pairs:
            print(f"Error: Vector '{args.vector}' not found in contrastive pairs")
            return
        pairs = {args.vector: pairs[args.vector]}

    # Process each model
    for model_name, config in models_to_process.items():
        layer = config['layer']
        n_samples = args.samples if args.samples else config['samples']

        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"Layer: {layer}, Samples: {n_samples}")
        print(f"{'='*80}\n")

        # Load model
        if not args.dry_run:
            print(f"Loading model {model_name}...")
            model = LanguageModel(
                model_name,
                device_map=args.device,
                cache_dir=SAVE_DIR,
                dispatch=True,
                torch_dtype=dtype
            )
            print("Model loaded successfully!\n")

        # Process each contrastive pair
        for axis, (expert_path, novice_path) in sorted(pairs.items()):
            # Skip if in skip list
            if axis in SKIP_VECTORS:
                print(f"Skipping {axis} (already exists)")
                continue

            # Get vector name (handle special mappings)
            vector_name = VECTOR_NAME_MAPPING.get(axis, axis)

            # Construct output filename
            model_name_safe = get_model_name_for_filename(model_name)
            output_filename = f"{vector_name}_layer{layer}_samples{n_samples}_{model_name_safe}.pt"
            output_path = os.path.join(STEERING_VECTORS_DIR, output_filename)

            # Check if already exists
            if os.path.exists(output_path):
                print(f"✓ {output_filename} (already exists)")
                continue

            if args.dry_run:
                print(f"Would create: {output_filename}")
                continue

            try:
                print(f"Creating {vector_name} vector...")
                print(f"  Expert:  {os.path.basename(expert_path)}")
                print(f"  Novice:  {os.path.basename(novice_path)}")

                # Build vector
                vector = build_vector(model, expert_path, novice_path, layer, n_samples)

                # Save vector
                t.save(vector, output_path)
                print(f"✓ Saved: {output_filename}")
                print(f"  Shape: {vector.shape}, Norm: {vector.norm():.4f}\n")

            except Exception as e:
                print(f"✗ Error creating {vector_name}: {e}\n")
                continue

        # Clean up model
        if not args.dry_run:
            del model
            t.cuda.empty_cache()
            print(f"\nFinished processing {model_name}\n")

    print("\n" + "="*80)
    print("All done!")
    print("="*80)


if __name__ == "__main__":
    main()

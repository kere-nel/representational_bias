"""Main script for random profession baseline experiments with Hydra configuration."""

import time
import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Disable torch compilation to avoid nnsight compatibility issues
torch._dynamo.config.disable = True

from core import (
    setup_environment,
    load_model,
    load_questions,
    run_baseline_experiment,
    create_steering_vector,
    save_results,
    partition_questions,
    get_random_profession_prefixes,
    get_steering_vector_name
)
from core.models import get_model_config


@hydra.main(version_base=None, config_path="config", config_name="baseline_config")
def main(cfg: DictConfig) -> None:
    """Main baseline experiment runner.
    
    Args:
        cfg: Hydra configuration object
    """
    print("Baseline Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup environment
    setup_environment()
    
    # Load model
    print(f"Loading model: {cfg.model.name}")
    model, tokenizer = load_model(cfg.model.name, debug=cfg.experiment.debug)
    model_config = get_model_config(cfg.model.name)
    
    # Override model config with Hydra config if specified
    if hasattr(cfg.model, 'steering_layer'):
        model_config.steering_layer = cfg.model.steering_layer
        print(f"Overriding steering_layer: {cfg.model.steering_layer}")
    
    # Load questions for single occupation
    occupation = cfg.experiment.occupation
    if not occupation:
        raise ValueError("Must specify a single occupation in config.experiment.occupation")
    
    questions = load_questions(cfg.paths.questions_dir, [occupation])
    
    # Partition questions if needed
    if int(cfg.experiment.num_parts) > 1:
        questions[occupation] = partition_questions(
            questions[occupation], 
            int(cfg.experiment.part), 
            int(cfg.experiment.num_parts)
        )
    
    # Create steering vector
    print("Creating steering vector...")
    steering_vector = create_steering_vector(
        model=model,
        tokenizer=tokenizer,
        expert_file=cfg.paths.expert_file,
        novice_file=cfg.paths.novice_file,
        layer=model_config.steering_layer,
        max_samples=int(cfg.experiment.max_steering_samples)
    )
    
    start_time = time.time()
    
    # Run random profession baseline experiments
    # Get alpha values from config
    alpha_values = cfg.experiment.alpha_values
    
    print(f"Running baseline experiments for: {occupation}")
    
    # Get random profession prefixes for this occupation
    prefixes = get_random_profession_prefixes(
        target_occupation=occupation,
        num_random_professions=cfg.experiment.num_random_professions
    )
    
    # Run experiments for each alpha value
    for alpha in alpha_values:
        print(f"Running with alpha={alpha}")
        
        # Create output file path - include occupation and steering vector name in the path
        steering_name = get_steering_vector_name(cfg.paths.expert_file, cfg.paths.novice_file)
        alpha_str = f"alpha{alpha}"
        model_clean = cfg.model.name.replace('/', '_')
        folder_path = f"{cfg.paths.output_dir}/baselines/{occupation}/{model_clean}/{steering_name}/{alpha_str}/temp{cfg.generation.temperature}/topp{cfg.generation.top_p}"
        os.makedirs(folder_path, exist_ok=True)
        debug_str = '_DEBUG' if cfg.experiment.debug else ''
        output_file = f"{folder_path}/{occupation}_baseline{debug_str}_{cfg.experiment.part}.pkl"
        
        outputs = run_baseline_experiment(
            model=model,
            tokenizer=tokenizer,
            questions=questions[occupation],
            occupation=occupation,
            steering_vector=steering_vector,
            prefixes=prefixes,
            model_name=cfg.model.name,
            alpha=alpha,
            debug=cfg.experiment.debug,
            max_tokens=cfg.generation.max_tokens,
            temperature=cfg.generation.temperature,
            top_p=cfg.generation.top_p,
            output_file_path=output_file
        )
        
        # Save results
        save_results(outputs, output_file)
        print(f"Saved baseline results to: {output_file}")
    
    end_time = time.time()
    print(f"Baseline execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
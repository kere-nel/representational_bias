"""Main script for running SAE gender bias experiments with Hydra configuration."""

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
    load_demographics,
    get_demographic_prefixes,
    create_name_based_prefixes,
    run_experiment,
    run_name_based_experiment,
    create_steering_vector,
    save_results,
    create_output_filepath,
    partition_questions,
    get_steering_vector_name
)
from core.models import get_model_config


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main experiment runner.
    
    Args:
        cfg: Hydra configuration object
    """
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup environment
    setup_environment()
    
    # Load model
    print(f"Loading model: {cfg.model.name}")
    model, tokenizer = load_model(cfg.model.name, debug=cfg.experiment.debug)
    model_config = get_model_config(cfg.model.name)
    
    # Load questions
    occupations = cfg.experiment.occupations
    if occupations == 'all':
        occupations = ['surgeon', 'carpenter', 'model', 'paralegal', 'social_worker']
    elif isinstance(occupations, str) and occupations != 'all':
        occupations = [occupations]
    
    questions = load_questions(cfg.paths.questions_dir, occupations)
    
    # Partition questions if needed
    print(f"Debug: part={cfg.experiment.part}, type={type(cfg.experiment.part)}")
    print(f"Debug: num_parts={cfg.experiment.num_parts}, type={type(cfg.experiment.num_parts)}")
    
    if int(cfg.experiment.num_parts) > 1:
        for occupation in questions:
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
    
    if cfg.experiment.implicit:
        # Name-based demographic experiments
        names_by_race_gender = load_demographics(cfg.paths.names_file)
        
        for occupation in occupations:
            print(f"Running name-based experiments for: {occupation}")
            
            outputs = run_name_based_experiment(
                model=model,
                tokenizer=tokenizer,
                questions=questions[occupation],
                occupation=occupation,
                steering_vector=steering_vector,
                names_by_race_gender=names_by_race_gender,
                model_name=cfg.model.name,
                debug=cfg.experiment.debug
            )
            
            # Save results with steering vector name included
            steering_name = get_steering_vector_name(cfg.paths.expert_file, cfg.paths.novice_file)
            model_clean = cfg.model.name.replace('/', '_')
            output_path = f"{cfg.paths.output_dir}/{model_clean}/{steering_name}/{occupation}_alpha{model_config.steering_alpha}_temp{cfg.generation.temperature}.pkl"
            save_results(outputs, output_path)
            
    else:
        # Standard demographic prefix experiments
        for occupation in occupations:
            print(f"Running experiments for: {occupation}")
            
            prefixes = get_demographic_prefixes(occupation)
            
            outputs = run_experiment(
                model=model,
                tokenizer=tokenizer,
                questions=questions[occupation],
                occupation=occupation,
                steering_vector=steering_vector,
                prefixes=prefixes,
                model_name=cfg.model.name,
                debug=cfg.experiment.debug,
                max_tokens=cfg.generation.max_tokens,
                temperature=cfg.generation.temperature,
                top_p=cfg.generation.top_p
            )
            
            # Save results with steering vector name included
            steering_name = get_steering_vector_name(cfg.paths.expert_file, cfg.paths.novice_file)
            model_clean = cfg.model.name.replace('/', '_')
            folder_path = f"{cfg.paths.output_dir}/{model_clean}/{steering_name}/alpha{model_config.steering_alpha}/temp{cfg.generation.temperature}/topp{cfg.generation.top_p}"
            os.makedirs(folder_path, exist_ok=True)
            debug_str = '_DEBUG' if cfg.experiment.debug else ''
            output_file = f"{folder_path}/{occupation}{debug_str}_{cfg.experiment.part}.pkl"
            
            save_results(outputs, output_file)
    
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    main()
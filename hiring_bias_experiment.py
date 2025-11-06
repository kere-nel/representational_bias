"""Hiring bias experiment script using expertise steering vectors."""

import os
import sys
import time
import pandas as pd
import pickle
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any

# Disable torch compilation to avoid nnsight compatibility issues
torch._dynamo.config.disable = True

from core import (
    setup_environment,
    load_model,
    create_steering_vector,
    save_results,
    get_steering_vector_name
)
from core.models import get_model_config


def generate_hiring_response(
    model, tokenizer, prompt: str, steering_vector=None, layer: int = None,
    alpha: float = 0.0, max_tokens: int = 50, temperature: float = 0.6,
    top_p: float = 0.8
) -> Dict[str, Any]:
    """Generate response for hiring decision prompt and calculate Yes/No logit differences.

    Args:
        model: Language model
        tokenizer: Model tokenizer
        prompt: The hiring decision prompt
        steering_vector: Optional steering vector
        layer: Layer to apply steering at
        alpha: Steering strength (0 for no steering)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Dictionary containing:
        - response: Generated response text
        - yes_no_logit_diff: Difference between Yes and No logits
        - yes_logit: Logit value for "Yes"
        - no_logit: Logit value for "No"
    """
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs['input_ids'].shape[-1]

    # Get Yes and No token IDs
    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

    # Generate with or without steering
    if alpha != 0.0 and steering_vector is not None:
        # Steered generation
        with model.generate(
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id
        ) as generator:
            with model.all():
                with generator.invoke(prompt):
                    layer_output = model.model.layers[layer]
                    # Handle different model architectures
                    if "gemma" in model.config.model_type.lower():
                        residual_tensor = layer_output.output[0]
                    else:
                        residual_tensor = layer_output.output

                    residual = residual_tensor.save()
                    residual_tensor[:] = residual + alpha * steering_vector
                    output = model.generator.output.save()
                    # Save logits from the language model head
                    logits = model.lm_head.output.save()
    else:
        # Baseline generation (no steering)
        with model.generate(
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id
        ) as generator:
            with model.all():
                with generator.invoke(prompt):
                    layer_output = model.model.layers[layer]
                    # Handle different model architectures
                    if "gemma" in model.config.model_type.lower():
                        residual_tensor = layer_output.output[0]
                    else:
                        residual_tensor = layer_output.output

                    residual = residual_tensor.save()
                    output = model.generator.output.save()
                    # Save logits from the language model head
                    logits = model.lm_head.output.save()

    # Decode response
    generated = output[:, input_len:]
    response = tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    # Calculate Yes/No logit differences from the first generated token position
    last_logits = logits[0, 0, :]  # Logits for the position where we predict the first new token
    yes_logit = last_logits[yes_token_id].item()
    no_logit = last_logits[no_token_id].item()
    yes_no_logit_diff = yes_logit - no_logit

    # Calculate projection of last token residual onto expertise vector
    expertise_projection = None
    last_token_residual = None

    # Get residual at the last input token position
    last_token_pos = input_len - 1
    if last_token_pos < residual.shape[1]:
        # Get residual at the last token position
        last_token_residual = residual[0, last_token_pos]#.cpu().float().detach().numpy()

        # Normalize steering vector and compute projection using einsum
        steering_norm = steering_vector / steering_vector.norm()
        #residual_tensor = torch.from_numpy(last_token_residual).float().to(steering_vector.device)
        # print("HEREEE")
        # print(last_token_residual.shape)
        # print(steering_norm.shape)
        expertise_projection = torch.einsum('d,d->', last_token_residual, steering_norm).item()

    return {
        "response": response,
        "yes_no_logit_diff": yes_no_logit_diff,
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "expertise_projection": expertise_projection,
        "last_token_residual": last_token_residual.cpu().float().detach().numpy() if last_token_residual is not None else None
    }


def run_hiring_bias_experiment(
    model, tokenizer, hiring_prompts_df: pd.DataFrame, steering_vector,
    model_name: str, alpha_positive: float = 5.0, alpha_negative: float = -5.0,
    max_tokens: int = 50, temperature: float = 0.6, top_p: float = 0.8,
    debug: bool = False
) -> List[Dict[str, Any]]:
    """Run hiring bias experiment with baseline and steered responses.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        hiring_prompts_df: DataFrame with hiring bias prompts
        steering_vector: Expertise steering vector
        model_name: Name of the model
        alpha_positive: Positive steering strength
        alpha_negative: Negative steering strength
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        debug: Debug mode (process only first 10 prompts)
        
    Returns:
        List of experiment results
    """
    config = get_model_config(model_name)
    results = []
    
    # Limit to first 100 prompts in debug mode
    prompts_to_process = hiring_prompts_df.head(2) if debug else hiring_prompts_df
    
    for idx, row in prompts_to_process.iterrows():
        print(f"Processing prompt {idx + 1}/{len(prompts_to_process)}")
        
        prompt = row['prompt']
        
        # Generate baseline response (no steering)
        baseline_result = generate_hiring_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steering_vector=steering_vector,
            layer=config.steering_layer,
            alpha=0.0,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        # Generate positive steering response
        positive_result = generate_hiring_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steering_vector=steering_vector,
            layer=config.steering_layer,
            alpha=alpha_positive,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )

        # Generate negative steering response
        negative_result = generate_hiring_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            steering_vector=steering_vector,
            layer=config.steering_layer,
            alpha=alpha_negative,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        # Store result with metadata
        result = {
            "prompt_id": idx,
            "prompt": prompt,
            "name": row.get('name', ''),
            "gender": row.get('gender', ''),
            "race": row.get('race', ''),
            "job_category": row.get('job_category', ''),
            "baseline_response": baseline_result["response"],
            "positive_steered_response": positive_result["response"],
            "negative_steered_response": negative_result["response"],
            "baseline_yes_no_logit_diff": baseline_result["yes_no_logit_diff"],
            "positive_yes_no_logit_diff": positive_result["yes_no_logit_diff"],
            "negative_yes_no_logit_diff": negative_result["yes_no_logit_diff"],
            "baseline_yes_logit": baseline_result["yes_logit"],
            "baseline_no_logit": baseline_result["no_logit"],
            "positive_yes_logit": positive_result["yes_logit"],
            "positive_no_logit": positive_result["no_logit"],
            "negative_yes_logit": negative_result["yes_logit"],
            "negative_no_logit": negative_result["no_logit"],
            "baseline_expertise_projection": baseline_result["expertise_projection"],
            "positive_expertise_projection": positive_result["expertise_projection"],
            "negative_expertise_projection": negative_result["expertise_projection"],
            "baseline_last_token_residual": baseline_result["last_token_residual"],
            "positive_last_token_residual": positive_result["last_token_residual"],
            "negative_last_token_residual": negative_result["last_token_residual"],
            "steering_layer": config.steering_layer,
            "steering_alpha": config.steering_alpha,
            "alpha_positive": alpha_positive,
            "alpha_negative": alpha_negative,
            "model_name": model_name
        }
        
        results.append(result)
    
    return results


@hydra.main(version_base=None, config_path="config", config_name="baseline_config")
def main(cfg: DictConfig) -> None:
    """Main hiring bias experiment runner.
    
    Args:
        cfg: Hydra configuration object
    """
    print("Hiring Bias Experiment Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Setup environment
    setup_environment()
    
    # Load model
    print(f"Loading model: {cfg.model.name}")
    model, tokenizer = load_model(cfg.model.name, debug=cfg.experiment.debug)
    model_config = get_model_config(cfg.model.name)
    
    # Load hiring bias prompts based on prompt_type
    prompt_type = cfg.experiment.get('prompt_type', 'high_bar')

    # Map prompt types to their corresponding CSV files
    prompt_files = {
        'simple': 'hiring_bias_prompts_20250825_130747.csv',
        'high_bar': 'hiring_bias_prompts_high_bar_20251029_233416.csv',
        'realistic': 'hiring_bias_prompts_realistic_20251029_233416.csv',  # To be created
        'realistic_gm': 'hiring_bias_prompts_realistic_gm_20251030_072738.csv',
        'high_bar_gm': 'hiring_bias_prompts_high_bar_gm_20251030_073454.csv'
    }

    if prompt_type not in prompt_files:
        raise ValueError(f"Invalid prompt_type '{prompt_type}'. Must be one of: {list(prompt_files.keys())}")

    hiring_prompts_path = f"{cfg.paths.hiring_prompts_dir}/{prompt_files[prompt_type]}"
    print(f"Loading hiring bias prompts ({prompt_type}) from: {hiring_prompts_path}")
    hiring_prompts_df = pd.read_csv(hiring_prompts_path)
    print(f"Loaded {len(hiring_prompts_df)} hiring bias prompts (prompt type: {prompt_type})")
    
    # Create steering vector
    print("Creating expertise steering vector...")
    steering_vector = create_steering_vector(
        model=model,
        tokenizer=tokenizer,
        expert_file=cfg.paths.expert_file,
        novice_file=cfg.paths.novice_file,
        layer=model_config.steering_layer,
        max_samples=int(cfg.experiment.max_steering_samples)
    )
    
    start_time = time.time()
    
    # Run hiring bias experiment
    print("Running hiring bias experiment...")
    results = run_hiring_bias_experiment(
        model=model,
        tokenizer=tokenizer,
        hiring_prompts_df=hiring_prompts_df,
        steering_vector=steering_vector,
        model_name=cfg.model.name,
        alpha_positive=model_config.steering_alpha,
        alpha_negative=model_config.alpha_negative or -model_config.steering_alpha,
        max_tokens=cfg.generation.max_tokens,
        temperature=cfg.generation.temperature,
        top_p=cfg.generation.top_p,
        debug=cfg.experiment.debug
    )
    
    # Save results
    steering_name = get_steering_vector_name(cfg.paths.expert_file, cfg.paths.novice_file)
    model_clean = cfg.model.name.replace('/', '_')
    debug_str = '_DEBUG' if cfg.experiment.debug else ''
    prompt_suffix = f"_{prompt_type}"  # Add prompt type to filename
    output_dir = f"{cfg.paths.output_dir}/hiring_bias/{model_clean}/{steering_name}/alpha{model_config.steering_alpha}/temp{cfg.generation.temperature}/topp{cfg.generation.top_p}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/hiring_bias_results{prompt_suffix}{debug_str}.pkl"

    save_results(results, output_file)
    print(f"Saved hiring bias results to: {output_file}")
    
    end_time = time.time()
    print(f"Hiring bias experiment execution time: {end_time - start_time:.4f} seconds")
    print(f"Processed {len(results)} hiring bias prompts")


if __name__ == "__main__":
    main()
"""Experiment and sampling functionality."""

import torch as t
import numpy as np
from typing import List, Dict, Any, Optional
from .models import get_model_config

def calculate_perplexity(model, tokenizer, text):
    """Calculate perplexity for a given text."""
    try:
        encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(model.device)
        
        with t.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = t.exp(loss)
        
        return perplexity.item()
    except:
        return float('inf')


t.no_grad()
def sample_model(
    model, tokenizer, questions, profession, steering_vector, layer,
    alpha=5, feature_idx=None, max_activation=None, debug=False, 
    prefix=None, num_responses=5, max_new_tokens=50, temperature=0.6, top_p=0.8
):
    """Sample model with and without steering.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        questions: List of questions to ask
        profession: Occupation/profession name
        steering_vector: Vector for steering the model
        layer: Layer to apply steering at
        alpha: Steering strength
        feature_idx: Optional feature index
        max_activation: Optional max activation
        debug: Debug mode (only process first question)
        prefix: Prefix to add to questions
        num_responses: Number of responses to generate
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        
    Returns:
        List of results with baseline and steered completions
    """
    eos_token_id = tokenizer.eos_token_id
    v_norm = steering_vector / steering_vector.norm()
    results = []    
    for q_i, q in enumerate(questions):
        t.manual_seed(42)
        t.cuda.manual_seed_all(42)

        q = f"{q} For reference, {prefix}" if prefix else q
        prompt = f"Question: {q} Answer:"
        if q_i == 0:
            print(prompt)
        if debug and q_i > 0:
            break

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs['input_ids'].shape[-1]

        # === BASELINE GENERATION ===
        with model.generate(
            num_return_sequences=num_responses, 
            max_new_tokens=max_new_tokens,
            do_sample=True, 
            temperature=temperature, 
            top_p=top_p,
            eos_token_id=eos_token_id
        ) as generator:
            with generator.invoke(prompt):

                layer_output = model.model.layers[layer]
                # Handle different model architectures - Gemma returns tuple, Llama returns tensor

                if "gemma" in model.config.model_type.lower():
                    residual_tensor = layer_output.output[0]
                else:
                    residual_tensor = layer_output.output ## llama models
                residual = residual_tensor.save()
                tokenized = tokenizer(prompt, return_offsets_mapping=True)
                offsets = tokenized["offset_mapping"]
                def get_token_position(char_pos, offsets):
                    for i, (start, end) in enumerate(offsets):
                        if start <= char_pos < end:
                            return i
                    return None

                pos_dot = get_token_position(prompt.find('.'), offsets)
                pos_q = get_token_position(prompt.find('?'), offsets)
                # score_pos = t.einsum(residual, v_norm.unsqueeze(-1))).squeeze(-1).mean(dim=0).save()
                
                score_pos = t.einsum('bsd,d->bs', residual, v_norm).mean(dim=0).save()
                
                score_dot = score_pos[pos_dot].save() if pos_dot is not None else 0
                score_q = score_pos[pos_q].save() if pos_q is not None else 0

                out_baseline = model.generator.output.save()
        generated_baseline = out_baseline[:, input_len:]
        # === STEERED GENERATION ===
        with model.generate(
            num_return_sequences=num_responses, 
            max_new_tokens=max_new_tokens,
            do_sample=True, 
            temperature=temperature, 
            top_p=top_p,
            eos_token_id=eos_token_id
        ) as generator:
            with model.all():
                with generator.invoke(prompt):
                    layer_output = model.model.layers[layer]
                    # Handle different model architectures - Gemma returns tuple, Llama returns tensor
                    if not hasattr(layer_output, "shape"):
                        residual_tensor = layer_output.output[0]
                        residual = residual_tensor.save()
                        residual_tensor[:] = residual + alpha * steering_vector
                    else:
                        residual_tensor = layer_output.output
                        residual = residual_tensor.save()
                        residual_tensor[:] = residual + alpha * steering_vector
                    out_steered = model.generator.output.save()
        generated_steered = out_steered[:, input_len:]

        # === Decode completions ===
        completions = []
        baseline_perplexities = []
        steered_perplexities = []
        
        for i in range(num_responses):
            base = tokenizer.decode(generated_baseline[i], skip_special_tokens=True)
            steer = tokenizer.decode(generated_steered[i], skip_special_tokens=True)
            
            # Calculate perplexities
            base_ppl = calculate_perplexity(model, tokenizer, base)
            steer_ppl = calculate_perplexity(model, tokenizer, steer)
            
            baseline_perplexities.append(base_ppl)
            steered_perplexities.append(steer_ppl)
            
            completions.append({
                "response_id": i,
                "baseline": base,
                "steered": steer,
                "baseline_perplexity": base_ppl,
                "steered_perplexity": steer_ppl
            })

        results.append({
            "question_id": q_i,
            "question": q,
            "score_dot": score_dot.item() if score_dot != 0 else 0,
            "score_qmark": score_q.item() if score_q != 0 else 0,
            "all_scores": score_pos.cpu().detach().tolist(),
            "residual": residual[0].cpu().float().detach().numpy(),
            "completions": completions,
            "avg_baseline_perplexity": np.mean(baseline_perplexities),
            "avg_steered_perplexity": np.mean(steered_perplexities),
        })

    return results


def run_experiment(
    model, tokenizer, questions, occupation, steering_vector, 
    prefixes, model_name, debug=False, max_tokens=100, temperature=0.6, top_p=0.8
):
    """Run complete experiment with multiple prefixes.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        questions: Questions for the occupation
        occupation: Occupation name
        steering_vector: Steering vector
        prefixes: Dictionary of experiment prefixes
        model_name: Name of the model
        debug: Debug mode flag
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p parameter
        
    Returns:
        List of experiment results
    """
    config = get_model_config(model_name)
    outputs = []
    
    for experiment, prefix in prefixes.items():
        if debug and experiment not in {"child", "none"}:
            continue

        results = sample_model(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            profession=occupation,
            prefix=prefix,
            steering_vector=steering_vector,
            layer=config.steering_layer,
            alpha=config.steering_alpha,
            num_responses=5,
            max_new_tokens=max_tokens,
            debug=debug,
            temperature=temperature,
            top_p=top_p
        )

        # Append metadata to each record
        for r in results:
            r["occupation"] = occupation
            r["experiment"] = experiment
            outputs.append(r)
    
    return outputs


def run_baseline_experiment(
    model, tokenizer, questions, occupation, steering_vector, 
    prefixes, model_name, alpha_positive=5.0, alpha_negative=-5.0, 
    debug=False, max_tokens=100, temperature=0.6, top_p=0.8, 
    output_file_path=None
):
    """Run baseline experiment with positive and negative alpha steering.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        questions: Questions for the occupation
        occupation: Occupation name
        steering_vector: Steering vector
        prefixes: Dictionary of experiment prefixes
        model_name: Name of the model
        alpha_positive: Positive steering alpha value
        alpha_negative: Negative steering alpha value
        debug: Debug mode flag
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Top-p parameter
        output_file_path: Path for saving intermediate results
        
    Returns:
        List of experiment results with both positive and negative steering
    """
    config = get_model_config(model_name)
    outputs = []
    
    for experiment, prefix in prefixes.items():
        if debug and experiment not in {"prefession", "none"}:
            continue

        # Run with positive alpha
        results_positive = sample_model(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            profession=occupation,
            prefix=prefix,
            steering_vector=steering_vector,
            layer=config.steering_layer,
            alpha=alpha_positive,
            num_responses=5,
            max_new_tokens=max_tokens,
            debug=debug,
            temperature=temperature,
            top_p=top_p
        )

        # Append metadata to each record for positive steering
        for r in results_positive:
            r["occupation"] = occupation
            r["experiment"] = experiment
            r["alpha"] = alpha_positive
            r["steering_direction"] = "positive"
            r["steering_layer"] = config.steering_layer
            r["steering_alpha"] = config.steering_alpha
            outputs.append(r)

        # Run with negative alpha
        results_negative = sample_model(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            profession=occupation,
            prefix=prefix,
            steering_vector=steering_vector,
            layer=config.steering_layer,
            alpha=alpha_negative,
            num_responses=5,
            max_new_tokens=max_tokens,
            debug=debug,
            temperature=temperature,
            top_p=top_p
        )

        # Append metadata to each record for negative steering
        for r in results_negative:
            r["occupation"] = occupation
            r["experiment"] = experiment
            r["steering_direction"] = "negative"
            r["steering_layer"] = config.steering_layer
            r["steering_alpha"] = config.steering_alpha
            outputs.append(r)
    
    return outputs


def run_name_based_experiment(
    model, tokenizer, questions, occupation, steering_vector, 
    names_by_race_gender, model_name, debug=False
):
    """Run experiment with name-based demographic prefixes.
    
    Args:
        model: Language model  
        tokenizer: Model tokenizer
        questions: Questions for the occupation
        occupation: Occupation name
        steering_vector: Steering vector
        names_by_race_gender: Demographic names data
        model_name: Name of the model
        debug: Debug mode flag
        
    Returns:
        Nested dictionary of results by race/gender/experiment type
    """
    config = get_model_config(model_name)
    outputs = {}
    
    for race, gender_list in names_by_race_gender.items():
        if race not in outputs:
            outputs[race] = {}
            
        for gender, names in gender_list.items():
            if debug and (race != 'white' and gender != 'male'):
                continue
                
            outputs[race][gender] = {}
            outputs[race][gender]['names'] = {
                "prefix": [], "outputs": [], "steer_outputs": [], "expert_scores": []
            }
            outputs[race][gender]['names_prof'] = {
                "prefix": [], "outputs": [], "steer_outputs": [], "expert_scores": []
            }
            
            for name in names:
                for exp, exp_prefix in [
                    ("names", f"I'm {name}. "),
                    ("names_prof", f"I'm {name}, and I'm a {occupation.replace('_', ' ')}. ")
                ]:
                    outputs[race][gender][exp]['prefix'].append(exp_prefix)
                    
                    results = sample_model(
                        model=model,
                        tokenizer=tokenizer,
                        questions=questions,
                        profession=occupation,
                        prefix=exp_prefix,
                        steering_vector=steering_vector,
                        layer=config.steering_layer,
                        alpha=config.steering_alpha,
                        num_responses=1,
                        max_new_tokens=50,
                        debug=debug
                    )
                    
                    # Store results in the expected format
                    outputs[race][gender][exp]["outputs"].append([r["completions"] for r in results])
                    outputs[race][gender][exp]["steer_outputs"].append([r["completions"] for r in results])
                    outputs[race][gender][exp]["expert_scores"].append([r["all_scores"] for r in results])
    
    return outputs
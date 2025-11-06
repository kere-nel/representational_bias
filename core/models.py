"""Model loading and steering functionality."""

import torch as t
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from config import MODEL_CONFIGS, ACCESS_TOKEN
from more_itertools import chunked 


def setup_model_authentication():
    """Set up HuggingFace authentication."""
    login(ACCESS_TOKEN)


def load_model(model_name, cache_dir="/projectnb/buinlp/kerenf", device="cuda", debug=False):
    """Load a language model with appropriate configuration.

    Args:
        model_name: Name of the model to load
        cache_dir: Directory for model caching
        device: Device to load model on
        debug: If True, force quantization for all models to save memory

    Returns:
        Tuple of (model, tokenizer)
    """
    setup_model_authentication()

    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Model {model_name} not configured")

    dtype = t.bfloat16

    # Configure quantization for large models or if debug mode is enabled
    if config.quantization_config or debug:
        # Use existing config or default debug quantization
        if config.quantization_config:
            quantization_config = BitsAndBytesConfig(**config.quantization_config)
        else:
            # Default quantization for debug mode
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                max_memory={0: "10GB"}
            )

        model = LanguageModel(
            model_name,
            device_map=device,
            cache_dir=cache_dir,
            dispatch=True,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            max_memory=config.max_memory or {}
        )
    else:
        model = LanguageModel(
            model_name,
            device_map=device,
            cache_dir=cache_dir,
            dispatch=True,
            torch_dtype=dtype
        )

    return model, model.tokenizer


def setup_model_hooks(model, tokenizer):
    """Set up any necessary model hooks or configurations.
    
    Args:
        model: The loaded language model
        tokenizer: The model's tokenizer
        
    Returns:
        Configured model and tokenizer
    """
    # Any model-specific setup can go here
    return model, tokenizer


def get_model_config(model_name):
    """Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        ModelConfig object
    """
    config = MODEL_CONFIGS.get(model_name)
    if not config:
        raise ValueError(f"Model {model_name} not configured")
    return config

@t.no_grad()
def get_mean_representation(model, tokenizer, prompts, layer, batch_size=1):
    """Get mean representation from model at specified layer.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        prompts: List of text prompts
        layer: Layer index to extract representations from
        batch_size: processing batch size 
    Returns:
        Mean representations tensor
    """
    t.cuda.empty_cache()
    all_vectors = []

    for batch in chunked(prompts, batch_size):
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        attn_mask = tokens['attention_mask']

        with model.trace(batch):  # no generation
            x = model.model.layers[layer].output[0].save()  # [batch_size, seq_len, d_model]

        flat_x = x.reshape(-1, x.shape[-1])                 # [B*T, d_model]
        flat_mask = attn_mask.reshape(-1).bool()            # [B*T]
        all_vectors.append(flat_x[flat_mask])

    return t.cat(all_vectors, dim=0)
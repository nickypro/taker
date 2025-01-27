# Code mostly from TransformerLens
# https://github.com/neelnanda-io/TransformerLens/blob/main/transformer_lens/loading_from_pretrained.py
import types
import copy
from typing import Callable, Any, Optional, Dict
from dataclasses import dataclass
import einops
from transformers import AutoConfig
import torch

@dataclass
class ConfigClass:
    d_model: int
    d_head: int
    n_heads: int
    d_mlp: int
    n_layers: int
    n_ctx: int
    eps: float
    d_vocab: int
    act_fn: str
    normalization_type: str
    architecture: str
    tokenizer_name: str
    n_key_value_heads: int = None
    is_low_precision: bool = False
    attn_types: list = None
    use_attn_scale: bool = None
    attn_scale: float = None
    use_local_attn: bool = None
    window_size: Optional[int] = None
    scale_attn_by_inverse_layer_idx: bool = None
    parallel_attn_mlp: bool = False
    pre_layernorm: bool = True
    post_layernorm: bool = False
    positional_embedding_type: str = "standard"
    rotary_dim: Optional[int] = None
    rotary_base: Optional[int] = None
    final_rms: bool = False
    attn_scores_soft_cap: int = None
    output_logits_soft_cap: int = None
    gated_mlp: bool = False
    model_type: str = "causal"
    model_modality: str = "language" # language, vision, (maybe "speech" one day?)
    label2id: Optional[Dict[str, int]] = None # for vision transformers
    id2label: Optional[Dict[int, str]] = None
    image_size: int = 224

def convert_hf_model_config(official_model_name: str):
    """
    Returns the model config for a HuggingFace model, converted to a dictionary
    in the fig format.

    Takes the official_model_name as an input.
    """
    # Load HuggingFace model config
    #if 'llama' in official_model_name and 'open_llama' not in official_model_name:
    #    architecture = "LLaMAForCausalLM"
    #else:
    hf_config = AutoConfig.from_pretrained(official_model_name)
    architecture = hf_config.architectures[0]

    if architecture.lower() == "LlamaForCausalLM".lower():
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads, #?
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "MistralForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "eps": hf_config.rms_norm_eps,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads, #?
            "use_local_attn": True,
            "gated_mlp": True,
        }
    elif architecture == "GemmaForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "eps": hf_config.rms_norm_eps,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads, #?
            "use_local_attn": True,
            "gated_mlp": True,
        }
    elif architecture == "Gemma2ForCausalLM":
        # Architecture for Gemma-2 9b and Gemma-2 27b models
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.head_dim,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": hf_config.rope_theta,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads, #?
            "use_attn_scale": True,
            "attn_scale": hf_config.query_pre_attn_scalar**0.5,
            "use_local_attn": True, #
            "window_size": hf_config.sliding_window, # 4096
            #"initializer_range": hf_config.initializer_range,
            "attn_types": ["global", "local"] * 21,  # Alternate global and local attn
            "attn_scores_soft_cap": hf_config.attn_logit_softcapping,
            "output_logits_soft_cap": hf_config.final_logit_softcapping,
            "gated_mlp": True,
            "final_rms": True,
            "pre_layernorm": True, # before and after!
            "post_layernorm": True,
        }
    elif architecture == "GPTNeoForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_heads,
            "n_heads": hf_config.num_heads,
            "d_mlp": hf_config.hidden_size * 4,
            "n_layers": hf_config.num_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "attn_types": hf_config.attention_layers,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": False,
            "use_local_attn": True,
            "window_size": hf_config.window_size,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPT2LMHeadModel":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_ctx,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
            "pre_layernorm": False,
            "post_layernorm": True,
        }
    elif architecture == "OPTForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.ffn_dim,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPTJForCausalLM":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": 4 * hf_config.n_embd,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_positions,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.rotary_dim,
            "normalization_type": "LN",
        }
    elif architecture == "GPTNeoXForCausalLM":
        cfg_dict = {
            "model_type": "causal",
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "normalization_type": "LN",
        }
        rotary_pct = hf_config.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    elif architecture == "RobertaForMaskedLM":
        cfg_dict = {
            "model_type": "masked",
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
            "pre_layernorm": False,
            "post_layernorm": True,
        }
    elif architecture == "PhiForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            #"initializer_range": hf_config.initializer_range,
            "normalization_type": "LN",
            "positional_embedding_type": "rotary",
            "rotary_base": hf_config.rope_theta,
            "use_local_attn": False,
            "gated_mlp": False,
            #"trust_remote_code": True,
            #"use_attn_scale": True,
            "parallel_attn_mlp": True,
        }
    elif architecture == "Phi3ForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "eps": hf_config.rms_norm_eps,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads, #?
            "gated_mlp": True,
        }
    elif architecture == "ViTForImageClassification":
        cfg_dict = {
            "model_type": "classification",
            "model_modality": "vision",
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": None, # max positional embeddings
            "eps": hf_config.layer_norm_eps,
            "d_vocab": None, # images are processed differently
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "use_local_attn": False,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
            "label2id": hf_config.label2id,
            "id2label": hf_config.id2label,
            "pre_layernorm": True,
            "post_layernorm": False,
            "image_size": hf_config.image_size,
        }
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    # All of these models use LayerNorm
    cfg_dict["architecture"] = architecture
    # The name such that AutoTokenizer.from_pretrained works
    cfg_dict["tokenizer_name"] = official_model_name
    return ConfigClass(**cfg_dict)


#####################################################################################
# Define Architecture Maps
#####################################################################################

def generate_attn_qkv_functions(weight_fn, bias_fn):
    return {
        "attn.W_Q"  : lambda layer, inpt=None: weight_fn(layer, "q", inpt),
        "attn.W_K"  : lambda layer, inpt=None: weight_fn(layer, "k", inpt),
        "attn.W_V"  : lambda layer, inpt=None: weight_fn(layer, "v", inpt),
        "attn.b_Q"  : lambda layer, inpt=None: bias_fn(layer, "q", inpt),
        "attn.b_K"  : lambda layer, inpt=None: bias_fn(layer, "k", inpt),
        "attn.b_V"  : lambda layer, inpt=None: bias_fn(layer, "v", inpt),
    }

def update_param(module, param_key, new_param):
    params = module.state_dict()
    assert param_key in params
    params[param_key] = new_param
    module.load_state_dict(params)

def generate_sizes_dict(einops_str, cfg):
    sizes_dict = {}
    if "qkv" in einops_str:
        sizes_dict["qkv"] = 3
    if "d_head" in einops_str:
        sizes_dict["d_head"] = cfg.d_head
    if "n_heads" in einops_str:
        sizes_dict["n_heads"] = cfg.n_heads
    if "d_model" in einops_str:
        sizes_dict["d_model"] = cfg.d_model
    if "d_mlp" in einops_str:
        sizes_dict["d_mlp"] = cfg.d_mlp
    if "n_layers" in einops_str:
        sizes_dict["n_layers"] = cfg.n_layers
    return sizes_dict


# Meta OPT and Galactica Models
###############################

opt_model_map = {
    "model"           : "model",
    "layers"          : "model.decoder.layers",
    "embed"           : "model.decoder.embed_tokens",
    "embed.W_E"       : "model.decoder.embed_tokens.weight",
    "pos_embed.W_pos" : "model.decoder.embed_positions",
    # in OPT, ln_final is only added for backwards compatibility
    "ln_final"        : "model.decoder.final_layer_norm",
    "ln_final.w"      : "model.decoder.final_layer_norm.weight",
    "ln_final.b"      : "model.decoder.final_layer_norm.bias",
    "unembed"         : "lm_head",
    "unembed.W_U"     : "lm_head.weight.T",
    "unembed.b_U"     : None,
}


def build_opt_layer_map(cfg: ConfigClass):
    attn_proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "out_proj"}

    def opt_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads d_head) d_model"
        my_shape    = "n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W

        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def opt_qkv_bias(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads d_head)"
        my_shape    = "n_heads d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        if inpt is None:
            b = attn_proj.bias
            b = einops.rearrange(b, f"{their_shape} -> {my_shape}", **sizes)
            return b

        # Set mode
        b = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "bias", b)


    opt_layer_map = {
        "attn.ln_in"           : "self_attn_layer_norm",
        "attn.ln_in.w"         : "self_attn_layer_norm.weight",
        "attn.ln_in.b"         : "self_attn_layer_norm.bias",

        "attn"          : "self_attn",
        "attn.q_proj"   : "self_attn.q_proj",
        "attn.k_proj"   : "self_attn.k_proj",
        "attn.v_proj"   : "self_attn.v_proj",

        **generate_attn_qkv_functions(opt_qkv_weight, opt_qkv_bias),

        "attn.out_proj" : "self_attn.out_proj",
        "attn.W_O"      : "self_attn.out_proj.weight",
        "attn.b_O"      : "self_attn.out_proj.bias",

        "attn.inv_out_proj" : "self_attn.inv_out_proj",
        "attn.W_O_inv"  : "self_attn.inv_out_proj.weight",
        "attn.b_O_inv"  : "self_attn.inv_out_proj.inverse_bias",

        "mlp.ln_in"           : "final_layer_norm",
        "mlp.ln_in.w"         : "final_layer_norm.weight",
        "mlp.ln_in.b"         : "final_layer_norm.bias",

        "mlp.in_proj"   : "fc1",
        "mlp.W_in"      : "fc1.weight",
        "mlp.b_in"      : "fc1.bias",

        "activation_fn" : "activation_fn",

        "mlp.out_proj"  : "fc2",
        "mlp.W_out"     : "fc2.weight",
        "mlp.b_out"     : "fc2.bias",
    }
    return opt_layer_map

# LLaMa Models
##############

llama_model_map = {
    "model"           : "model",
    "layers"          : "model.layers",
    "embed"           : "model.embed_tokens",
    "embed.W_E"       : "model.embed.weights",
    "pos_embed"       : "model.embed_positions",
    "pos_embed.W"     : "model.embed_positions.weight",
    "ln_final"        : "model.norm",
    "ln_final.w"      : "model.norm.weight",
    "unembed"         : "lm_head",
    "unembed.W_U"     : "lm_head.weight.T",
}

def build_llama_layer_map(cfg: ConfigClass):
    attn_proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
    mlp_proj_map = {"mlp.in_proj": "up_proj", "mlp.out_proj": "down_proj", "mlp.gate_proj": "gate_proj"}

    def llama_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads d_head) d_model"
        my_shape    = "n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W

        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def llama_attn_bias(layer, key: str, _inpt: Optional[Any]=None):
        # Create fake bias with zeros because is easier to handle
        their_shape = "(n_heads d_head)"
        my_shape    = "n_heads d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        attn = layer.self_attn
        _proj = get_attrs(attn, attn_proj_map[key]).weight
        b = torch.zeros(
            _proj.shape[:-1], dtype=_proj.dtype, device=_proj.device
        )
        if key == "o":
            return b
        return einops.rearrange(b, f"{their_shape} -> {my_shape}", **sizes)


    def llama_mlp_bias(layer, key: str, _inpt: Optional[Any]=None):
        mlp = layer.mlp
        _proj = get_attrs(mlp, mlp_proj_map[key]).weight
        b = torch.zeros(_proj.shape[:-1], dtype=_proj.dtype, device=_proj.device)
        return b

    def get_attn_weights(attn_outputs):
        attn_out, attn_weights, past_key_value = attn_outputs
        return attn_weights

    def set_attn_weights(attn_weights, orig_output):
        return orig_output[0], attn_weights, orig_output[2]


    llama_layer_map = {
        "attn.ln_in"           : "input_layernorm",
        "attn.ln_in.w"         : "input_layernorm.weight",
        "attn.ln_in.b"         : None,

        "attn"          : "self_attn",
        "attn.q_proj"   : "self_attn.q_proj",
        "attn.k_proj"   : "self_attn.k_proj",
        "attn.v_proj"   : "self_attn.v_proj",

        **generate_attn_qkv_functions(llama_qkv_weight, llama_attn_bias),

        "attn.out_proj" : "self_attn.o_proj",
        "attn.W_O"      : "self_attn.o_proj.weight",
        "attn.b_O"      : lambda layer, _inpt=None: llama_attn_bias(layer, "o", _inpt),

        "attn.inv_out_proj" : "self_attn.inv_out_proj",
        "attn.W_O_inv"  : "self_attn.inv_out_proj.weight",
        "attn.b_O_inv"  : "self_attn.inv_out_proj.bias",

        "mlp.ln_in"           : "post_attention_layernorm",
        "mlp.ln_in.w"         : "post_attention_layernorm.weight",
        "mlp.ln_in.b"         : None,

        "mlp"           : "mlp",
        "mlp.in_proj"   : "mlp.up_proj",
        "mlp.gate_proj" : "mlp.gate_proj",
        "mlp.W_in"      : "mlp.up_proj.weight",
        "mlp.W_gate"    : "mlp.gate_proj.weight",
        "mlp.b_in"      : lambda layer, _inpt=None: llama_mlp_bias(layer, "mlp.in_proj", _inpt),
        "mlp.b_gate"    : lambda layer, _inpt=None: llama_mlp_bias(layer, "mlp.out_proj", _inpt),

        "activation_fn" : "mlp.act_fn",

        "mlp.out_proj"  : "mlp.down_proj",
        "mlp.W_out"     : "mlp.down_proj.weight",
        "mlp.b_out"     : lambda layer, _inpt=None: llama_mlp_bias(layer, "mlp.gate_proj", _inpt),
    }
    return llama_layer_map

# Mistral Model
###############

mistral_model_map = {
    "model"           : "model",
    "layers"          : "model.layers",
    "embed"           : "model.embed_tokens",
    "embed.W_E"       : "model.embed.weights",
    "pos_embed"       : "model.embed_positions",
    "pos_embed.W"     : "model.embed_positions.weight",
    "ln_final"        : "model.norm",
    "ln_final.w"      : "model.norm.weight",
    "unembed"         : "model.lm_head",
    "unembed.W_U"     : "model.lm_head.weight.T",
    "unembed.b_U"     : None,
}

def build_mistral_layer_map(cfg: ConfigClass):
    attn_proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
    mlp_proj_map = {"mlp.in_proj": "up_proj", "mlp.out_proj": "down_proj", "mlp.gate_proj": "gate_proj"}

    def mistral_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        if key in ['k', 'v']:  # Adjust for Mistral's unique n_key_value_heads
            their_shape = "(n_key_value_heads d_head) d_model"
        else:
            their_shape = "(n_heads d_head) d_model"
        my_shape = "n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W

        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def mistral_attn_bias(layer, key: str, _inpt: Optional[Any]=None):
        # Create fake bias with zeros because is easier to handle
        their_shape = "(n_heads d_head)"
        my_shape = "n_heads d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        attn = layer.self_attn
        _proj = get_attrs(attn, attn_proj_map[key]).weight
        b = torch.zeros(_proj.shape[:-1], dtype=_proj.dtype, device=_proj.device)
        if key == "o":
            return b
        return einops.rearrange(b, f"{their_shape} -> {my_shape}", **sizes)

    def mistral_mlp_bias(layer, key: str, _inpt: Optional[Any]=None):
        mlp = layer.mlp
        _proj = get_attrs(mlp, mlp_proj_map[key]).weight
        b = torch.zeros(_proj.shape[:-1], dtype=_proj.dtype, device=_proj.device)
        return b

    mistral_layer_map = {
        "attn.ln_in"           : "input_layernorm",
        "attn.ln_in.w"         : "input_layernorm.weight",
        "attn.ln_in.b"         : None,

        "attn"          : "self_attn",
        "attn.q_proj"   : "self_attn.q_proj",
        "attn.k_proj"   : "self_attn.k_proj",
        "attn.v_proj"   : "self_attn.v_proj",

        **generate_attn_qkv_functions(mistral_qkv_weight, mistral_attn_bias),

        "attn.inv_out_proj" : "self_attn.inv_out_proj",
        "attn.out_proj" : "self_attn.o_proj",
        "attn.W_O"      : "self_attn.o_proj.weight",
        "attn.b_O"      : lambda layer, _inpt=None: mistral_attn_bias(layer, "o", _inpt),

        "mlp.ln_in"           : "post_attention_layernorm",
        "mlp.ln_in.w"         : "post_attention_layernorm.weight",
        "mlp.ln_in.b"         : None,

        "mlp"           : "mlp",
        "mlp.in_proj"   : "mlp.up_proj",
        "mlp.gate_proj" : "mlp.gate_proj",
        "mlp.W_in"      : "mlp.up_proj.weight",
        "mlp.W_gate"    : "mlp.gate_proj.weight",
        "mlp.b_in"      : lambda layer, _inpt=None: mistral_mlp_bias(layer, "mlp.in_proj", _inpt),
        "mlp.b_gate"    : lambda layer, _inpt=None: mistral_mlp_bias(layer, "mlp.out_proj", _inpt),

        "activation_fn" : "mlp.act_fn",

        "mlp.out_proj"  : "mlp.down_proj",
        "mlp.W_out"     : "mlp.down_proj.weight",
        "mlp.b_out"     : lambda layer, _inpt=None: mistral_mlp_bias(layer, "mlp.gate_proj", _inpt),

    }

    return mistral_layer_map


# GEMMA
#######

gemma_model_map = {
"model" : "model",
"layers" : "model.layers",
"embed" : "model.embed_tokens",
"embed.W_E" : "model.embed.weights",
"pos_embed" : "model.embed_positions",
"pos_embed.W" : "model.embed_positions.weight",
"ln_final" : "model.norm",
"ln_final.w" : "model.norm.weight",
"unembed"     : "lm_head",
"unembed.W_U" : "lm_head.weight.T",
"unembed.b_U" : None,
}

def build_gemma_layer_map(cfg: ConfigClass):
    attn_proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
    mlp_proj_map = {"mlp.in_proj": "up_proj", "mlp.out_proj": "down_proj", "mlp.gate_proj": "gate_proj"}

    def gemma_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads d_head) d_model"
        my_shape    = "n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W

        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def gemma_attn_bias(layer, key: str, _inpt: Optional[Any]=None):
        # Create fake bias with zeros because is easier to handle
        their_shape = "(n_heads d_head)"
        my_shape    = "n_heads d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        attn = layer.self_attn
        _proj = get_attrs(attn, attn_proj_map[key]).weight
        b = torch.zeros(
            _proj.shape[:-1], dtype=_proj.dtype, device=_proj.device
        )
        if key == "o":
            return b
        return einops.rearrange(b, f"{their_shape} -> {my_shape}", **sizes)


    def gemma_mlp_bias(layer, key: str, _inpt: Optional[Any]=None):
        mlp = layer.mlp
        _proj = get_attrs(mlp, mlp_proj_map[key]).weight
        b = torch.zeros(_proj.shape[:-1], dtype=_proj.dtype, device=_proj.device)
        return b

    gemma_layer_map = {
        "attn.ln_in"           : "input_layernorm",
        "attn.ln_in.w"         : "input_layernorm.weight",
        "attn.ln_in.b"         : None,

        "attn"          : "self_attn",
        "attn.q_proj"   : "self_attn.q_proj",
        "attn.k_proj"   : "self_attn.k_proj",
        "attn.v_proj"   : "self_attn.v_proj",

        **generate_attn_qkv_functions(gemma_qkv_weight, gemma_attn_bias),

        "attn.out_proj" : "self_attn.o_proj",
        "attn.W_O"      : "self_attn.o_proj.weight",
        "attn.b_O"      : lambda layer, _inpt=None: gemma_attn_bias(layer, "o", _inpt),

        "attn.inv_out_proj" : "self_attn.inv_out_proj",
        "attn.W_O_inv"  : "self_attn.inv_out_proj.weight",
        "attn.b_O_inv"  : "self_attn.inv_out_proj.bias",

        "mlp.ln_in"           : "post_attention_layernorm",
        "mlp.ln_in.w"         : "post_attention_layernorm.weight",
        "mlp.ln_in.b"         : None,

        "mlp"           : "mlp",
        "mlp.in_proj"   : "mlp.up_proj",
        "mlp.gate_proj" : "mlp.gate_proj",
        "mlp.W_in"      : "mlp.up_proj.weight",
        "mlp.W_gate"    : "mlp.gate_proj.weight",
        "mlp.b_in"      : lambda layer, _inpt=None: gemma_mlp_bias(layer, "mlp.in_proj", _inpt),
        "mlp.b_gate"    : lambda layer, _inpt=None: gemma_mlp_bias(layer, "mlp.gate_proj", _inpt),

        "activation_fn" : "mlp.act_fn",

        "mlp.out_proj"  : "mlp.down_proj",
        "mlp.W_out"     : "mlp.down_proj.weight",
        "mlp.b_out"     : lambda layer, _inpt=None: gemma_mlp_bias(layer, "mlp.out_proj", _inpt),
    }
    return gemma_layer_map

# Gemma 2
#########

gemma2_model_map = {
    "model"           : "model",
    "layers"          : "model.layers",
    "embed"           : "model.embed_tokens",
    "embed.W_E"       : "model.embed_tokens.weight",
    "ln_final"        : "model.norm",
    "ln_final.w"      : "model.norm.weight",
    "unembed"         : "lm_head",
    "unembed.W_U"     : "lm_head.weight.T",
    "unembed.b_U"     : None,
}

def build_gemma2_layer_map(cfg: ConfigClass):
    attn_proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "o_proj"}
    mlp_proj_map = {"mlp.gate_proj": "gate_proj", "mlp.up_proj": "up_proj", "mlp.down_proj": "down_proj"}

    def gemma2_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads d_head) d_model"
        my_shape    = "n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W

        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def gemma2_attn_bias(layer, key: str, _inpt: Optional[Any]=None):
        # Gemma2 doesn't use biases in attention
        return None

    def gemma2_mlp_weight(layer, key: str, inpt: Optional[Any]=None):
        mlp = layer.mlp
        proj = get_attrs(mlp, mlp_proj_map[key])

        if inpt is None:
            return proj.weight
        update_param(proj, "weight", inpt)

    def gemma2_mlp_bias(layer, key: str, _inpt: Optional[Any]=None):
        # Gemma2 doesn't use biases in MLP
        return None

    gemma2_layer_map = {
        "attn.ln_in"    : "input_layernorm",
        "attn.ln_in.w"  : "input_layernorm.weight",
        "attn.ln_in.b"  : None,

        "attn"          : "self_attn",
        "attn.q_proj"   : "self_attn.q_proj",
        "attn.k_proj"   : "self_attn.k_proj",
        "attn.v_proj"   : "self_attn.v_proj",

        **generate_attn_qkv_functions(gemma2_qkv_weight, gemma2_attn_bias),

        "attn.out_proj" : "self_attn.o_proj",
        "attn.W_O"      : "self_attn.o_proj.weight",
        "attn.b_O"      : None,

        "attn.ln_out"   : "post_attention_layernorm",
        "attn.ln_out.w" : "post_attention_layernorm.weight",
        "attn.ln_out.b" : None,

        "mlp"           : "mlp",
        "mlp.gate_proj" : "mlp.gate_proj",
        "mlp.up_proj"   : "mlp.up_proj",
        "mlp.out_proj" : "mlp.down_proj",
        "mlp.W_gate"    : lambda layer, inpt=None: gemma2_mlp_weight(layer, "mlp.gate_proj", inpt),
        "mlp.W_up"      : lambda layer, inpt=None: gemma2_mlp_weight(layer, "mlp.up_proj", inpt),
        "mlp.W_out"    : lambda layer, inpt=None: gemma2_mlp_weight(layer, "mlp.down_proj", inpt),
        "mlp.b_gate"    : lambda layer, _inpt=None: gemma2_mlp_bias(layer, "mlp.gate_proj", _inpt),
        "mlp.b_up"      : lambda layer, _inpt=None: gemma2_mlp_bias(layer, "mlp.up_proj", _inpt),
        "mlp.b_out"    : lambda layer, _inpt=None: gemma2_mlp_bias(layer, "mlp.down_proj", _inpt),

        "activation_fn" : "mlp.act_fn",

        "mlp.ln_in"     : "pre_feedforward_layernorm",
        "mlp.ln_in.w"   : "pre_feedforward_layernorm.weight",
        "mlp.ln_in.b"   : None,

        "mlp.ln_out"    : "post_feedforward_layernorm",
        "mlp.ln_out.w"  : "post_feedforward_layernorm.weight",
        "mlp.ln_out.b"  : None,
    }

    return gemma2_layer_map

# PHI 1 and 2 models
####################

phi_model_map = {
    "model": "model",
    "layers": "model.layers",
    "embed": "model.embed_tokens",
    "embed.W_E": "model.embed_tokens.weight",
    "ln_final": "model.final_layernorm",
    "ln_final.w": "model.final_layernorm.weight",
    "ln_final.b": "model.final_layernorm.bias",
    "unembed"    : "lm_head",
    "unembed.W_U": "lm_head.weight.T",
    "unembed.b_U": "lm_head.bias",
}

def build_phi_layer_map(cfg: ConfigClass):
    attn_proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "dense"}
    mlp_proj_map = {"mlp.W_in": "fc1", "mlp.W_out": "fc2"}

    def phi_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_head d_head) d_model"
        my_shape = "n_head d_model d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W
        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def phi_attn_bias(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_head d_head)"
        my_shape = "n_head d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            b = attn_proj.bias
            b = einops.rearrange(b, f"{their_shape} -> {my_shape}", **sizes)
            return b
        # Set mode
        b = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "bias", b)

    def phi_mlp_bias(layer, key: str, inpt: Optional[Any]=None):
        mlp = layer.mlp
        _proj = get_attrs(mlp, mlp_proj_map[key]).weight
        b = torch.zeros(_proj.shape[:-1], dtype=_proj.dtype, device=_proj.device)
        return b

    phi_layer_map = {
        "attn.ln_in":   "input_layernorm",
        "attn.ln_in.w": "input_layernorm.weight",
        "attn.ln_in.b": "input_layernorm.bias",
        "attn": "self_attn",
        **generate_attn_qkv_functions(phi_qkv_weight, phi_attn_bias),
        "attn.W_O": "self_attn.dense.weight",
        "attn.b_O": "self_attn.dense.bias",
        "mlp.ln_in": "input_layernorm", # parallel attn mlp
        "mlp.ln_in.w": "input_layernorm.weight",
        "mlp.ln_in.b": "input_layernorm.bias",
        "mlp": "mlp",
        "activation_fn" : "mlp.act_fn",
        "mlp.W_in"      : "mlp.fc1.weight",
        "mlp.b_in"      : lambda layer, _inpt=None: phi_mlp_bias(layer, "mlp.fc1", _inpt),
        "mlp.W_gate"    : "mlp.fc2.weight",
        "mlp.b_gate"    : lambda layer, _inpt=None: phi_mlp_bias(layer, "mlp.fc2", _inpt),
    }

    return phi_layer_map

# PHI-3
#######

phi3_model_map = {
    "model": "model",
    "layers": "model.layers",
    "embed": "model.embed_tokens",
    "embed.W_E": "model.embed_tokens.weight",
    "ln_final": "model.norm",
    "ln_final.w": "model.norm.weight",
    "ln_final.b": "model.norm.bias",
    "unembed"    : "lm_head",
    "unembed.W_U": "lm_head.weight.T",
    "unembed.b_U": "lm_head.bias",
}

def build_phi3_layer_map(cfg: ConfigClass):
    attn_proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "dense"}

    def phi3_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_head d_head) d_model"
        my_shape = "n_head d_model d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Attention qkv proj is combined in phi:
        # op_size = self.num_heads * self.head_dim + 2 * (self.num_key_value_heads * self.head_dim)
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        # self.qkv_proj = nn.Linear(self.hidden_size, op_size, bias=False)

        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W
        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def phi3_attn_bias(layer, key: str, inpt: Optional[Any]=None):
        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            b = attn_proj.bias
            return b
        # Set mode
        update_param(attn_proj, "bias", inpt)

    def phi3_mlp_bias(layer, key: str, _inpt: Optional[Any]=None):
        mlp = layer.mlp
        if key == "mlp.b_in":
            _proj = mlp.gate_up_proj.weight
        elif key == "mlp.b_out":
            _proj = mlp.down_proj.weight
        else:
            raise ValueError(f"Unknown MLP bias key {key}")

        b = torch.zeros(_proj.shape[1] // 2 if key == "mlp.b_in" else _proj.shape[0], dtype=_proj.dtype)
        return b

    phi3_layer_map = {
        "attn.ln_in": "input_layernorm",
        "attn.ln_in.w": "input_layernorm.weight",
        "attn.ln_in.b": "input_layernorm.bias",

        "attn": "self_attn",
        **generate_attn_qkv_functions(phi3_qkv_weight, phi3_attn_bias),
        "attn.out_proj": "self_attn.o_proj",
        "attn.W_O": "self_attn.o_proj.weight",
        "attn.b_O": "self_attn.o_proj.bias",

        "mlp.ln_in"  : "post_attention_layernorm",
        "mlp.ln_in.w": "post_attention_layernorm.weight",
        "mlp.ln_in.b": None,

        "mlp": "mlp",
        #"mlp.in_proj" : "mlp.gate_up_proj", # this combined up and gate proj matrices
        "mlp.out_proj": "mlp.down_proj",
        #"mlp.W_in": "mlp.gate_up_proj.weight",
        #"mlp.b_in": lambda layer, _inpt=None: phi3_mlp_bias(layer, "mlp.b_in", _inpt),
        "mlp.W_out": "mlp.down_proj.weight",
        "mlp.b_out": lambda layer, _inpt=None: phi3_mlp_bias(layer, "mlp.b_out", _inpt),
        "activation_fn": "mlp.activation_fn",
    }

    return phi3_layer_map


# GPT NEO X and Pythia Models
#############################

gpt_neox_model_map = {
    "model"           : "base_model",
    "layers"          : "base_model.layers",
    "embed"           : "base_model.embed_in",
    "embed.W_E"       : "base_model.embed_in.weight",
    "pos_embed.W_pos" : "base_model.embed_pos.weight",
    "ln_final"        : "base_model.final_layer_norm",
    "ln_final.w"      : "base_model.final_layer_norm.weight",
    "ln_final.b"      : "base_model.final_layer_norm.bias",
    "unembed"         : "base_model.embed_out",
    "unembed.W_U"     : "base_model.embed_out.weight",
    "unembed.b_U"     : "base_model.embed_out.bias",
}

def build_gpt_neox_layer_map(cfg: ConfigClass):
    def gpt_neox_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads qkv d_head) d_model"
        my_shape    = "qkv n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)
        qkv_map = {"q": 0, "k": 1, "v": 2}
        index = qkv_map[key]

        # Get the head weights
        qkv_heads = layer.attention.query_key_value
        W = qkv_heads.weight
        W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)

        # Get mode
        if inpt is None:
            return W[index]

        # Set mode
        W[index] = inpt
        W = einops.rearrange(W, f"{my_shape} -> {their_shape}", **sizes)
        update_param(qkv_heads, "weight", W)

    def gpt_neox_qkv_bias(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads qkv d_head)"
        my_shape    = "qkv n_heads d_head"
        sizes = generate_sizes_dict(my_shape, cfg)
        qkv_map = {"q": 0, "k": 1, "v": 2}
        index = qkv_map[key]

        # Get the head biases
        qkv_head = layer.attention.query_key_value
        qkv_bias = qkv_head.bias
        qkv_bias = einops.rearrange(qkv_bias, f"{their_shape} -> {my_shape}", **sizes)

        # Get mode
        if inpt is None:
            return qkv_bias[index]

        # Set mode
        qkv_bias[index] = inpt
        qkv_bias = einops.rearrange(qkv_bias, f"{my_shape} -> {their_shape}", **sizes)
        update_param(qkv_head, "bias", qkv_bias)

    gpt_neox_layer_map = {
        "attn.ln_in"       : "input_layernorm",
        "attn.ln_in.w"     : "input_layernorm.weight",
        "attn.ln_in.b"     : "input_layernorm.bias",

        "attn"      : "attention",
        "attn.q_proj"   : None,
        "attn.k_proj"   : None,
        "attn.v_proj"   : None,

        **generate_attn_qkv_functions(gpt_neox_qkv_weight, gpt_neox_qkv_bias),

        "attn.out_proj" : "attention.dense",
        "attn.W_O"      : "attention.dense.weight",
        "attn.b_O"      : "attention.dense.bias",

        "attn.inv_out_proj" : "attention.inv_out_proj",
        "attn.W_O_inv"      : "attention.inv_out_proj.weight",
        "attn.b_O_inv"      : "attention.inv_out_proj.inverse_bias",

        "mlp.ln_in"       : "post_attention_layernorm",
        "mlp.ln_in.w"     : "post_attention_layernorm.weight",
        "mlp.ln_in.b"     : "post_attention_layernorm.bias",

        "mlp"         : "mlp",
        "mlp.in_proj" : "mlp.dense_h_to_4h",
        "mlp.out_proj": "mlp.dense_4h_to_h",
        "mlp.W_in"  : "mlp.dense_h_to_4h.weight",
        "mlp.W_out" : "mlp.dense_4h_to_h.weight",
        "mlp.b_in"  : "mlp.dense_h_to_4h.bias",
        "mlp.b_out" : "mlp.dense_4h_to_h.bias",
        "activation_fn" : "mlp.act",
    }
    return gpt_neox_layer_map

# GPT2 Models
#############

gpt2_model_map = {
    "model"           : "transformer",
    "layers"          : "transformer.h",
    "embed"           : "transformer.wte",
    "embed.W_E"       : "transformer.wte.weight",
    "pos_embed.W_pos" : "transformer.wpe",
    "ln_final"        : "transformer.ln_f",
    "ln_final.w"      : "transformer.ln_f.weight",
    "ln_final.b"      : "transformer.ln_f.bias",
    "unembed"         : "lm_head",
    "unembed.W_U"     : "lm_head.weight.T",
    "unembed.b_U"     : None,
}

def build_gpt2_layer_map(cfg: ConfigClass):
    def gpt2_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        their_shape = "d_model (qkv n_heads d_head)"
        my_shape    = "qkv n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)
        qkv_map = {"q": 0, "k": 1, "v": 2}
        index = qkv_map[key]

        # Get the head weights
        qkv_heads = layer.attn.c_attn
        W = qkv_heads.weight
        W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)

        # Get mode
        if inpt is None:
            return W[index]

        # Set mode
        W[index] = inpt
        W = einops.rearrange(W, f"{my_shape} -> {their_shape}", **sizes)
        update_param(qkv_heads, "weight", W)

    def gpt2_qkv_bias(layer, key: str, inpt: Optional[Any]=None):
        their_shape = "(qkv n_heads d_head)"
        my_shape    = "qkv n_heads d_head"
        sizes = generate_sizes_dict(my_shape, cfg)
        qkv_map = {"q": 0, "k": 1, "v": 2}
        index = qkv_map[key]

        # Get the head biases
        qkv_heads = layer.attn.c_attn
        qkv_bias = qkv_heads.bias
        qkv_bias = einops.rearrange(qkv_bias, f"{their_shape} -> {my_shape}", **sizes)

        # Get mode
        if inpt is None:
            return qkv_bias[index]

        # Set mode
        qkv_bias[index] = inpt
        qkv_bias = einops.rearrange(qkv_bias, f"{my_shape} -> {their_shape}", **sizes)
        update_param(qkv_heads, "bias", qkv_bias)

    # GPT2 uses Conv1D instead of Linear, so we must get the transpose
    def conv1d_weight(module, inpt=None):
        if inpt is None:
            return module.weight.T
        params = module.state_dict()
        params["weight"] = inpt.T
        module.load_state_dict(params)

    def gpt2_out_weight(layer, inpt=None):
        return conv1d_weight(layer.attn.c_proj, inpt)

    def gpt2_mlp_in_weight(layer, inpt=None):
        return conv1d_weight(layer.mlp.c_fc, inpt)

    def gpt2_mlp_out_weight(layer, inpt=None):
        return conv1d_weight(layer.mlp.c_proj, inpt)

    def get_attn_weight(attn_outputs):
            # outputs # a, present, (attentions)
            # TODO: make sure use_cache is true!
            attn_out, key_value_cache, attn_weights = attn_outputs
            return attn_weights


    gpt2_layer_map = {
        "attn.ln_out"       : "ln_1",
        "attn.ln_out.w"     : "ln_1.weight",
        "attn.ln_out.b"     : "ln_1.bias",
        "attn"      : "attn",
        "attn.q_proj"   : None,
        "attn.k_proj"   : None,
        "attn.v_proj"   : None,
        **generate_attn_qkv_functions(gpt2_qkv_weight, gpt2_qkv_bias),
        "attn.out_proj" : "attn.c_proj",
        "attn.W_O"      : lambda layer, inpt=None: gpt2_out_weight(layer, inpt),
        "attn.b_O"      : "attn.c_proj.bias",
        "attn.inv_out_proj" : "attn.inv_out_proj",
        "attn.W_O_inv"      : "attn.inv_out_proj.weight",
        "attn.b_O_inv"      : "attn.inv_out_proj.inverse_bias",
        "mlp.ln_out"       : "ln_2",
        "mlp.ln_out.w"     : "ln_2.weight",
        "mlp.ln_out.b"     : "ln_2.bias",
        "mlp"         : "mlp",
        "mlp.in_proj" : "mlp.c_fc",
        "mlp.out_proj": "mlp.c_proj",
        "mlp.W_in"  : lambda layer, inpt=None: gpt2_mlp_in_weight(layer, inpt),
        "mlp.W_out" : lambda layer, inpt=None: gpt2_mlp_out_weight(layer, inpt),
        "mlp.b_in"  : "mlp.c_fc.bias",
        "mlp.b_out" : "mlp.c_proj.bias",
        "activation_fn" : "mlp.act",
    }
    return gpt2_layer_map

#####################################################################################
# "Masked LM" Models
#####################################################################################

# Roberta Model Map
###################

roberta_model_map = {
    "model"           : "roberta",
    "layers"          : "roberta.encoder.layer",
    "embed"           : "roberta.embeddings",
    "embed.W_E"       : "roberta.embeddings.word_embeddings.weight",
    "pos_embed.W_pos" : "roberta.embeddings.position_embeddings",
    "ln_final"        : "lm_head.layer_norm",
    "ln_final.w"      : "lm_head.layer_norm.weight",
    "ln_final.b"      : "lm_head.layer_norm.bias",
    "unembed"         : "lm_head",
    "unembed.W_U"     : "lm_head.dense.weight",
    "unembed.b_U"     : None,
}

def build_roberta_layer_map(cfg: ConfigClass):
    attn_proj_map = {
        "q": "self.query",
        "k": "self.key",
        "v": "self.value",
        "o": "output.dense"
    }

    def roberta_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads d_head) d_model"
        my_shape    = "n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.attention
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W

        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def roberta_qkv_bias(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads d_head)"
        my_shape    = "n_heads d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.attention
        attn_proj = get_attrs(attn, attn_proj_map[key])

        if inpt is None:
            b = attn_proj.bias
            b = einops.rearrange(b, f"{their_shape} -> {my_shape}", **sizes)
            return b

        # Set mode
        b = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "bias", b)


    roberta_layer_map = {
        "attn.ln_out"           : "attention.output.LayerNorm",
        "attn.ln_out.w"         : "attention.output.LayerNorm.weight",
        "attn.ln_out.b"         : "attention.output.LayerNorm.bias",

        "attn"          : "attention",
        "attn.q_proj"   : "attention.self.query",
        "attn.k_proj"   : "attention.self.key",
        "attn.v_proj"   : "attention.self.value",

        **generate_attn_qkv_functions(roberta_qkv_weight, roberta_qkv_bias),

        "attn.out"      : "attention.output",
        "attn.out_proj" : "attention.output.dense",
        "attn.W_O"      : "attention.output.dense.weight",
        "attn.b_O"      : "attention.output.dense.bias",

        "attn.inv_out_proj" : "attention.inv_out_proj",
        "attn.W_O_inv"  : "attention.inv_out_proj.weight",
        "attn.b_O_inv"  : "attention.inv_out_proj.inverse_bias",

        "mlp.ln_out"           : "output.LayerNorm",
        "mlp.ln_out.w"         : "output.LayerNorm.weight",
        "mlp.ln_out.b"         : "output.LayerNorm.bias",

        # "mlp"           : "intermediate",
        "mlp.in_proj"   : "intermediate.dense",
        "mlp.W_in"      : "intermediate.dense.weight",
        "mlp.b_in"      : "intermediate.dense.bias",

        "activation_fn" : "intermediate.intermediate_act_fn",

        "mlp.out_proj"  : "output.dense",
        "mlp.W_out"     : "output.dense.weight",
        "mlp.b_out"     : "output.dense.bias",
    }
    return roberta_layer_map

#####################################################################################
# VISION TRANSFORMERS (eg: ViT)
#####################################################################################

# ViT
#####

vit_model_map = {
    "model"           : "vit",
    "layers"          : "vit.encoder.layer",
    "embed"           : "vit.embeddings",
    #"embed.W_embed"   : "vit.embeddings.patch_embeddings.weight", ? is CNN
    #"pos_embed.W_pos" : Can't see any ???
    "ln_final"        : "layernorm",
    "ln_final.w"      : "layernorm.weight",
    "ln_final.b"      : "layernorm.bias",
    "unembed"         : "classifier",
    "unembed.W_U"     : "classifier.weight",
    "unembed.b_U"     : "classifier.bias",
}

def build_vit_layer_map(cfg: ConfigClass):
    attn_proj_map = {
        "q": "attention.query",
        "k": "attention.key",
        "v": "attention.value",
        "o": "output.dense"
    }

    def vit_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads d_head) d_model"
        my_shape    = "n_heads d_head d_model"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.attention
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W

        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def vit_qkv_bias(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads d_head)"
        my_shape    = "n_heads d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.attention
        attn_proj = get_attrs(attn, attn_proj_map[key])

        if inpt is None:
            b = attn_proj.bias
            b = einops.rearrange(b, f"{their_shape} -> {my_shape}", **sizes)
            return b

        # Set mode
        b = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "bias", b)


    vit_layer_map = {
        "attn.ln_in"    : "layernorm_before",
        "attn.ln_in.w"  : "layernorm_before.weight",
        "attn.ln_in.b"  : "layernorm_before.bias",

        "attn"          : "attention",
        "attn.q_proj"   : "attention.attention.query",
        "attn.k_proj"   : "attention.attention.key",
        "attn.v_proj"   : "attention.attention.value",

        **generate_attn_qkv_functions(vit_qkv_weight, vit_qkv_bias),

        "attn.out"      : "attention.output",
        "attn.out_proj" : "attention.output.dense",
        "attn.W_O"      : "attention.output.dense.weight",
        "attn.b_O"      : "attention.output.dense.bias",

        "attn.inv_out_proj" : "attention.inv_out_proj",
        "attn.W_O_inv"  : "attention.inv_out_proj.weight",
        "attn.b_O_inv"  : "attention.inv_out_proj.inverse_bias",

        "mlp.ln_in"     : "layernorm_after",
        "mlp.ln_in.w"   : "layernorm_after.weight",
        "mlp.ln_in.b"   : "layernorm_after.bias",

        # "mlp"           : "intermediate",
        "mlp.in_proj"   : "intermediate.dense",
        "mlp.W_in"      : "intermediate.dense.weight",
        "mlp.b_in"      : "intermediate.dense.bias",

        "activation_fn" : "intermediate.intermediate_act_fn",

        "mlp.out_proj"  : "output.dense",
        "mlp.W_out"     : "output.dense.weight",
        "mlp.b_out"     : "output.dense.bias",
    }

    return vit_layer_map

#####################################################################################
# Encoder + Decoder Models
#####################################################################################

# T5 Model Map for PyTorch T5 Model
###################################

t5_model_map = {
    "model"           : "t5",
    "layers"          : "encoder.block",
    "embed"           : "shared",
    "embed.W_E"       : "shared.weight",
    "pos_embed.W_pos" : "encoder.embed_tokens.position_embeddings.weight",
    "ln_final"        : "encoder.final_layer_norm",
    "ln_final.w"      : "encoder.final_layer_norm.weight",
    "ln_final.b"      : "encoder.final_layer_norm.bias",
    "unembed"         : "lm_head",
    "unembed.W_U"     : "lm_head.weight",
    "unembed.b_U"     : None,
}

def build_t5_layer_map(cfg: ConfigClass):
    attn_proj_map = {
        "q": "self_attn.q",
        "k": "self_attn.k",
        "v": "self_attn.v",
        "o": "self_attn.o"
    }

    def t5_qkv_weight(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads, d_head) d_model"
        my_shape    = "n_heads, d_head, d_model"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        # Get mode
        if inpt is None:
            W = attn_proj.weight
            W = einops.rearrange(W, f"{their_shape} -> {my_shape}", **sizes)
            return W

        # Set mode
        W = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "weight", W)

    def t5_qkv_bias(layer, key: str, inpt: Optional[Any]=None):
        # Prepare shape changing
        their_shape = "(n_heads, d_head)"
        my_shape    = "n_heads, d_head"
        sizes = generate_sizes_dict(my_shape, cfg)

        # Get attn proj module
        attn = layer.self_attn
        attn_proj = get_attrs(attn, attn_proj_map[key])

        if inpt is None:
            b = attn_proj.bias
            b = einops.rearrange(b, f"{their_shape} -> {my_shape}", **sizes)
            return b

        # Set mode
        b = einops.rearrange(inpt, f"{my_shape} -> {their_shape}", **sizes)
        update_param(attn_proj, "bias", b)


    t5_layer_map = {
        "attn.ln_in"    : "self_attn.layer_norm",
        "attn.ln_in.w"  : "self_attn.layer_norm.weight",
        "attn.ln_in.b"  : "self_attn.layer_norm.bias",

        "attn"          : "self_attn",
        "attn.q_proj"   : "self_attn.q",
        "attn.k_proj"   : "self_attn.k",
        "attn.v_proj"   : "self_attn.v",

        **generate_attn_qkv_functions(t5_qkv_weight, t5_qkv_bias),

        "attn.out_proj" : "self_attn.o",
        "attn.W_O"      : "self_attn.o.weight",
        "attn.b_O"      : "self_attn.o.bias",

        "mlp.ln_in"     : "fc.layer_norm",
        "mlp.ln_in.w"   : "fc.layer_norm.weight",
        "mlp.ln_in.b"   : "fc.layer_norm.bias",

        "mlp"           : "fc",
        "mlp.in_proj"   : "fc.DenseReluDense.wi",
        "mlp.W_in"      : "fc.DenseReluDense.wi.weight",
        "mlp.b_in"      : "fc.DenseReluDense.wi.bias",

        "activation_fn" : "fc.act",

        "mlp.out_proj"  : "fc.DenseReluDense.wo",
        "mlp.W_out"     : "fc.DenseReluDense.wo.weight",
        "mlp.b_out"     : "fc.DenseReluDense.wo.bias",
    }
    return t5_layer_map


#####################################################################################
# Build Model Layer Map interfaces
#####################################################################################

# Define Helper Functions
#########################

def get_attrs(obj, attr_string):
    nested_attributes = attr_string.split('.')
    current_attr = obj
    for attr_name in nested_attributes:
        current_attr = getattr(current_attr, attr_name)
    return current_attr

def set_attrs(obj, attr_string, value, override=True):
    nested_attrs = attr_string.split('.')
    nested_attrs, final_attr = nested_attrs[:-1], nested_attrs[-1]
    current_attr = get_attrs(obj, ".".join(nested_attrs)) if len(nested_attrs) > 0 else obj
    if not override and hasattr(current_attr, final_attr):
        return
    setattr(current_attr, final_attr, value)

def get_model_key_map(config: ConfigClass):
    architecture = config.architecture
    if architecture == "OPTForCausalLM":
        return opt_model_map
    if architecture in ["LLaMAForCausalLM", "LlamaForCausalLM"]:
        return llama_model_map
    if architecture == "MistralForCausalLM":
        return mistral_model_map
    if architecture == "GemmaForCausalLM":
        return gemma_model_map
    if architecture == "Gemma2ForCausalLM":
        return gemma2_model_map
    if architecture == "PhiForCausalLM":
        return phi_model_map
    if architecture == "Phi3ForCausalLM":
        return phi3_model_map
    if architecture == "GPTNeoXForCausalLM":
        return gpt_neox_model_map
    if architecture == "GPT2LMHeadModel":
        return gpt2_model_map
    if architecture == "RobertaForMaskedLM":
        return roberta_model_map
    if architecture == "ViTForImageClassification":
        return vit_model_map

    raise NotImplementedError(f"Architecture {architecture} not implemented")

def get_layer_key_map(config: ConfigClass):
    architecture = config.architecture

    if architecture == "OPTForCausalLM":
        return build_opt_layer_map(config)
    if architecture in ["LLaMAForCausalLM", "LlamaForCausalLM"]:
        return build_llama_layer_map(config)
    if architecture == "MistralForCausalLM":
        return build_mistral_layer_map(config)
    if architecture == "GemmaForCausalLM":
        return build_gemma_layer_map(config)
    if architecture == "Gemma2ForCausalLM":
        return build_gemma2_layer_map(config)
    if architecture == "PhiForCausalLM":
        return build_phi_layer_map(config)
    if architecture == "Phi3ForCausalLM":
        return build_phi3_layer_map(config)
    if architecture == "GPTNeoXForCausalLM":
        return build_gpt_neox_layer_map(config)
    if architecture == "GPT2LMHeadModel":
        return build_gpt2_layer_map(config)
    if architecture == "RobertaForMaskedLM":
        return build_roberta_layer_map(config)
    if architecture == "ViTForImageClassification":
        return build_vit_layer_map(config)

    raise NotImplementedError(f"Architecture {architecture} not implemented")


# Define Real Model Map and Layer Maps
######################################

class ModelMap:
    def __init__(self, model, cfg):
        self.cfg         = cfg
        self.model       = model
        self.key_map     = get_model_key_map(cfg)

        # Handle layers
        self.orig_layers = self["layers"]
        self.layers = [
            ModelLayerMap(self.cfg, layer) for layer in self.orig_layers
        ]

    def __getitem__(self, __name: str):
        key = self.key_map[__name]
        return get_attrs(self.model, key)

    def __setitem__(self, key, inpt):
        keys = key.split('.')
        attr = keys[-1]
        module = get_attrs(self.model, ".".join(keys[:-1]))
        params = module.state_dict()
        params[attr] = inpt
        module.load_state_dict(params)

class ModelLayerMap:
    def __init__(self, cfg, layer):
        self.cfg   = cfg
        self.layer = layer
        self.key_map = get_layer_key_map(cfg)

    @property
    def names(self):
        return list(self.key_map.keys())

    def __contains__(self, __name):
        return (__name in self.key_map)

    def __getitem__(self, __name):
        key = self.key_map[__name]

        if isinstance(key, str):
            if key == "layer":
                return self.layer
            if "inv_out_proj" in key:
                return NotImplementedError()
            return get_attrs(self.layer, key)

        if isinstance(key, Callable):
            return key(self.layer)

    def __setitem__(self, __name: str, __value: Any) -> None:
        key = self.key_map[__name]
        if isinstance(key, Callable):
            return key(self.layer, __value)

        if key is None:
            return None

        if not isinstance(key, str):
            raise ValueError("Invalid key, must be string or callable")

        # Get the module and attribute name
        keys = key.split('.')
        module = get_attrs(self.layer, ".".join(keys[:-1]))
        attr   = keys[-1]

        if attr == "inv_out_proj":
            setattr(module, attr, __value)
            return

        # If setting an attribute of a module (eg: weights or biases), update
        params = module.state_dict()
        params[attr] = __value
        module.load_state_dict(params)
        return

    def __str__(self):
        out_str  = "Wrapper for Transformer Layer\n"
        out_str += self.key_map.keys().__str__()
        out_str += "\nOriginal Layer Structure:\n"
        out_str += self.layer.__str__()
        return out_str

    def __getattr__(self, __name):
        key = self.key_map[__name]

        # Find all names that start with __name followed by a dot
        prefix = __name + "."
        remaining_names = sorted([n.split(prefix, 1)[1] for n in self.names if n.startswith(prefix)])

        if isinstance(key, str):
            mod = get_attrs(self.layer, key)
            for n in remaining_names:
                set_attrs(mod, n, self[f"{__name}.{n}"], override=False)
            return mod

        if isinstance(key, Callable):
            return key(self.layer)

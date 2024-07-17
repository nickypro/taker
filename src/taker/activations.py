"""
Code for getting attention activations and evaluating model.
"""

from typing import Optional, Dict, Tuple, List
import datetime
import math
import os

import torch
from torch import Tensor
import numpy as np
import einops
from tqdm import tqdm

from .texts import prepare_dataset, infer_dataset_config
from .model import Model
from .data_classes import RunDataItem, ActivationCollector, ActivationSummaryHolder, ActivationOverview, EvalConfig
from .eval import evaluate_all, Generators

######################################################################################
# New code for getting attention activations and evaluating model
######################################################################################

def get_input_activations(opt: Model, eval_config: EvalConfig, dataset_item: dict):
    """ dataset_item --> opt --> (input_ids, text_activations, residual_stream) """
    model_modality = opt.cfg.model_modality
    other_data = {}

    if model_modality == "vision":
        raw_img   = dataset_item[eval_config.dataset_image_key]
        img_label = dataset_item[eval_config.dataset_image_label_key]

        pixel_values = opt.get_pixel_values(raw_img)
        inputs_embeds = opt.get_inputs_embeds(pixel_values=pixel_values)
        [_batch, _num_tokens, _d_model] = inputs_embeds.shape
        _outputs_embeds = opt.get_outputs_embeds(pixel_values=pixel_values)

        input_ids    = torch.tensor([[img_label]*_num_tokens])
        expected_ids = torch.tensor([[img_label, *([-1]*(_num_tokens-1))]])
        return input_ids, expected_ids

    if model_modality == "language":
        text  = dataset_item[eval_config.dataset_text_key]
        input_ids = opt.get_ids(text).detach()

        if opt.cfg.model_type == "masked":
            orig_ids = input_ids
            input_ids, indices = \
                Generators.run_random_masking(opt, eval_config, orig_ids)
            other_data["expected_ids"] = orig_ids
        else:
            other_data["expected_ids"] = input_ids[..., 1:]

        _outputs_embeds = opt.get_outputs_embeds(input_ids=input_ids)
        return input_ids, other_data

    raise NotImplementedError(f"Invalid model modality {model_modality}")

######################################################################################
# New code for getting attention activations and evaluating model
######################################################################################

def get_midlayer_data(opt: Model,
        dataset_name: str = None,
        sample_size: int = 10000,
        check_accuracy: bool = False,
        k: int = 10,
        check_skips: bool = False,
        skip_ids: set = None,
        skip_type: str = "blacklist",
        skip_input_or_output: str = "output",
        calculate_ff: bool = True,
        calculate_attn: bool = True,
        collect_ff: bool = False,
        collect_attn: bool = False,
        collect_ids: bool = False,
        dataset_texts_to_skip: int = None,
        random_subset_frac: float = None,
        eval_config: EvalConfig = None,
        masked_mode: bool = False,
        ):
    """
    Gets the activations of the midlayer ('key' layer) of MLPs and the pre_out layer of attention for each layer.
    """
    if eval_config is None:
        assert dataset_name is not None, "Must provide either an EvalConfig or a dataset name"
        eval_config = infer_dataset_config(dataset_name)
        eval_config.dataset_split = "train"
        eval_config.is_train_mode = True
        if dataset_texts_to_skip is not None:
            eval_config.num_texts_to_skip = dataset_texts_to_skip
    if "MaskedLM" in opt.cfg.architecture and masked_mode:
        eval_config.masked_model = True
    dataset    = prepare_dataset(eval_config)
    skip_eval  = eval_config.skip_token_strings or []

    do_ff      = calculate_ff or collect_ff
    do_attn    = calculate_attn or collect_attn
    do_collect = collect_ff or collect_attn or collect_ids

    # Get things ready for collection
    opt.hooks.disable_all_collect_hooks()

    if do_ff:
        ff_shape = (opt.cfg.n_layers, opt.cfg.d_mlp)
        ff_data = ActivationCollector( ff_shape, opt.output_device, collect_ff )
        opt.hooks.enable_collect_hooks(["mlp_pre_out"])

    # self-attention activation collector
    if do_attn:
        attn_shape = (opt.cfg.n_layers, opt.cfg.n_heads, opt.cfg.d_head)
        attn_data = ActivationCollector( attn_shape, opt.output_device, collect_attn )
        opt.hooks.enable_collect_hooks(["attn_pre_out"])

    if do_collect:
        criteria_raw = []

    if collect_ids:
        input_id_data = []
        output_id_data = []

    curr_count = 0
    texts_viewed = 0

    with tqdm(total=sample_size, desc=eval_config.dataset_name) as pbar:
        for data in dataset:
            texts_viewed += 1
            with torch.no_grad():

                # Get activations
                try:
                    input_ids, other_data = get_input_activations(opt, eval_config, data)
                except:
                    print("Error processing dataset input. Skipping...")
                    continue

                if do_ff:
                    ff_acts = opt.collect_recent_mlp_pre_out()
                    ff_acts = einops.rearrange(ff_acts, "b l t d -> (b t) l d")
                if do_attn:
                    attn_acts = opt.collect_recent_attn_pre_out()
                    attn_acts = einops.rearrange(attn_acts, "b l t nh dh -> (b t) l nh dh")

                # set up criteria for filtering which activations we actually want
                ids = einops.rearrange(input_ids, "b t -> (b t)")
                criteria = torch.ones_like(ids, dtype=torch.bool)

                if check_skips:
                    n_plus_one = int(skip_input_or_output == "output")
                    for index in range(len(input_ids[0])-n_plus_one):
                        pos = index + n_plus_one
                        if skip_type == "whitelist":
                            criteria[index] *= (input_ids[0, pos].item() in skip_ids)
                        if skip_type == "blacklist":
                            criteria[index] *= (input_ids[0, pos].item() not in skip_ids)

                if random_subset_frac:
                    criteria = get_random_subset(criteria, random_subset_frac)

                if collect_ids:
                    for token_index, input_id in enumerate(input_ids[0]):
                        if not criteria[token_index]:
                            continue
                        input_id_data.append(input_id)
                        output_id_data.append(input_ids[0, token_index+1] if token_index+1 < len(input_ids[0]) else torch.tensor(-1))

                # Collect activations according to criteria
                criteria_indices = criteria.nonzero().flatten()
                if do_ff:
                    ff_data.add_all(ff_acts[criteria_indices])
                if do_attn:
                    attn_data.add_all(attn_acts[criteria_indices])
                if do_collect:
                    for criterion in criteria:
                        criteria_raw.append(criterion.cpu())

                # Count. Check if we are done
                num_valid_tokens = criteria.sum()
                curr_count += num_valid_tokens
                pbar.update(int(num_valid_tokens))

                if curr_count > sample_size:
                    break

    output = {
        "texts_viewed": texts_viewed,
    }

    if calculate_ff:
        output["mlp"] = ActivationSummaryHolder(
            orig=ff_data.summary(dtype=opt.dtype),
        )
    if calculate_attn:
        output["attn"] = ActivationSummaryHolder(
            orig=attn_data.summary(dtype=opt.dtype),
        )

    if do_collect:
        output["raw"] = {"criteria": torch.stack(criteria_raw)}
    if collect_ff:
        output["raw"]["mlp"] = ff_data.get_raw()
    if collect_attn:
        output["raw"]["attn"] = attn_data.get_raw()
    if collect_ids:
        output["raw"]["input_ids"] = torch.stack(input_id_data)
        if len(output_id_data):
            output["raw"]["expected_ids"] = torch.stack(output_id_data)
        else:
            output["raw"]["expected_ids"] = torch.ones_like(output["raw"]["input_ids"]) * -1

    return ActivationOverview(**output)

#####################################################################################
#
#####################################################################################

def get_random_subset(keep_indices, frac):
    "Get random subset. E.g: [0,1,1,0,1,1], 0.5 -> [0,1,0,0,0,1]"
    ones_indices = torch.nonzero(keep_indices).reshape([-1])
    num_to_select = math.ceil(len(ones_indices) * frac)
    random_indices = torch.randperm(len(ones_indices))[:num_to_select]
    new_ones = torch.zeros_like(keep_indices)
    new_ones[ones_indices[random_indices]] = True
    return new_ones

def get_top_frac(values_tensor: Tensor, top_frac: float) -> Tuple[Tensor, float]:
    """
    Return top-k values and their fraction
    """
    shape = values_tensor.shape
    n_entries = np.prod(shape)
    k = int(top_frac * n_entries)

    topk_values = torch.topk(values_tensor.flatten(), k, dim=-1, largest=True, sorted=False)

    criteria = torch.zeros(n_entries, dtype=torch.bool)
    criteria[topk_values.indices] = True
    criteria = criteria.reshape(shape)

    threshold = float(topk_values.values.flatten().min())

    return criteria, threshold

def choose_attn_heads_by_mean(opt: Model,
        attn_scores: Tensor,
        top_frac: float,
        ):
    std_ratio_medians = torch.quantile(attn_scores.to(dtype=torch.float32), q=0.5, dim=-1)
    return get_top_frac(std_ratio_medians, top_frac)

def choose_attn_heads_by_median(opt: Model,
        attn_scores: Tensor,
        top_frac: float
        ):
    std_ratio_means = attn_scores.mean(dim=-1)
    return get_top_frac(std_ratio_means, top_frac)

def choose_attn_heads_by(key: str):
    choosing_map = {
        'mean': choose_attn_heads_by_mean,
        'median': choose_attn_heads_by_median,
    }
    return choosing_map[key]

def save_timestamped_tensor_dict(opt: Model,
        data: Dict[str, Tensor],
        name: str,
        path: str = None,
        file: str = None,
        ):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    path = path or f'tmp/{opt.cfg.model_size}'
    filename = f'{path}/{file}' or f'{path}/{opt.cfg.model_size}-{name}-{now}.pt'
    torch.save(data, filename)
    print(f'Saved {filename} to {opt.cfg.model_size}')
    return filename

def save_numpy_ff(opt: Model,
        freq_multiple: float,
        array: np.ndarray,
        name: str,
        ):
    filename = f'tmp/{opt.cfg.model_size}/{opt.cfg.model_size}-ff-{freq_multiple}x-{name}.npy'
    os.makedirs(f'tmp/{opt.cfg.model_size}', exist_ok=True)
    with open(filename, 'wb') as f:
        np.save(f, np.array(array))
    print("saved successfully")

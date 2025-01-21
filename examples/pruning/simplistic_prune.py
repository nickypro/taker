import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Dict
import einops
from tqdm import tqdm
from taker import Model
from taker.texts import prepare
from taker.eval import evaluate_all
from welford_torch import Welford
import time

def get_top_frac(values_tensor: Tensor, top_frac: float) -> Tuple[Tensor, float]:
    """ Returns:
    - criteria: Tensor of bools with shape (values_tensor.shape)
    - threshold: minimum value needed to be in the top_frac of values.
    """
    # Get the number of entries in the tensor, and the number of entries to get
    shape = values_tensor.shape
    n_entries = np.prod(shape)
    k = int(top_frac * n_entries)

    # Get the top k values
    topk_values = torch.topk(values_tensor.flatten(), k, dim=-1, largest=True, sorted=False)

    # Create a criteria tensor with value 1 for all values in topk_values
    criteria = torch.zeros(n_entries, dtype=torch.bool)
    criteria[topk_values.indices] = True
    criteria = criteria.reshape(shape)

    # Get the threshold value, the value above which all values are in topk_values
    try:
        threshold = float(topk_values.values.flatten().min())
    except:
        threshold = None

    return criteria, threshold

def get_activation_data(m: Model, dataset_name: str, split: str = 'train', num_tokens:int=1e5) -> Dict[str, Tensor]:
    """Gets the mean absolute activations of the midlayer ('key' layer) of MLPs for
    each layer, as well as for the pre_out layer of attention for each layer.
    """
    print(f"Collecting activation data for dataset: {dataset_name}, split: {split}")
    # prepare dataset
    ds, ds_text_label, _ = prepare(dataset_name, split=split)

    # setup model activation collection
    m.hooks.disable_all_collect_hooks()
    m.hooks.enable_collect_hooks(["mlp_pre_out", "attn_pre_out"])

    # save running mean + std of values online
    mlp_abs = Welford(dtype=m.dtype, device=m.device).detach()
    attn_abs = Welford(dtype=m.dtype, device=m.device).detach()

    for data in tqdm(ds):
        # run model so that activations are collected
        _ = m.get_outputs_embeds(text=data[ds_text_label])

        # collect activations
        attn_act = m.collect_recent_attn_pre_out()
        attn_act = einops.rearrange(attn_act, 'batch layer token head pos -> (batch token) layer head pos')

        mlp_act = m.collect_recent_mlp_pre_out()
        mlp_act = einops.rearrange(mlp_act, 'batch layer token pos -> (batch token) layer pos')

        # save statistics about the activaionn
        mlp_abs.add_all(mlp_act.abs())
        attn_abs.add_all(attn_act.abs())

        if mlp_abs.count > num_tokens:
            break

    return {
        "mlp":  mlp_abs.mean,
        "attn": attn_abs.mean,
    }

@torch.no_grad
def run_selective_pruning(m:Model, retain_dataset:str, forget_dataset:str, mlp_frac:float, attn_frac:float, sample_size:int=1e5):
    print(f"Running selective pruning with mlp_frac = {mlp_frac}, attn_frac = {attn_frac}")
    print(f"Retain dataset: {retain_dataset}, Forget dataset: {forget_dataset}")

    # collect activations
    retain_act = get_activation_data(m, retain_dataset, num_tokens=sample_size)
    forget_act = get_activation_data(m, forget_dataset, num_tokens=sample_size)

    # score the neurons
    mlp_scores = forget_act["mlp"] / (retain_act["mlp"] + 1e-5)
    attn_scores = forget_act["attn"] / (retain_act["attn"] + 1e-5)

    # select the highest scoring neurons
    mlp_top_neurons, mlp_threshold = get_top_frac(mlp_scores, mlp_frac)
    attn_top_neurons, attn_threshold = get_top_frac(attn_scores, attn_frac)

    print(f"MLP neurons selected for pruning: {mlp_top_neurons.sum().item()}")
    print(f"Attention neurons selected for pruning: {attn_top_neurons.sum().item()}")
    print(f"MLP threshold: {mlp_threshold}, Attention threshold: {attn_threshold}")

    # delete the selected neurons
    m.hooks.delete_mlp_neurons(mlp_top_neurons)
    m.hooks.delete_attn_neurons(attn_top_neurons)
    return

if __name__ == "__main__":
    print("## Starting selective pruning script")
    # taker.Model supports many models from huggingface. If not working, email me
    # - supports quantization also! "fp16" >= "hqq8" > "int8" > "hqq4" > "nf4" recommended
    m = Model("facebook/galactica-125m", limit=1000, dtype="nf4")

    # Define parameters for the selective pruning
    retain_dataset = "pile_codeless"
    forget_dataset = "code"
    all_datasets = [retain_dataset, forget_dataset]
    eval_sample_size    = 100000 # bigger tends to be better
    collect_sample_size = 100000 #
    mlp_frac  = 0.05 # prune 5% of mlp neurons here
    attn_frac = 0.00 # don't prune attn neurons here

    print("## Running evaluation before any pruning")
    results_before = evaluate_all(m, sample_size=eval_sample_size, datasets=all_datasets, dataset_tokens_to_skip=collect_sample_size)
    # NOTE: dataset_tokens_to_skip only required because the "code" dataset has no explicit "test" split. TODO: upload pre-split dataset

    print("## Running selective pruning procedure")
    run_selective_pruning(m, retain_dataset, forget_dataset, mlp_frac=mlp_frac, attn_frac=attn_frac, sample_size=collect_sample_size)

    print("## Running evaluation after pruning")
    results_after = evaluate_all(m, sample_size=eval_sample_size, datasets=all_datasets, dataset_tokens_to_skip=collect_sample_size)

    print("## Selective pruning process completed. Results:")
    print("pile loss before:", results_before["loss_data"]["pile_codeless"]["loss"],
                  "-> after:",  results_after["loss_data"]["pile_codeless"]["loss"])
    print("code loss before:", results_before["loss_data"]["code"]["loss"],
                  "-> after:",  results_after["loss_data"]["code"]["loss"])
    print(f'pile top1 accuracy before: {results_before["accuracy"]["pile_codeless"]["base"]:.2f}%',
                           f'-> after: { results_after["accuracy"]["pile_codeless"]["base"]:.2f}%')
    print(f'code top1 accuracy before: {results_before["accuracy"]["code"]["base"]:.2f}%',
                           f'-> after: { results_after["accuracy"]["code"]["base"]:.2f}%')

# MLP hooks gemma-2-2b (non-it)
# note: 26x layers with 16k width, d_in=2306, d_sae=16384
# total parameters: 2 * 26 * 2306 * 16384 = 1_964_638_208

import torch
import gc
from tqdm import tqdm
from taker import Model
from taker.eval import evaluate_all
from taker.data_classes import RunDataHistory
torch.set_grad_enabled(False)
import wandb

hook_config = """
post_mlp: collect
"""
m = Model("gpt2", limit=1000, hook_config=hook_config)
history = RunDataHistory()
wandb.init(m.model_repo, entity="seperability", project="SAE_tests")

# evaluate the model
from taker.eval import evaluate_all
gc.collect()
torch.cuda.empty_cache()
data = evaluate_all(m, 1e5, ["pile", "lm_eval:mmlu"])
history.add(data)

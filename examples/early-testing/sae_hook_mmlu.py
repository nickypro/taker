
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
post_mlp: sae_encode, collect, sae_decode
"""
m = Model("google/gemma-2-2b", limit=1000, hook_config=hook_config)
wandb.init(m.model_repo, project="SAE_tests", entity="seperability")
history = RunDataHistory()

# evaluate the model
from taker.eval import evaluate_all
gc.collect()
torch.cuda.empty_cache()
#data = evaluate_all(m, 1e5, ["pile", "mmlu"])
#history.add(data)

# show that the collection hooks are working, and that the SAE is not loaded
m.hooks.enable_collect_hooks(["post_mlp"])
m.get_outputs_embeds("Hello, world!")
acts = []
for layer in range(m.cfg.n_layers):
    acts.append( m.hooks.collects[f"layer_{layer}_post_mlp"].activation )
print("post_mlp shape:", acts[0].shape)

m.hooks.disable_all_collect_hooks()

# load the SAE
for layer in tqdm(range(m.cfg.n_layers), desc="Loading SAE"):
    sae_hook = m.hooks.neuron_sae_encode[f"layer_{layer}_post_mlp"]
    # >>> SAE.from_pretrained("gemma-scope-2b-pt-mlp-canonical", "layer_0/width_16k/canonical")
    #sae_hook.load_sae_from_pretrained("gemma-scope-2b-pt-mlp-canonical", f"layer_{layer}/width_65k/canonical")
    sae_hook.load_sae_from_pretrained("gemma-scope-2b-pt-mlp-canonical", f"layer_{layer}/width_16k/canonical")
    sae_hook.sae = sae_hook.sae.to(acts[layer].device)
    #print(sae_hook)
    #exit()

# test that the SAE collection hooks are good
m.hooks.enable_collect_hooks(["post_mlp"])
m.get_outputs_embeds("Hello, world!")
for layer in range(m.cfg.n_layers):
    act = m.hooks.collects[f"layer_{layer}_post_mlp"].activation
    print(act.shape)
m.hooks.disable_all_collect_hooks()

# evaluate the model again
gc.collect()
torch.cuda.empty_cache()
data = evaluate_all(m, 1e5, ["pile", "mmlu"])
history.add(data)

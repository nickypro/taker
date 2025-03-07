
# Pre-residual hooks
from taker import Model
import torch
from taker.activations import get_midlayer_data

#pre_decoder: sae_encode, collect, sae_decode
hook_config = """
pre_decoder: sae_encode, collect, sae_decode
mlp_pre_out: collect
"""
m = Model("gpt2", hook_config=hook_config)
#m = Model("gpt2")
#print(len(m.hooks["mlp_pre_out"]["collect"]))
print("model info:")
print(m.cfg.n_layers)
print(m.cfg.d_model)
print(m.cfg.n_heads)
print(m.cfg.d_mlp)
#---------------------
sae_hook_points = [point for point, layers in m.hooks.hook_config.hook_points.items() 
                   if 'all' in layers and any('sae' in hook for hook in layers['all'])]
sae_dict = dict.fromkeys(sae_hook_points)
#---------------------
print(m.hooks.hook_config.hook_points)
#print(m.hooks.hook_config.hook_points.items(), " hook points")
print(sae_hook_points, " hook points")
for layer in range(m.cfg.n_layers):
    sae_hook = m.hooks.neuron_sae_encode[f"layer_{layer}_pre_decoder"]
    sae_hook.load_sae_from_pretrained("gpt2-small-res-jb", f"blocks.{layer}.hook_resid_pre")
    #print(sae_hook.sae_config["d_sae"])

#testing enabling hhooks
#m.hooks.enable_collect_hooks(["pre_decoder", "mlp_pre_out"], run_assert=True)
m.hooks.enable_collect_hooks(["mlp_pre_out"], run_assert=True)
m.hooks.enable_collect_hooks(["pre_decoder"], run_assert=True)

#m.get_outputs_embeds("Hello, world!")
#m.get_outputs_embeds("kill all humans!")

print("sae data: ")
# Working with layers
#layer_0 = m.layers[0]

#stuff = layer_0
#print(stuff)
cripple_data = get_midlayer_data( m, "toxic", 100, collect_sae=True, collect_attn=False, collect_ff=True, calculate_attn=False, calculate_ff=False)
sae_acts = (m.hooks.collects["layer_0_pre_decoder"].activation)

#test deletes
print("deleting neurons")
#print("hooks ", m.hooks)
#m.hooks.collects["layer_0_pre_decoder"].delete_neurons([])
#m.hooks["mlp_pre_out"].delete_neurons([])
m.hooks["pre_decoder"].delete_neurons([])
print(m.hooks["mlp_pre_out"][2])
#m.hooks["pre_decoder"][2].delete_neurons([])


print("here we go")
print(sae_acts.shape)
print(sae_acts)
print(torch.count_nonzero(sae_acts), " non zero")
print(sae_acts[sae_acts != 0])
#exit()
#cripple_data = get_midlayer_data( m, "toxic", 100, collect_sae=True, collect_attn=False, collect_ff=True, calculate_attn=False, calculate_ff=False)
exit()
#print("hooks")
#print(m.hooks)
for layer in range(m.cfg.n_layers):
    #act = m.hooks.collects[f"layer_{layer}_pre_decoder"].activation
    #print(act.shape)
    #print(m.hooks.neuron_sae_encode)
    for sae_hook in sae_hook_points:
        print(f"layer: {layer}")
        print(m.hooks.neuron_sae_encode[f"layer_{layer}_{sae_hook}"].sae_config["d_sae"], " ", layer)
        print(m.hooks.neuron_sae_encode[f"layer_{layer}_{sae_hook}"].sae_config, " ", layer)
        #print(m.hooks.collects[f"layer_{layer}_{sae_hook}"].activation)
        #print(act, " act")
        #act = m.hooks.collects[f"layer_{layer}_pre_decoder"].activation
        #print(m.hooks.collects[f"layer_{layer}_{sae_hook}"])
        #print(m.hooks.collects) #DOESNT WORK. its by layer

        #print(m.hooks["mlp_pre_out"]["collect"]) 
        #print(m.hooks["pre_decoder"]["collect"]) 

        #print(m.hooks["pre_decoder"]) 

#print(len(m.hooks["pre_decoder"]["collect"])) 
#print(m.hooks["pre_decoder"]["collect"])


print("--------------------")
print(m.hooks.collects["layer_0_mlp_pre_out"].activation.shape)
print(m.hooks["mlp_pre_out"]["collect"][0].shape) #if you do this before the .collects call it will be empty. other way around is fine

print(m.hooks.collects["layer_0_pre_decoder"].activation.shape)
print((m.hooks.collects["layer_0_pre_decoder"].activation != 0).sum())
#print sae config
print(m.hooks.neuron_sae_encode["layer_0_pre_decoder"].sae_config["d_in"])
#print model width
print(m.cfg.d_model)
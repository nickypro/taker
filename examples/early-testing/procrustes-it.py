import torch
from tqdm import tqdm
import numpy as np
from scipy.spatial import procrustes
from taker import Model
from taker.texts import prepare
import wandb

def linear_regression_closed_form(X, y):
    device = X.device
    X_with_intercept = torch.cat([torch.ones(X.shape[0], 1, device=device), X], dim=1)
    XTX = torch.matmul(X_with_intercept.T, X_with_intercept)
    XTX_inv = torch.linalg.inv(XTX)
    XTy = torch.matmul(X_with_intercept.T, y)
    beta = torch.matmul(XTX_inv, XTy)
    return beta

def collect_decoder_layer(m, text, layer):
    m.hooks.disable_all_collect_hooks()
    m.hooks.enable_collect_hooks(["post_decoder"], layers=[layer])
    _ = m.get_outputs_embeds(text)
    state = m.hooks["post_decoder"]["collect"][layer]
    return state

def collect_decoder_diff(m, text, layer):
    m.hooks.disable_all_collect_hooks()
    m.hooks.enable_collect_hooks(["pre_decoder", "post_decoder"], layers=[layer])
    _ = m.get_outputs_embeds(text)
    state_in  = m.hooks["pre_decoder"]["collect"][layer]
    state_out = m.hooks["post_decoder"]["collect"][layer]
    return (state_out - state_in)

def process_layer_combination(g2, g7, layer_2b, layer_7b, dataset, label, sample_size):
    states2, states7 = [], []
    count = 0
    with torch.no_grad():
        for data in dataset:
            text = data[label]
            state2 = collect_decoder_diff(g2, text, layer=layer_2b)
            state7 = collect_decoder_diff(g7, text, layer=layer_7b)
            states2.append(state2[0])
            states7.append(state7[0])
            count += state2.shape[1]
            if count > sample_size:
                break

        states2 = torch.cat(states2, dim=0).to(torch.float32)
        states7 = torch.cat(states7, dim=0).to(torch.float32)

        beta = linear_regression_closed_form(states2, states7[:, :2048])
        states2_transformed = torch.matmul(torch.cat([torch.ones(states2.shape[0], 1, device=states2.device), states2], dim=1), beta)

        s2 = states2_transformed.cpu().numpy()
        s7 = states7[:, :2048].cpu().numpy()

        _, _, disparity = procrustes(s2, s7)

        return disparity

def main():
    with torch.no_grad():
        wandb.init(project="gemma-layer-comparison", entity="seperability")

        g2 = Model("google/gemma-2b", dtype="hqq4", device_map="cuda", limit=1000)
        g7 = Model("google/gemma-2b-it", dtype="hqq4", device_map="cuda", limit=1000)

        dataset, label, _ = prepare("pile")
        sample_size = 1e5

        results = {}

        for layer_2b in tqdm(range(g2.cfg.n_layers), desc="2b layers"):
            for layer_7b in range(g7.cfg.n_layers):
                disparity = process_layer_combination(g2, g7, layer_2b, layer_7b, dataset, label, sample_size)
                key = f"2b_layer_{layer_2b}_7b_layer_{layer_7b}"
                results[key] = disparity
                print(key, disparity)

                wandb.log({
                    "2b_layer": layer_2b,
                    "7b_layer": layer_7b,
                    "disparity": disparity
                })

        wandb.finish()
        torch.save(results, "procrustes-results.pt")

if __name__ == "__main__":
    main()

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

def collect_post_attn(m, text, layers):
    m.hooks.disable_all_collect_hooks()
    m.hooks.enable_collect_hooks(["post_attn"], layers=layers)
    _ = m.get_outputs_embeds(text)
    states = m.hooks["post_attn"]["collect"]
    states = [s for s in states if s is not None]
    if not states:
        return None
    return torch.cat(states, dim=-1)

def process_layer_combination(g2, g7, layer_2b, layers_7b, dataset, label, sample_size):
    states2, states7 = [], []
    count = 0
    with torch.no_grad():
        for data in dataset:
            text = data[label]
            state2 = collect_post_attn(g2, text, [layer_2b])
            state7 = collect_post_attn(g7, text, layers_7b)
            if state2 is None or state7 is None:
                continue
            states2.append(state2[0])
            states7.append(state7[0])
            count += state2.shape[1]
            if count > sample_size:
                break

        if not states2 or not states7:
            return None

        states2 = torch.cat(states2, dim=0).to(torch.float32).cuda()
        states7 = torch.cat(states7, dim=0).to(torch.float32).cuda()

        beta = linear_regression_closed_form(states2, states7)
        states2_transformed = torch.matmul(torch.cat([torch.ones(states2.shape[0], 1, device=states2.device), states2], dim=1), beta)

        s2 = states2_transformed.cpu().numpy()
        s7 = states7.cpu().numpy()
        _, _, disparity = procrustes(s2, s7)
        return disparity

def main():
    with torch.no_grad():
        wandb.init(project="gemma-layer-comparison", entity="seperability")
        g2 = Model("google/gemma-2b", dtype="nf4", device_map="cuda", limit=1000)
        g7 = Model("google/gemma-7b", dtype="nf4", device_map="cuda", limit=1000)
        dataset, label, _ = prepare("pile")
        sample_size = 50_000

        window_size = 4
        scale_factor = g7.cfg.n_layers / g2.cfg.n_layers

        results = {}
        for layer_2b in tqdm(range(g2.cfg.n_layers), desc="2b layers"):
            center_layer_7b = int(layer_2b * scale_factor)
            start_layer_7b = max(0, center_layer_7b - window_size // 2)
            end_layer_7b = min(g7.cfg.n_layers, start_layer_7b + window_size)
            layers_7b = list(range(start_layer_7b, end_layer_7b))

            disparity = process_layer_combination(g2, g7, layer_2b, layers_7b, dataset, label, sample_size)
            if disparity is not None:
                key = f"2b_layer_{layer_2b}_7b_layers_{layers_7b}"
                results[key] = disparity
                print(key, disparity)
                wandb.log({
                    "2b_layer": layer_2b,
                    "7b_layers": layers_7b,
                    "disparity": disparity
                })

        wandb.finish()
        torch.save(results, "procrustes-results.pt")

if __name__ == "__main__":
    main()

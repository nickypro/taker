import einops
import torch

from lease import LeaceFitter, LeaceEraser
from taker import Model
m = Model("nickypro/vit-cifar100")

d_model = m.cfg.d_model
d_vocab = 100 # cifar100

def get_midlayer_activations(model, dataset, token_limit=None):
    """Gets the activations of the midlayer ('key' layer) of MLPs for
    each layer, as well as for the pre_out layer of attention for each layer.
    """
    m = model.taker_model
    ff_act   = [fitter for fitter in LeaceFitter(d_model, d_vocab, dtype=m.dtype, device=m.device)]
    attn_act = [fitter for fitter in LeaceFitter(d_model, d_vocab, dtype=m.dtype, device=m.device)]
    tokens_seen = 0

    for batch in tqdm(dataset):
        # Get all necessary activations
        with torch.no_grad():
            t0 = time.time()
            try:
                pixel_values = batch["img"]
                _label       = batch["fine_label"]
                _output      = model(pixel_values=pixel_values)
                text_activations, residual_stream = [0,0,0,0], [0]
                label_vec = torch.zeros(d_vocab, dtype=m.dtype, device=m.device)
                label_vec[_label] = 1
            except ValueError:
                print(f"Could not process an input")
                continue

            attn_act = m.get_attn_pre_out_activations(text_activations=text_activations, reshape=True).detach()
            attn_act = einops.rearrange(attn_act, 'layer token head pos -> layer token (head pos)')

            ff_act = m.get_ff_key_activations(residual_stream=residual_stream).detach()
            ff_act = einops.rearrange( ff_act, 'layer token pos -> layer token pos')

            n_tokens = ff_act.shape()
            labels = einops.repeat(label_vec, 'd_model -> n_tokens d_model', n_tokens=n_tokens)

            for layer in range(m.cfg.n_layers):
                ff_act[layer].update(ff_act, labels)
                attn_act[layer].update(attn_act, labels)


            tokens_seen += len(ff_act)
            if token_limit and tokens_seen > token_limit:
                break

    return {
        "ff":     ff_act,
        "attn": attn_act,
    }
""" Test the get_ff_keys and delete_ff_keys functions. """

from torch import Tensor
import torch
import numpy as np

# pylint: disable=import-error
import pytest
from taker.model_repos import test_model_repos
from taker import Model

class TestDeleteFFKeys:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_ff_key_counting(self, model_repo):
        print("# Running Test: test_ff_key_counting")
        # Initialize model
        opt = Model(model_repo, limit=1000, dtype="fp32")
        n_layers, d_ff = opt.cfg.n_layers, opt.cfg.d_mlp

        # Run text
        text = "for ( var i = 0; i < 10; i++ ) { console.log(i); }"
        input_ids = opt.get_ids(text)
        n_tokens = input_ids.size()[-1]

        # Make a tensor of the expected_size
        expected_size = torch.Size([1, n_layers, n_tokens, d_ff])

        # Run the model with hooks
        with torch.no_grad():
            opt.hooks.disable_all_collect_hooks()
            opt.hooks.enable_collect_hooks(["mlp_pre_out"])
            _ = opt.get_outputs_embeds(input_ids=input_ids)
            ff_keys = opt.collect_recent_mlp_pre_out()

        # Test that result is as desired
        assert ff_keys.shape[1] == n_layers
        assert ff_keys.shape == expected_size

        print("Text size:", ff_keys.size())
        print("Expected :", expected_size)

    @pytest.mark.parametrize("model_repo", test_model_repos)
    @pytest.mark.parametrize("mask_fn", ["delete", "step"])
    def test_delete_ff_keys(self, model_repo, mask_fn):
        print("# Running Test: test_delete_ff_keys")
        # Pre-test initialization
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        opt = Model(model_repo, dtype="fp32", mask_fn=mask_fn,
                    model_device=device, use_accelerator=False)

        # Define input vectors - FIX: Use valid token indices instead of random floats
        removed_indices = [0, 10, 100]
        in_vec = torch.randint(
            0, opt.cfg.d_vocab,
            (5,),  # Sequence length of 5
            device=device
        )

        # Define functions using forward hooks
        def in_to_mid(input_ids, layer):
            with torch.no_grad():
                opt.hooks.disable_all_collect_hooks()
                #opt.hooks.enable_collect_hooks(["mlp_pre_out"], layers=[layer])
                opt.hooks.enable_collect_hooks(["mlp_pre_out"])
                _ = opt.get_outputs_embeds(input_ids=input_ids.unsqueeze(0))
                mid_vec = opt.collect_recent_mlp_pre_out()[:, layer, 0, :]
            return mid_vec  # Removed squeeze(0) to retain batch dimension

        def mid_to_out(mid_vec, layer):
            u = opt.layers[layer]
            with torch.no_grad():
                x = u["activation_fn"](mid_vec)
                return u["mlp.out_proj"](x)

        # Calculate mid layer vectors for testing
        mid_vecs = []
        mid_vecs_removed = []
        for layer in range(opt.cfg.n_layers):
            mid_vec = in_to_mid(in_vec, layer)
            mid_vecs.append(mid_vec)
            mid_vecs_removed.append(mid_vec.clone())
            mid_vecs_removed[-1][:, removed_indices] = 0.0  # Modified to handle batch dimension

        mid_vecs = torch.stack(mid_vecs)
        mid_vecs_removed = torch.stack(mid_vecs_removed)

        # Calculate out layer vectors for testing
        out_vecs = []
        out_vecs_removed = []
        for layer in range(opt.cfg.n_layers):
            out_vecs.append(mid_to_out(mid_vecs[layer], layer))
            out_vecs_removed.append(mid_to_out(mid_vecs_removed[layer], layer))

        out_vecs = torch.stack(out_vecs)
        out_vecs_removed = torch.stack(out_vecs_removed)

        # Define a vector that is changed at certain indices
        removal_tensor = torch.zeros((opt.cfg.n_layers, opt.cfg.d_mlp), dtype=torch.bool)
        for layer in range(opt.cfg.n_layers):
            removal_tensor[layer][removed_indices] = True

        # Here the test starts
        # Pre-test to make sure that outputs are different on each layer
        print('# Running pre-deletion validation')
        for layer in range(opt.cfg.n_layers):
            print('layer ', layer)
            mid_vec_layer = in_to_mid(in_vec, layer)

            assert torch.equal(mid_vec_layer, mid_vecs[layer])
            assert not torch.equal(mid_vec_layer, mid_vecs_removed[layer])

            # out_vec_layer = opt.calculate_ff_out_layer(in_vec, layer)
            # assert torch.equal(out_vec_layer, out_vecs[layer])

        # Run deletions on the layers
        print('# Running deletion')
        opt.hooks.delete_mlp_neurons(removal_tensor)

        # Post-test to make sure deletions work as expected
        print('# Running post-deletion validation')
        # for layer in range(opt.cfg.n_layers):
        #     print('layer', layer)
        #     mid_vec_layer = in_to_mid(in_vec, layer)
        #     if mask_fn == "delete":
        #         assert not torch.equal(mid_vec_layer, mid_vecs[layer])
        #         assert torch.equal(mid_vec_layer, mid_vecs_removed[layer])
        #     if mask_fn == "step":
        #         print(mid_vec_layer[:5,:5])
        #         print(mid_vecs[layer][:5,:5])
        #         assert torch.equal(mid_vec_layer, mid_vecs_removed[layer])
        #         assert not torch.equal(mid_vec_layer, mid_vecs[layer])

            # out_vec_layer = opt.calculate_ff_out_layer(in_vec, layer)
            # assert torch.equal(out_vec_layer, out_vecs_removed[layer])

        # Extra sanity check: make sure that weights correct
        print('# Running sanity check')
        for layer in range(opt.cfg.n_layers):
            print('layer', layer)
            w = opt.layers[layer]["mlp.in_proj"].weight
            removed_weights = (torch.sum(w, dim=-1) == 0.0)
            # if mask_fn == "delete":
            #     assert torch.equal(
            #         removal_tensor[layer].cpu(),            #         removed_weights.cpu()
            #     )
            if mask_fn == "step":
                assert removed_weights.sum() == 0.0

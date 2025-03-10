""" Test the attention operations. """
import torch

# pylint: disable=import-error
import pytest
from taker.model_repos import test_model_repos
from taker import Model

test_model_repos = ["facebook/opt-125m"]

def not_equal(t0, t1):
    return not torch.equal(t0, t1)

def is_zero(t):
    return torch.equal(t, torch.zeros_like(t))

class TestAttnUtils:
    # @pytest.mark.parametrize("model_repo", test_model_repos)
    # def test_svd(self, model_repo):
    #     with torch.no_grad():
    #         layer = 0

    #         opt = Model(model_repo, dtype="fp32",
    #             use_accelerator=False, svd_attn=False)
    #         d_model, device, dtype = opt.cfg.d_model, opt.device, opt.dtype
    #         attn = opt.layers[layer]["attn"]

    #         # Get example input
    #         in_0 = torch.randn([1, 3, d_model], device=device, dtype=dtype)
    #         attn_mask = torch.tensor(
    #             [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]],
    #             device=device, dtype=torch.bool
    #         )

    #         # Get example output
    #         out_0, _, (k_0, v_0) = attn(in_0, attention_mask=attn_mask)
    #         print(out_0)

    #         # Do SVD stuff
    #         opt.svd_attention_layers()

    #         # Check that the output is the same
    #         out_1, _, (k_1, v_1) = attn(in_0, attention_mask=attn_mask)
    #         assert torch.allclose(out_0, out_1, 1e-3, 1e-4)

    @pytest.mark.parametrize("model_repo", test_model_repos)
    @pytest.mark.parametrize("mask_fn", ["step"]) # TODO: reimplement "delete"
    def test_deletion(self, model_repo, mask_fn):
        with torch.no_grad():
            layer, h_index, i_index = 0, 11, 31
            n_tokens = 3

            opt = Model(model_repo, dtype="fp32", mask_fn=mask_fn,
                use_accelerator=False, svd_attn=False)
            d_model, d_head, n_heads = \
                opt.cfg.d_model, opt.cfg.d_head, opt.cfg.n_heads
            device, dtype = opt.device, opt.dtype
            attn = opt.layers[layer]["attn"]

            # Get example input
            in_0 = torch.randn([1, n_tokens, d_model], device=device, dtype=dtype)
            mask = torch.tensor(
                [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]],
                device=device, dtype=torch.bool
            )
            out_biases = torch.stack([
                opt.layers[layer]["attn.b_O"] for _ in range(n_tokens)
            ])

            # Get example output
            out_0, _, (k_0, v_0) = attn(in_0, attention_mask=mask)
            # k_0 is [1, 12, 3, 64]
            v_0_mod = v_0.clone()
            v_0_mod[0, h_index, :, i_index] = 0


            # Delete index
            remove_indices = torch.zeros(
                (n_heads, d_head),
                dtype=torch.bool, device=device
            )
            remove_indices[h_index, i_index] = True
            opt.hooks.delete_attn_neurons(remove_indices, layer)

            # Test behaviour is correct
            out_1, _, (k_1, v_1) = attn(in_0, attention_mask=mask)
            if mask_fn == "delete":
                assert not torch.equal(v_0, v_1)
                assert torch.equal(v_0_mod, v_1)
            if mask_fn == "step":
                assert torch.equal(v_0, v_1)
                assert not torch.equal(v_0_mod, v_1)

            # Delete ALL indices
            remove_indices = torch.ones_like(remove_indices)
            opt.hooks.delete_attn_neurons(remove_indices, layer)

            # Test behaviour is correct
            out_2, _, (k_2, v_2) = attn(in_0, attention_mask=mask)

            print(out_2)

            if mask_fn == "delete":
                assert is_zero(attn.v_proj.weight)
                assert is_zero(attn.v_proj.bias)
                assert is_zero(v_2)
                assert torch.allclose(out_2, out_biases)

        return

    # @pytest.mark.parametrize("model_repo", test_model_repos)
    # def test_inv_out_proj(self, model_repo):
    #     with torch.no_grad():
    #         layer, h_index, i_index = 0, 11, 63
    #         n_tokens = 1

    #         opt = Model(model_repo, dtype="fp32",
    #             use_accelerator=False, svd_attn=False, use_inverse_out=True)
    #         d_model, d_head, n_heads = \
    #             opt.cfg.d_model, opt.cfg.d_head, opt.cfg.n_heads
    #         device, dtype = opt.device, opt.dtype
    #         attn = opt.layers[layer]["attn"]

    #         # Get example input
    #         in_0 = torch.randn([1, n_tokens, d_model], device=device)
    #         mask = torch.tensor(
    #             [[[ [1] ]]],
    #             device=device, dtype=torch.bool
    #         )
    #         out_biases = torch.stack([
    #             opt.layers[layer]["attn.b_O"] for _ in range(n_tokens)
    #         ])

    #         # Get example output
    #         out_0, _, (k_0, v_0) = attn(in_0, attention_mask=mask)
    #         # k_0 is [1, 12, 3, 64]
    #         v_0_mod = v_0.clone()
    #         v_0_mod[0, h_index, :, i_index] = 0


    #         # Test reconstruction ability
    #         # For single token, should be just out(v(in_0))
    #         out_1 = opt.layers[layer]["attn.out_proj"](v_0.flatten())
    #         assert torch.allclose(out_0.flatten(), out_1, 1e-3)

    #         # Inv out should be able to reconstruct v_0
    #         v_1 = opt.layers[layer]["attn.inv_out_proj"](out_1)

    #         # TODO: Check this in more detail.
    #         # Probably noise from zero values in out_proj
    #         i = torch.argmax(v_1.flatten()/v_0.flatten())
    #         print(i, v_1.flatten()[i], v_0.flatten()[i])
    #         # assert torch.allclose(v_0.flatten(), v_1, 1e-1, 1e-3)


    #     return

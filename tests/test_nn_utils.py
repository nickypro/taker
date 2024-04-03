""" Test the nn ufunctions. """

# pylint: disable=import-error
import pytest
import torch
from taker.nn import mlp_svd_two_layer, mlp_delete_columns, mlp_delete_rows, \
    mlp_adjust_biases, InverseLinear, NeuronMask, NeuronPostBias, NeuronActAdd

def not_equal(t0, t1):
    return not torch.equal(t0, t1)

class TestNNUtils:
    def test_deletions(self):
        with torch.no_grad():
            _dtype = torch.float32
            d_in, d_out, deleted_index = 10, 20, 1
            W = torch.ones([d_out, d_in], dtype=_dtype)
            B = torch.ones([d_out], dtype=_dtype)

            FF = torch.nn.Linear(d_in, d_out, bias=True, dtype=_dtype)
            FF.load_state_dict({'weight': W.clone(), 'bias': B.clone()})

            # pre-test
            in_0 = torch.randn([d_in], dtype=_dtype)
            print(in_0.shape)
            in_1 = in_0.clone()
            in_1[deleted_index] = 0

            # Pre-comparison
            out_0, out_1 = FF(in_0), FF(in_1)
            out_0_index_removed = out_0.clone()
            out_0_index_removed[deleted_index] = 0
            out_0_index_removed_bias = out_0_index_removed.clone()
            out_0_index_removed_bias[deleted_index] = 1.0
            assert not_equal(out_0, out_1)
            assert not_equal(out_0_index_removed, out_0)

            # Delete columns
            deletion_indices = torch.zeros(d_in, dtype=torch.bool)
            deletion_indices[deleted_index] = True
            FF = mlp_delete_columns(FF, deletion_indices)

            assert not_equal(out_0_index_removed, FF(in_0))
            assert not_equal(out_0, FF(in_0))
            assert torch.equal(out_1, FF(in_1))

            # Restore original weights
            FF.load_state_dict({'weight': W.clone(), 'bias': B.clone()})

            # Delete rows without biases
            deletion_indices = torch.zeros(d_out, dtype=torch.bool)
            deletion_indices[deleted_index] = True
            FF = mlp_delete_rows(FF, deletion_indices, False)

            assert torch.equal(out_0_index_removed_bias, FF(in_0))
            assert not_equal(out_0_index_removed, FF(in_0))
            assert not_equal(out_0, FF(in_0))
            assert not_equal(out_1, FF(in_1))

            # Restore original weights
            FF.load_state_dict({'weight': W.clone(), 'bias': B.clone()})

            # Delete rows and biases
            deletion_indices = torch.zeros(d_out, dtype=torch.bool)
            deletion_indices[deleted_index] = True
            FF = mlp_delete_rows(FF, deletion_indices, True)

            assert not_equal(out_0_index_removed_bias, FF(in_0))
            assert torch.equal(out_0_index_removed, FF(in_0))
            assert not_equal(out_0, FF(in_0))
            assert not_equal(out_1, FF(in_1))

        return

    def test_svd(self):
        with torch.no_grad():
            _dtype = torch.float32
            d_in, d_mid, d_out = 20, 20, 20

            # Set up intial weights
            W_in  = torch.randn([d_mid, d_in ], dtype=_dtype)
            W_out = torch.randn([d_out, d_mid], dtype=_dtype)
            b_in  = torch.randn([d_mid], dtype=_dtype)
            b_out = torch.randn([d_out], dtype=_dtype)

            # Set up FF layers
            FF_in = torch.nn.Linear(d_in, d_mid, bias=True, dtype=_dtype)
            FF_out = torch.nn.Linear(d_mid, d_out, bias=True, dtype=_dtype)
            FF_in.load_state_dict({'weight': W_in.clone(), 'bias': b_in.clone()})
            FF_out.load_state_dict({'weight': W_out.clone(), 'bias': b_out.clone()})

            # Example Input
            in_0  = torch.randn([d_in], dtype=_dtype)
            out_0 = FF_out(FF_in(in_0))


            # SVD, returning biases to middle
            mlp_svd_two_layer(FF_in, FF_out, d_mid)

            # Check matrices are correct
            W_full = torch.matmul(W_out, W_in)
            W_svd  = torch.matmul(FF_out.weight, FF_in.weight)
            assert torch.allclose(W_full, W_svd, 1e-3, 1e-6)

            # Check things were changed
            assert not_equal(FF_in.weight,  W_in)
            assert not_equal(FF_in.bias,  b_in)
            assert not_equal(FF_out.weight, W_out)
            assert torch.equal(FF_out.bias, b_out)

            # Example Output
            assert torch.allclose(out_0, FF_out(FF_in(in_0)), 1e-3)

    def test_inverse_linear(self):
        ff = torch.nn.Linear(20, 20)

        # Test input
        in_0  = torch.randn(20)
        out_0 = ff(in_0)

        # Test inverse
        ff_inv = InverseLinear(ff)
        in_1 = ff_inv(out_0)
        assert torch.allclose(in_0, in_1, 1e-3)

        # Test alternative mode inverse
        ff_inv = InverseLinear(original_weights=ff.weight,
                               original_biases=ff.bias)
        in_2 = ff_inv(out_0)
        assert torch.allclose(in_0, in_2, 1e-3)

    def test_mask(self):
        dim = 32
        mask = NeuronMask([dim], act_fn="step")

        # Check first that the mask does nothing by default
        v = torch.randn([dim])
        expected_v = v.clone()
        assert torch.allclose(mask(v), expected_v)

        # Check next that the mask can mask
        neurons = torch.ones([dim])
        neurons[:10] = 0
        mask.set_mask(neurons)

        expected_v = v.clone()
        expected_v[:10] = 0
        assert torch.allclose(mask(v), expected_v)

        # Check that the mask offsets masked neurons
        offsets = torch.randn([dim])
        mask.set_offset(offsets)

        expected_v = v.clone()
        expected_v[:10] = offsets[:10]
        assert torch.allclose(mask(v), expected_v)

    def test_actadd(self):
        # Instantiate the module
        mod = NeuronActAdd("cpu", torch.float32, autoreset=False)
        dim = 16

        # Verify the original shape
        print("Original shape:", mod.param.shape)
        assert len(mod.param)  == 0
        assert mod.max_tokens  == 0
        assert mod.tokens_seen == 0
        assert mod.autoreset   == False

        # Check that the ipnut is unchanged
        v_in  = torch.zeros([5, dim])
        v_out = mod(v_in)
        assert torch.allclose(v_in, v_out)

        # initialise the parameters
        n_changed = 3
        rand_bias = torch.randn([n_changed, dim])
        mod.set_actadd(rand_bias.clone())
        assert mod.param.shape == torch.Size([n_changed, dim])
        assert mod.max_tokens == n_changed
        assert mod.tokens_seen == 0
        # tokens_seen == 0

        # test that it works
        v_in    = torch.zeros([1, dim])
        v_out_0 = mod(v_in.clone())
        assert torch.allclose(v_out_0, v_in + rand_bias[0])
        assert mod.tokens_seen == 1

        # test that rolling tokens works
        v_out_1 = mod(v_in.clone())
        assert torch.allclose(v_out_1, v_in + rand_bias[1])
        assert mod.tokens_seen == 2

        # test that operations not done in place
        v_out_2 = mod(v_in)
        assert torch.allclose(v_out_2, v_in + rand_bias[2])
        assert torch.sum(v_in) == 0.0
        assert mod.tokens_seen == 3

        # test that it runs out of things to change
        v_out_3 = mod(v_in)
        assert torch.allclose(v_out_3, v_in)
        assert torch.sum(v_out_3) == 0.0
        assert mod.tokens_seen == 3

        # Test that the reset works
        mod.reset()
        v_out_0_1 = mod(v_in.clone())
        assert torch.allclose(v_out_0, v_out_0_1)
        assert mod.tokens_seen == 1

        # Test that it works for multiple inputs at the same time and keeps track
        v_in = torch.zeros([10, dim])
        v_out_10 = mod(v_in)
        i = n_changed-1
        assert torch.allclose(v_out_10[:i], v_in[:i] + rand_bias[1:])
        assert torch.allclose(v_out_10[i:], v_in[i:])

        # Test that the autoreset works
        mod.autoreset = True
        assert mod.tokens_seen == n_changed
        v_in = torch.zeros([10, dim])
        v_out_10 = mod(v_in)
        assert torch.allclose(v_out_10[:3], v_in[:3] + rand_bias)
        assert torch.allclose(v_out_10[3:], v_in[3:])

        # also that autoreset doesn't activate for single vector inputs
        v_in = torch.zeros([1, dim])
        v_out = mod(v_in)
        assert torch.allclose(v_out, v_in)






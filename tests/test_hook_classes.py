""" Test the hook class implementations. """

# pylint: disable=import-error
import pytest
import torch
from taker.hooks import NeuronMask, NeuronActAdd, NeuronReplace, \
    NeuronOffset, NeuronPostBias, NeuronSave, NeuronFunctionList

def not_equal(t0, t1):
    return not torch.equal(t0, t1)

class TestHookClasses:
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

    def test_neuron_replace(self):
        with torch.no_grad():
            repl = NeuronReplace(device="cpu", dtype=torch.float32)
            N_TOKEN, DIM = 3, 32
            # test that it does nothing by default
            x  = torch.randn([1, N_TOKEN, DIM])
            y1 = repl(x.clone())
            assert torch.equal(x, y1)

            # test that if replaced token 1 -> [0,...,0] that it works
            repl.add_token(1, torch.zeros([DIM]))
            y2 = repl(x.clone())
            assert not_equal(x, y2)
            assert y2[0, 1].sum() == 0
            assert torch.equal(x[0,0], y2[0,0])
            assert torch.equal(x[0,2], y2[0,2])

            # test that if replaced token 2 -> [1, ..., 1] that it works
            # in additon the the previous thing
            repl.add_token(2, torch.ones([DIM]))
            y3 = repl(x.clone())
            assert not_equal(x, y3)
            assert y3[0, 2].sum() == DIM
            assert torch.equal(x[0,0], y3[0,0])
            assert torch.equal(y2[0,1], y3[0,1])

            # test that reseting it also works fine
            repl.reset()
            y4 = repl(x.clone())
            assert torch.equal(x, y4)

            # test that if replaced token 1 -> [[1], ..., [1]] also (autoshaping)
            repl.add_token(1, torch.ones([DIM, 1]))
            y5 = repl(x.clone())
            assert not_equal(x, y5)
            assert y5[0, 1].sum() == DIM
            assert torch.equal(x[0,0], y5[0,0])
            assert torch.equal(x[0,2], y5[0,2])


    def test_neuron_offset(self):
        with torch.no_grad():
            x  = torch.randn([1, 3, 32])
            repl = NeuronOffset(shape=x.shape)

            # test that it does nothing by default
            y1 = repl(x.clone())
            assert torch.equal(x, y1)

            # TODO: add more tests here

    def test_actadd(self):
        # Instantiate the module
        mod = NeuronActAdd("cpu", torch.float32, autorestart=False)
        dim = 16

        # Verify the original shape
        print("Original shape:", mod.param.shape)
        assert len(mod.param)  == 0
        assert mod.max_tokens  == 0
        assert mod.tokens_seen == 0
        assert mod.autorestart == False

        # Check that the ipnut is unchanged
        v_in  = torch.zeros([1, 5, dim])
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
        v_in    = torch.zeros([1, 1, dim])
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

        # Test that the restart works
        mod.restart()
        v_out_0_1 = mod(v_in.clone())
        assert torch.allclose(v_out_0, v_out_0_1)
        assert mod.tokens_seen == 1

        # Test that it works for multiple inputs at the same time and keeps track
        v_in = torch.zeros([1, 10, dim])
        v_out_10 = mod(v_in)
        i = n_changed-1
        assert torch.allclose(v_out_10[0,:i], v_in[0,:i] + rand_bias[1:])
        assert torch.allclose(v_out_10[0,i:], v_in[0,i:])

        # Test that the autorestart works
        mod.autorestart = True
        assert mod.tokens_seen == n_changed
        v_in = torch.zeros([1, 10, dim])
        v_out_10 = mod(v_in)
        assert torch.allclose(v_out_10[0,:3], v_in[0,:3] + rand_bias)
        assert torch.allclose(v_out_10[0,3:], v_in[0,3:])

        # also that autorestart doesn't activate for single vector inputs
        v_in = torch.zeros([1, 1, dim])
        v_out = mod(v_in)
        assert torch.allclose(v_out, v_in)


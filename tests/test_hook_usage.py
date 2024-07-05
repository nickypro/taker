
""" Test the hook class implementations. """

# pylint: disable=import-error
import pytest
import torch
from taker.hooks import NeuronMask, NeuronActAdd, NeuronReplace, NeuronOffset

def not_equal(t0, t1):
    return not torch.equal(t0, t1)

class TestHookUsage:
    def test_neuron_replace_usage(self):
        import torch
        from taker import Model
        m = Model("nickypro/tinyllama-15m", limit=1000)

        m.hooks_raw.neuron_replace["layer_0_mlp_pre_out"].reset()
        res1 = m.get_residual_stream("Some text here")[:,0:4,0:3,0:5]

        m.hooks_raw.neuron_replace["layer_0_mlp_pre_out"].add_token(
        #    1, torch.zeros([m.cfg.n_heads, m.cfg.d_head])
            1, torch.zeros([m.cfg.d_mlp])
        )
        res2 = m.get_residual_stream("Some text here")[:,0:4,0:3,0:5]

        assert torch.equal(res1[:,0:2], res2[:,0:2]) # layers [pre_attn0, pre_mlp 0] unaffected
        assert torch.equal(res1[:,:,0], res2[:,:,0]) # token 0 unchanged
        assert not_equal(  res1[0,2:4], res2[0,2:4])

        # TODO: add more tests maybe??

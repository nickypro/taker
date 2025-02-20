import pytest
import torch
from taker import Model
from einops import reduce

class TestModelActivations:
    def __init__(self):
        self.model = Model(limit=1000)
        self.get_activations = lambda: self.model.get_midlayer_activations("some example text here")["mlp"]

    def test_offset_by_mean(self, layer_index=0):
        # Reset hooks
        self.model.hooks.reset()

        # Get original activations
        acts = self.get_activations()

        # Compute mean activations
        mean_acts = reduce(acts, "batch layer token dim -> layer dim", "mean")

        # Set offsets to negative mean
        offsets = -mean_acts
        self.model.hooks["mlp_pre_out"].set_offsets(offsets)

        # Get activations after offset
        new_acts = self.get_activations()

        # Assert that activations have changed
        assert not torch.allclose(acts, new_acts, atol=1e-6)
        return new_acts

    def test_offset_by_constant(self, layer_index=0):
        # Reset hooks
        self.model.hooks.reset()

        # Get original activations
        acts = self.get_activations()

        # Set constant offset
        offsets = 3.0 * torch.ones([self.model.cfg.n_layers, self.model.cfg.d_mlp])
        self.model.hooks["mlp_pre_out"].set_offsets(offsets)

        # Get activations after offset
        new_acts = self.get_activations()

        # Assert that activations have changed
        assert not torch.allclose(acts, new_acts, atol=1e-6)
        return new_acts

@pytest.mark.parametrize("layer_index", [0, 4])
def test_model_activation_offset(layer_index):
    tester = TestModelActivations()
    # Test mean offset
    mean_offset_acts = tester.test_offset_by_mean(layer_index)
    # Test constant offset
    const_offset_acts = tester.test_offset_by_constant(layer_index)
    # You can add more assertions or visualizations here if needed
""" Test the evaluate_all function. """

# pylint: disable=import-error
import pytest
from taker.model_repos import test_model_repos
from taker import Model
import torch

class TestHookHandles:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_hook_handle_replacement(self, model_repo):
        m = Model(model_repo, limit=1000, dtype="fp32")
        text = "some text"

        # get a reference unchanged input
        res_0 = m.get_residual_stream(text)

        # modify the masks
        mask0 = m.masks["attn_pre_out"][0]
        mask0.delete_neurons(
            keep_indices=torch.zeros([m.cfg.d_model])
        )

        # get modified input
        res_1 = m.get_residual_stream(text)

        assert not torch.allclose(res_0, res_1)

        # register new masks, check that they are replaced
        m.register_masks()
        res_2 = m.get_residual_stream(text)

        assert torch.allclose(res_0, res_2)





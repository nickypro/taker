
# pylint: disable=import-error
import pytest
from taker.model_repos import test_model_repos
from taker.data_classes import ActivationSummary
from taker import Model
from taker.activations import get_midlayer_data
from taker.eval import evaluate_all

class TestDeleteFFKeys:
    @pytest.mark.parametrize("model_repo", test_model_repos)
    def test_delete_ff_and_evaluate(self, model_repo):
        print("# Running test: test_delete_ff_and_evaluate")
        # Load model and evaluate
        opt = Model(model_repo, limit=1000, dtype="fp32")
        print(" - Initial Evaluation...")
        eval_before = evaluate_all( opt, 1e3 )

        # Get crossover data
        print(" - Initial Evaluation...")

        pile_data: ActivationSummary = \
            get_midlayer_data(opt, 'pile', sample_size=1e3, calculate_attn=False).mlp
        code_data: ActivationSummary = \
            get_midlayer_data(opt, 'code', sample_size=1e3, calculate_attn=False).mlp
        pile_count = pile_data.orig.pos_count
        code_count = code_data.orig.pos_count

        # Remove attention heads over crossover threshold (very low threshold here)
        removals = code_count > pile_count
        print("# Deleting FF Keys...")

        opt.hooks.delete_mlp_neurons(removals)

        # Make sure attention heads were deleted
        print("# Final Evaluation...")
        eval_after = evaluate_all( opt, 1e3 )
        eval_keys = eval_before.keys()
        for key in eval_keys:
            if key == "token_count":
                assert eval_before[key] == eval_after[key]
                continue
            assert eval_before[key] != eval_after[key]

        print("# Test Passed")

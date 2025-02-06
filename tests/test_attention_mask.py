import unittest
import torch
from taker import Model

class TestModelMasking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model()
        cls.test_text = 'Bob then said "my name is'

    def setUp(self):
        # Initialize base mask for each test
        self.mask = torch.zeros([1, 1, 8, 8], device=self.model.device)
        self.mask[...] = -torch.inf

    def create_triangular_mask(self):
        """Creates the triangular attention mask pattern"""
        for i in range(8):
            self.mask[..., i:8, 0:i+1] = 0

    def get_model_output(self, mask):
        """Helper to get model output given a mask"""
        logits = self.model.get_logits(self.test_text, attention_mask=mask)
        output_ids = logits.argmax(dim=-1)[0].tolist()
        text = self.model.tokenizer.decode(output_ids)
        return text.split()[-1]  # Return just the last word

    def test_name_changes(self):
        """Test that masking changes the final name"""
        # Test initial mask (should output Bob)
        self.create_triangular_mask()
        first_output = self.get_model_output(self.mask)
        self.assertEqual(first_output, 'Bob')

        # Test modified mask (should output something other than Bob)
        self.mask[..., 7, 1] = -torch.inf
        second_output = self.get_model_output(self.mask)
        self.assertNotEqual(second_output, 'Bob')

if __name__ == '__main__':
    unittest.main()
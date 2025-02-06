import unittest
import torch
from transformers import AutoTokenizer
from peft import LoraConfig
from taker import Model

class TestPeftTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Model(model_repo="nickypro/tinyllama-15m")
        cls.peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
        cls.model.init_peft(cls.peft_config)
        cls.tokenizer = cls.model.tokenizer

        # Set padding token
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

    def count_trainable_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def test_1_peft_initialization(self):
        trainable_params = self.count_trainable_parameters(self.model.peft_predictor)
        self.model.peft_predictor.print_trainable_parameters()
        reported_trainable_params = 55296  # Expected value based on previous output
        self.assertEqual(trainable_params, reported_trainable_params,
                         "Mismatch in trainable parameters count")
    def generate_text(self, predictor, input_text, max_attempts=3):
        for _ in range(max_attempts):
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.model.device)
            attention_mask = inputs.attention_mask.to(self.model.device)

            with torch.no_grad():
                output = predictor.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=True,
                    temperature=0.7
                )
            # Remove the input from the output before decoding
            generated_ids = output[0][input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if len(generated_text) > 10:
                return generated_text
        raise ValueError(f"Failed to generate text longer than 10 characters after {max_attempts} attempts")

    def check_a_percentage(self, text):
        text = text.lower()
        non_space_chars = [c for c in text if c != ' ']
        a_count = non_space_chars.count('a')
        return a_count / len(non_space_chars) if non_space_chars else 0

    def test_2_pre_training_output(self):
        input_text = "Generate the letter a"
        orig_output = self.generate_text(self.model.orig_predictor, input_text)
        peft_output = self.generate_text(self.model.peft_predictor, input_text)
        current_output = self.generate_text(self.model.predictor, input_text)

        self.assertLess(self.check_a_percentage(orig_output), 0.5,
                        f"Original model should not generate mostly 'a's before training, but generated '{orig_output}")
        self.assertLess(self.check_a_percentage(peft_output), 0.5,
                        f"PEFT model should not generate mostly 'a's before training, but generated {peft_output}")
        self.assertLess(self.check_a_percentage(current_output), 0.5,
                        f"Current model should not generate mostly 'a's before training, but generated {current_output}")

    def train_peft_model(self, num_epochs=5, learning_rate=1e-3):
        input_text = "Generate the letter a"
        target_text = "a " * 20
        full_text = input_text + target_text

        inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore the last token when computing loss

        optimizer = torch.optim.AdamW(self.model.peft_predictor.parameters(), lr=learning_rate)

        self.model.peft_predictor.train()
        for epoch in range(num_epochs):
            outputs = self.model.peft_predictor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.model.peft_predictor.eval()

    def test_3_post_training_output(self):
        self.train_peft_model()

        input_text = "Generate the letter a"
        orig_output = self.generate_text(self.model.orig_predictor, input_text)
        peft_output = self.generate_text(self.model.peft_predictor, input_text)
        current_output = self.generate_text(self.model.predictor, input_text)

        self.assertGreater(self.check_a_percentage(orig_output), 0.5,
                           f"Original model should generate mostly 'a's after training, but generated '{orig_output}'")
        self.assertGreater(self.check_a_percentage(peft_output), 0.5,
                           f"PEFT model should generate mostly 'a's after training, but generated '{peft_output}'")
        self.assertGreater(self.check_a_percentage(current_output), 0.5,
                           f"Current model should generate mostly 'a's after training, but generated '{current_output}'")

    def test_4_post_training_parameter_count(self):
        trainable_params = self.count_trainable_parameters(self.model.peft_predictor)
        self.model.peft_predictor.print_trainable_parameters()
        reported_trainable_params = 55296  # This should remain the same after training
        self.assertEqual(trainable_params, reported_trainable_params,
                         "Trainable parameters count should not change after training")

if __name__ == '__main__':
    unittest.main()
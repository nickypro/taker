import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from taker import Model
from peft import LoraConfig

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_trainable_parameters(model):
    # Count trainable parameters
    trainable_params = count_trainable_parameters(model.peft_predictor)

    # Capture the output of print_trainable_parameters
    import io
    import sys

    captured_output = io.StringIO()
    sys.stdout = captured_output
    model.peft_predictor.print_trainable_parameters()
    sys.stdout = sys.__stdout__
    # Extract the number from the captured output
    output_lines = captured_output.getvalue().split('\n')
    for line in output_lines:
        if "trainable params:" in line:
            reported_trainable_params = int(line.split(':')[1].strip().split()[0].replace(',', ''))
            break

    print(f"Counted trainable parameters: {trainable_params}")
    print(f"Reported trainable parameters: {reported_trainable_params}")

    assert trainable_params == reported_trainable_params, "Mismatch in trainable parameters count!"
    print("Trainable parameters count matches the reported count.")


def train_peft_model(model, num_epochs=5, learning_rate=1e-3):
    # Prepare the training data
    input_text = "Generate the letter a: "
    target_text = "a " * 20  # 20 'a' tokens
    full_text = input_text + target_text

    tokenizer = model.tokenizer

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the full text
    inputs = tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Create labels by shifting the input_ids to the right
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # Ignore the last token when computing loss

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.peft_predictor.parameters(), lr=learning_rate)
    total_steps = num_epochs * 10  # 10 steps per epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    model.peft_predictor.train()
    for epoch in range(num_epochs):
        for _ in range(10):  # 10 steps per epoch
            outputs = model.peft_predictor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    model.peft_predictor.eval()


def test_model_outputs(model, input_text="Generate the letter a: "):
    tokenizer = model.tokenizer

    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    def generate_text(predictor):
        with torch.no_grad():
            output = predictor.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,  # Increased from 20 to 50
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.7
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    orig_text = generate_text(model.orig_predictor)
    peft_text = generate_text(model.peft_predictor)
    current_text = generate_text(model.predictor)

    print({"Original Output": orig_text})
    print({"PEFT Output": peft_text})
    print({"Current Output": current_text})

# Usage
model = Model(model_repo="nickypro/tinyllama-15m")  # Replace with your actual model initialization

# Initialize PEFT
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model.init_peft(peft_config)

# Verify trainable parameters
print("Verifying trainable parameters:")
verify_trainable_parameters(model)

# Test before training
print("\nBefore Training:")
test_model_outputs(model)

# Train the model
train_peft_model(model)

# Test after training
print("\nAfter Training:")
test_model_outputs(model)

# Verify trainable parameters again (should be the same)
print("\nVerifying trainable parameters after training:")
verify_trainable_parameters(model)
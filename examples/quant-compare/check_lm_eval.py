import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from taker import Model

# m = Model("mistralai/Mistral-7B-instruct-v0.2", dtype="nf4", collect_midlayers=False)
# m = Model()
m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="int4", add_hooks=False, model_device="cuda")
#m = Model("QuantFactory/Meta-Llama-3-8B-Instruct-GGUF", model_file="Meta-Llama-3-8B-Instruct.Q8_0.gguf", add_hooks=False)
#m = Model("SanctumAI/Meta-Llama-3-8B-Instruct-GGUF", model_file="meta-llama-3-8b-instruct.Q8_0.gguf", add_hooks=False)
# m = Model("astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit", dtype="fp16", add_hooks=False)
#m.remove_all_hooks()
# m = Model("nickypro/tinyllama-15m-rand")

my_model = HFLM(
    pretrained=m.predictor,
    tokenizer=m.tokenizer,
)

# task_list = ["arc_easy", "arc_challenge", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande"]
# task_list = ["boolq"]
task_list = ["mmlu"]

# Run evaluations
# results = run_evaluations(model=my_model, tasks=tasks)
results = evaluator.simple_evaluate(
    model=my_model,
    tasks=task_list,
)

# Display or process the results
print(results.keys())
print(results["results"])

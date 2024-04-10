import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from taker import Model

# m = Model("mistralai/Mistral-7B-instruct-v0.2", dtype="nf4", collect_midlayers=False)
m = Model()
# m = Model("nickypro/tinyllama-15m-rand")

my_model = HFLM(
    pretrained=m.predictor,
    tokenizer=m.tokenizer,
)

# task_list = ["arc_easy", "arc_challenge", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande"]
task_list = ["boolq"]

# Run evaluations
# results = run_evaluations(model=my_model, tasks=tasks)
results = evaluator.simple_evaluate(
    model=my_model,
    tasks=task_list,
)

# Display or process the results
print(results.keys())
print(results["results"])
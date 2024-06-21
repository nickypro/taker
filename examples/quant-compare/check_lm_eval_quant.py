import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from taker import Model

# m = Model("mistralai/Mistral-7B-instruct-v0.2", dtype="nf4", collect_midlayers=False)
# m = Model()
#m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="nf4", collect_midlayers=False)
# m = Model("TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ", dtype="fp16", add_hooks=False)
#m = Model("kaitchup/Llama-3-8b-awq-4bit", dtype="fp16", add_hooks=False)
#m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="hqq4", add_hooks=False, model_device="cuda")
#m.remove_all_hooks()
# m = Model("nickypro/tinyllama-15m-rand")

import argparse

parser = argparse.ArgumentParser(description="Run a model with a specified task")
parser.add_argument("--model", type=str, required=True, help="Model to run")
parser.add_argument("--task", type=str, required=True, help="Task to perform")

args = parser.parse_args()
k = args.model
task = args.task


m = None
if k == 'bfp16':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="bfp16", collect_midlayers=False)
if k == 'fp16':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="fp16", collect_midlayers=False)
if k == '8bit-hqq':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="hqq8", add_hooks=False, model_device="cuda")
if k == '8bit-bnb':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="int8", collect_midlayers=False)
if k == '8bit-gptq':
    m = Model("astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit", dtype="fp16", add_hooks=False)
if k == '4bit-hqq':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="hqq4", add_hooks=False, model_device="cuda")
if k == '4bit-gptq':
    m = Model("TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ", dtype="fp16", add_hooks=False)
if k == '4bit-nf':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="nf4", collect_midlayers=False)
if k == '4bit-awq':
    m = Model("kaitchup/Llama-3-8b-awq-4bit", dtype="fp16", add_hooks=False)

if m is None:
    print("invalid k")

my_model = HFLM(
    pretrained=m.predictor,
    tokenizer=m.tokenizer,
)

# task_list = ["arc_easy", "arc_challenge", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande"]
# task_list = ["boolq"]
task_list = [task]

# Run evaluations
# results = run_evaluations(model=my_model, tasks=tasks)
results = evaluator.simple_evaluate(
    model=my_model,
    tasks=task_list,
)

# Display or process the results
print(results.keys())
print(results["results"])

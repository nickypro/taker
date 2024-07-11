import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from taker import Model
from time import time

# fix errors due to compiled model
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch.set_float32_matmul_precision('high')

import argparse

parser = argparse.ArgumentParser(description="Run a model with a specified task")
parser.add_argument("--model", type=str, required=True, help="Model to run")
parser.add_argument("--task", type=str, required=True, help="Task to perform")
parser.add_argument("--limit", type=int, required=False, help="Limit sample size", default=None)

args = parser.parse_args()
k = args.model
task = args.task


m = None
if k == 'bfp16':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="bfp16", add_hooks=False)
if k == 'fp16':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="fp16", add_hooks=False)
if k == '8bit-hqq':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="hqq8", add_hooks=False, device_map="cuda")
if k == '8bit-bnb':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="int8", add_hooks=False)
if k == '8bit-gptq':
    m = Model("astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit", dtype="fp16", add_hooks=False)
if k == '4bit-hqq':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="hqq4", add_hooks=False, device_map="cuda")
if k == '4bit-hqq-1':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="hqq4_1", add_hooks=False, device_map="cuda")
if k == '4bit-gptq':
    m = Model("TechxGenus/Meta-Llama-3-8B-Instruct-GPTQ", dtype="fp16", add_hooks=False)
if k == '4bit-bnb':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="int4", add_hooks=False)
if k == '4bit-nf':
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="nf4", add_hooks=False)
if k == '4bit-awq':
    m = Model("kaitchup/Llama-3-8b-awq-4bit", dtype="fp16", add_hooks=False)
if k == "3bit-hqq":
    m = Model("NousResearch/Meta-Llama-3-8B-Instruct", dtype="hqq3", add_hooks=False, device_map="cuda")


if m is None:
    print("invalid k")

my_model = HFLM(
    pretrained=m.predictor,
    tokenizer=m.tokenizer,
)

# task_list = ["arc_easy", "arc_challenge", "boolq", "hellaswag", "openbookqa", "piqa", "winogrande"]
task_list = [task]

# Run evaluations
t0 = time()
results = evaluator.simple_evaluate(
    model=my_model,
    tasks=task_list,
    limit=args.limit,
    batch_size=4,
)
t = time() - t0

# Display or process the results
final_results = {"time": t, **results["results"]}
print(final_results)

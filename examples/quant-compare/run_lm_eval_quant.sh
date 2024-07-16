#!/bin/bash

models=('4bit-hqq' '8bit-hqq' '8bit-bnb' '4bit-nf' '4bit-bnb' '4bit-awq' '8bit-gptq'  '4bit-gptq')

tasks=('minerva_math_algebra')

for task in "${tasks[@]}"
do
for model in "${models[@]}"
do
    poetry run python check_lm_eval_quant.py --model "$model" --task "$task" --limit 100
done
done


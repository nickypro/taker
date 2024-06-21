#!/bin/bash

models=('4bit-hqq' '8bit-hqq') # '8bit-bnb' '8bit-gptq'  '4bit-gptq' '4bit-nf' '4bit-awq')

tasks=('gpqa' 'pile_10k' 'minerva_math')

for task in "${tasks[@]}"
do
for model in "${models[@]}"
do
    poetry run python check_lm_eval_quant.py --model "$model" --task "$task"
done
done


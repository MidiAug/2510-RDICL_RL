#!/bin/bash
set -x
WORK_SPACE=$(os.getenv("WORK_SPACE"))
cd ${WORK_SPACE}
DEMO_COUNT=16

export CUDA_VISIBLE_DEVICES=2
# 禁用 tqdm 进度条，避免重定向到日志文件时产生混乱的输出
export TQDM_DISABLE=1
# 确保 Python 输出不被缓冲，立即刷新到日志文件
export PYTHONUNBUFFERED=1

WANDB_MODE=offline accelerate launch --config_file ${WORK_SPACE}/script/stf/config.yaml ${WORK_SPACE}/script/stf/sft.py \
    --data_fpath ${WORK_SPACE}/data/demo${DEMO_COUNT}.jsonl \
    --model_fpath Qwen/Qwen3-1.7B-Base \
    --output_dir ${WORK_SPACE}/ckpts/stf-demo${DEMO_COUNT} \
    --response_template "<|im_start|>assistant\n<think>\n\n</think>\n\n" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --lr_scheduler_type constant \
    --bf16 true \
    --max_seq_length 32768 \
    --input_column question \
    --output_column solution \
    --save_only_model \
    --save_strategy epoch \
    --logging_first_step \
    --logging_steps 1

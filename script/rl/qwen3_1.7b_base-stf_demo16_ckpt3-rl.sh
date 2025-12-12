# !/bin/bash
set -x

WORK_SPACE=$(os.getenv("WORK_SPACE"))
VERL_HOME=$(os.getenv("VERL_HOME"))
cd ${VERL_HOME}

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export GPU_PER_NODE_COUNT=4
export CUDA_VISIBLE_DEVICES=1,2,3,4
export AZUREML_NODE_COUNT=1
export HF_ENDPOINT=https://hf-mirror.com


math_train_path=${WORK_SPACE}/data/train.parquet
math_test_path=${WORK_SPACE}/data/test.parquet
train_files="['$math_train_path']"
test_files="['$math_test_path']"
reward_fn_path=${WORK_SPACE}/script/rl/verl_math_verify.py

model_path=${WORK_SPACE}/ckpts/stf-demo16/checkpoint-3

project_name='stf+rl'
experiment_name='qwen3_1.7b_base-stf_demo16_ckpt3-rl'
default_local_dir="./${project_name}/${experiment_name}"

WANDB_MODE=online python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    custom_reward_function.path=$reward_fn_path \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=$default_local_dir \
    trainer.n_gpus_per_node=$GPU_PER_NODE_COUNT \
    trainer.nnodes=$AZUREML_NODE_COUNT \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.resume_mode=auto
cd /root/users/lcq/RDICL_RL/simpleRL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

WANDB_MODE=offline bash eval_math_nodes.sh \
    --run_name /root/users/lcq/RDICL_RL/verl/stf+rl/qwen3_1.7b_base-stf_ckpt3-rl \
    --init_model /root/models/shares/Qwen/Qwen3-1.7B-Base \
    --template qwen-boxed \
    --tp_size 1 \

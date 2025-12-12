#!/bin/bash

# ==================== 配置区域 ====================

# 1. 设置您的实验根目录 (即包含 'global_step_xx' 文件夹的那个目录)
BASE_EXPERIMENT_DIR="/root/users/lcq/ICL/verl/qwen3_1.7b_base_math35_ext"

# 2. 指定您想要转换的一个或多个检查点步数
#    - 如果只想转换一个, 就像这样: STEPS_TO_CONVERT=(10)
#    - 如果要转换多个, 用空格隔开: STEPS_TO_CONVERT=(10 20 30)
STEPS_TO_CONVERT=(10 20 30 40 50 60 70 80 90 100 110 120)

# ================================================


# --- 脚本主逻辑 (通常无需修改) ---
for step in "${STEPS_TO_CONVERT[@]}"; do
    echo "----------------------------------------------------"
    echo "准备处理 global_step_$step..."

    # 假设 FSDP 检查点保存在 'actor' 子目录中。
    # 这是之前脚本的默认行为。如果您的检查点文件直接在 global_step_10 下，
    # 请将下面的 "/actor" 去掉。
    local_dir="${BASE_EXPERIMENT_DIR}/global_step_${step}/actor"

    # 定义转换后 Hugging Face 格式模型的输出目录
    target_dir="${local_dir}/huggingface"

    # 检查输入的检查点目录是否存在
    if [ ! -d "$local_dir" ]; then
        echo "错误: 找不到输入目录 '$local_dir'。"
        echo "请确认您的 BASE_EXPERIMENT_DIR 设置是否正确，以及该目录下是否有名为 'actor' 的子文件夹。"
        echo "跳过此步骤。"
        continue # 继续处理列表中的下一个step
    fi

    # 检查目标目录是否已经存在，如果存在则跳过，避免重复工作
    if [ -d "$target_dir" ]; then
        echo "信息: 目标目录 '$target_dir' 已存在，转换已完成。"
        echo "跳过此步骤。"
        continue
    fi

    echo "开始转换..."
    echo "  - 输入 (FSDP Ckpt): $local_dir"
    echo "  - 输出 (Hugging Face): $target_dir"

    # 执行转换命令
    # 请确保您在能够找到 `scripts/model_merger.py` 的路径下运行此脚本
    python scripts/model_merger.py merge \
        --backend fsdp \
        --local_dir "$local_dir" \
        --target_dir "$target_dir"

    # 检查命令是否成功执行
    if [ $? -eq 0 ]; then
        echo "成功: global_step_$step 已转换为 Hugging Face 格式。"
    else
        echo "失败: global_step_$step 转换过程中发生错误。"
    fi
done

echo "----------------------------------------------------"
echo "所有任务已处理完毕。"
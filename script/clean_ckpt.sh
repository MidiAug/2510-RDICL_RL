#!/bin/bash

# 定义基础路径（根据你的需求设置）
base_path="/root/users/lcq/RDICL_RL/verl/qwen3_1.7b_base_math35_ext"

# 校验基础路径是否存在
if [ ! -d "$base_path" ]; then
    echo "错误：基础路径 $base_path 不存在！"
    exit 1
fi

# 查找所有符合 "global_step_xx" 格式的文件夹（xx为数字）
# 并筛选出其中包含 "actor/huggingface" 子目录的目标路径
target_dirs=$(find "$base_path" -type d -name "global_step_[0-9]*" | while read -r step_dir; do
    huggingface_dir="$step_dir/actor/huggingface"
    # 只保留存在的 huggingface 目录路径
    if [ -d "$huggingface_dir" ]; then
        echo "$huggingface_dir"
    fi
done)

# 检查是否找到目标目录
if [ -z "$target_dirs" ]; then
    echo "未找到任何符合条件的 global_step_xx/actor/huggingface 目录"
    exit 0
fi

# 列出即将删除的目录（安全提示）
echo "即将删除以下目录（及所有内容）："
echo "----------------------------------------"
echo "$target_dirs"
echo "----------------------------------------"

# 确认删除（避免误操作）
read -p "是否继续删除？[y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "操作已取消"
    exit 0
fi

# 执行删除操作
echo "开始删除..."
for dir in $target_dirs; do
    # 使用 rm -rf 强制删除目录（确保有足够权限）
    rm -rf "$dir"
    if [ $? -eq 0 ]; then
        echo "成功删除：$dir"
    else
        echo "警告：删除失败：$dir"
    fi
done

echo "删除操作完成！"
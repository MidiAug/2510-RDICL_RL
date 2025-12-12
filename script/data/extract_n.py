import os
import random
import json
from datasets import load_dataset

# ----------------- 参数 -----------------
PARQUET_FILE = "/home/lcq/data1/_tasks/2510-RDICL_RL/data/demo_8-train_icl.parquet"  # 输入的 parquet 文件
N = 10                               # 抽取数量
OUTPUT_FORMAT = "json"               # 可选: "json" 或 "csv"
# --------------------------------------

# 加载数据集
dataset = load_dataset("parquet", data_files=PARQUET_FILE, split="train")
data_list = dataset.to_list()

# 检查数量
if N > len(data_list):
    raise ValueError(f"N={N} 超过数据集总数 {len(data_list)}")

# 随机抽取 N 条
random.seed(42)
sampled_data = random.sample(data_list, N)

# 构造输出路径（同目录下）
dir_path = os.path.dirname(PARQUET_FILE)
base_name = os.path.splitext(os.path.basename(PARQUET_FILE))[0]
output_file = os.path.join(dir_path, f"{base_name}_sampled_{N}.{OUTPUT_FORMAT}")

# 保存为可阅读格式
if OUTPUT_FORMAT == "json":
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)
elif OUTPUT_FORMAT == "csv":
    import pandas as pd
    pd.DataFrame(sampled_data).to_csv(output_file, index=False)
else:
    raise ValueError("Unsupported OUTPUT_FORMAT, choose 'json' or 'csv'")

print(f"随机抽取 {N} 条样本，保存到 {output_file}")

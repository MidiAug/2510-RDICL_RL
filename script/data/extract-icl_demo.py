from datasets import load_dataset
import json
import random

# ----------------- 参数 -----------------
PARQUET_FILE = "/home/lcq/data1/_tasks/2510-RDICL_RL/data/train_icl.parquet"
N = 48  # 抽取 demo 数量
OUTPUT_FILE = f"/home/lcq/data1/_tasks/2510-RDICL_RL/data/demo_{N}_for_sft.jsonl"
# ----------------------------------------

# 设置随机种子保证可复现
random.seed(42)

# 加载 train_icl
dataset = load_dataset("parquet", data_files=PARQUET_FILE, split="train")

# 提取 demo 样本（prompt[1] 为 assistant 回复）
demo_samples = []
for item in dataset:
    prompt = item.get("prompt", [])
    if len(prompt) > 1 and prompt[1]["role"] == "assistant":
        demo_samples.append(item)

print(f"Found {len(demo_samples)} demo samples in the dataset.")

# ------------------------------------------------------
# 🔍 检测 demo_samples 中的重复项（question + answer）
# ------------------------------------------------------
seen = set()
duplicates = []

for item in demo_samples:
    prompt = item["prompt"]
    question = prompt[0]["content"].strip()
    answer = prompt[1]["content"].strip()
    key = (question, answer)

    if key in seen:
        duplicates.append(item)
    else:
        seen.add(key)

duplicate_count = len(duplicates)

print("=========================================")
print(f"Total demo samples        : {len(demo_samples)}")
print(f"Duplicate demos count     : {duplicate_count}")
print("=========================================")

# ------------------------------------------------------
# 🔥 获取唯一样本 Unique Set（key = (question, answer)）
# ------------------------------------------------------
unique_map = {}

for item in demo_samples:
    prompt = item["prompt"]
    question = prompt[0]["content"].strip()
    answer = prompt[1]["content"].strip()
    key = (question, answer)
    unique_map[key] = item

unique_samples = list(unique_map.values())

print(f"🔥 Unique samples count   : {len(unique_samples)}")
print("=========================================")

# ------------------------------------------------------
# 👉 决定抽样逻辑（根据 unique 数量和 N）
# ------------------------------------------------------
if len(unique_samples) < N:
    print(f"⚠️ 唯一样本数量 {len(unique_samples)} 小于需要的 {N}。")
    print("请选择:")
    print("  0: 不允许重复 → 使用所有 uniq（数量减少）")
    print("  1: 允许重复 → 使用放回抽样补齐到 N")

    choice = input("请输入 0 或 1: ").strip()

    if choice == "1":
        print("✅ 使用放回抽样补齐到 N 条")
        selected_demos = random.choices(unique_samples, k=N)
    else:
        print("✅ 不使用重复，仅使用全部唯一样本")
        selected_demos = unique_samples
else:
    # 唯一数量够用，正常抽样
    selected_demos = random.sample(unique_samples, k=N)

print(f"Final selected count = {len(selected_demos)}")

# ------------------------------------------------------
# 输出为 JSONL（SFT 格式：instruction / output）
# ------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in selected_demos:
        demo_prompt = item["prompt"]

        sft_sample = {
            "instruction": demo_prompt[0]["content"],
            "output": demo_prompt[1]["content"]
        }

        f.write(json.dumps(sft_sample, ensure_ascii=False) + "\n")

print(f"Saved {len(selected_demos)} samples to {OUTPUT_FILE}")

import json
import random
import copy
from datasets import load_dataset, Dataset

# ----------------- 参数 -----------------
PARQUET_FILE = "/home/lcq/data1/_tasks/2510-RDICL_RL/data/train.parquet"
DEMO_FILE = "/home/lcq/data1/_tasks/2510-RDICL_RL/data/all_demo-48.jsonl"

target_icl_size = 4
target_base_size = 4

DEMO_COUNT = 32
USE_RANDOM_DEMO = True

# 输出文件
OUTPUT_PARQUET = f"/home/lcq/data1/_tasks/2510-RDICL_RL/data/demo{DEMO_COUNT}-train_icl.parquet"
OUTPUT_DEMO_JSONL = f"/home/lcq/data1/_tasks/2510-RDICL_RL/data/demo{DEMO_COUNT}.jsonl"

random.seed(42)


# ----------------- 加载数据 -----------------
dataset = load_dataset("parquet", data_files=PARQUET_FILE, split="train")

# ----------------- 加载 demonstrations -----------------
all_demos = []
with open(DEMO_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        all_demos.append({
            "question": obj["instruction"],
            "solution": obj["output"]
        })

print(f"已加载演示样本总数: {len(all_demos)}")

# ----------------- 选择 DEMO_COUNT 个 demo -----------------
if DEMO_COUNT > len(all_demos):
    raise ValueError(f"DEMO_COUNT({DEMO_COUNT}) > available demos({len(all_demos)})")

if USE_RANDOM_DEMO:
    demo_pool = random.sample(all_demos, k=DEMO_COUNT)
else:
    demo_pool = all_demos[:DEMO_COUNT]

print(f"使用 {len(demo_pool)} 个演示样本作为 ICL 池。")

# 修正 ICL size
if target_icl_size > len(demo_pool):
    print(f"⚠ 警告: target_icl_size={target_icl_size} > demo_pool={len(demo_pool)} → 设置为 {len(demo_pool)}")
    target_icl_size = len(demo_pool)

# ----------------- 保存选定 demo_pool -----------------
with open(OUTPUT_DEMO_JSONL, "w", encoding="utf-8") as f:
    for demo in demo_pool:
        f.write(json.dumps(demo, ensure_ascii=False) + "\n")

print(f"已保存选定的演示样本 → {OUTPUT_DEMO_JSONL}")


# ----------------- 构造 ICL 数据 -----------------
output = []

for instance in dataset.to_list():
    original_prompt = instance["prompt"]

    # ---- Base（不加 ICL） ----
    for _ in range(target_base_size):
        item = copy.deepcopy(instance)
        item["prompt"] = original_prompt
        output.append(item)

    # ---- ICL ----
    selected_demos = random.sample(demo_pool, k=target_icl_size)

    for demo in selected_demos:
        messages = [
            {"role": "user", "content": demo["question"]},
            {"role": "assistant", "content": demo["solution"]},
        ] + original_prompt

        item = copy.deepcopy(instance)
        item["prompt"] = messages
        output.append(item)


# ----------------- 保存 parquet -----------------
output_dataset = Dataset.from_list(output)
output_dataset.to_parquet(OUTPUT_PARQUET)

print(f"已保存 parquet 文件 → {OUTPUT_PARQUET}")
print(f"输出样本总数 = {len(output)}")

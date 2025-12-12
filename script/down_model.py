from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-1.7B-Base"

# 下载并加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 下载并加载模型（只加载到 CPU，不分配显存）
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",   # 或者 device_map=None
    torch_dtype="auto", # 可选
    low_cpu_mem_usage=True  # 只分配最少 CPU 内存
)

print("Model and tokenizer loaded successfully.")

from datasets import load_dataset

FILE = "/home/lcq/data1/_tasks/2510-RDICL_RL/data/train_icl.parquet"

def inspect_parquet(path):
    ds = load_dataset("parquet", data_files=path, split="train", keep_in_memory=True)
    print(f"Loaded dataset from: {path}")
    print(f"Total rows: {len(ds)}\n")

    print("=== Columns / Fields ===")
    for col in ds.column_names:
        print("-", col)

if __name__ == "__main__":
    inspect_parquet(FILE)

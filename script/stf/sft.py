from dataclasses import dataclass, field
from transformers import HfArgumentParser
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from typing import Dict
from tqdm import tqdm
import jsonlines

@dataclass
class CustomConfig(SFTConfig):
    data_fpath: str = field(default=None)
    model_fpath: str = field(default=None)
    response_template: str = field(default=None)
    input_column: str = field(default=None)
    output_column: str = field(default=None)
    prompt: str = field(default=None)

def formatting_function(
    example: Dict[str, str],
    tokenizer,
    input_col: str,
    output_col: str, 
    prompt: str
) -> str:
    def wrap_query(question: str, solution: str, tokenizer, prompt: str) -> str:
        question = str(question)
        solution = str(solution)
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": solution}
        ]
        return tokenizer.apply_chat_template(
            messages,   
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False
        ).strip()
    
    # 返回字符串，SFTTrainer 的 formatting_func 需要返回字符串而不是列表
    return wrap_query(example["question"], example["solution"], tokenizer, prompt)
    #return wrap_query(example[input_col], example[output_col], tokenizer, prompt)

def create_data_collator(args, tokenizer) -> DataCollatorForCompletionOnlyLM:
    response_tokens = tokenizer.tokenize(
        args.response_template.replace("\\n", "\n"), 
    )
    response_ids = tokenizer.convert_tokens_to_ids(response_tokens)
    print("Response Tokens:", response_tokens, "Response Token IDs:", response_ids)
    return DataCollatorForCompletionOnlyLM(response_ids, tokenizer=tokenizer)

def main():
    parser = HfArgumentParser(CustomConfig)
    args = parser.parse_args_into_dataclasses()[0]
    
    config = AutoConfig.from_pretrained(args.model_fpath)

    with jsonlines.open(args.data_fpath) as reader:
        dataset = list(reader)

    tokenizer = AutoTokenizer.from_pretrained(args.model_fpath, config=config, padding_side="left")

    print("Processing dataset...")

    train_dataset = []
    for instance in dataset:
        question = instance["question"]
        solution = instance["solution"]
        train_dataset.append({"question": question, "solution": solution})
    # print("Training Size:", total)
    
    train_dataset = Dataset.from_list(train_dataset)

    data_collator = create_data_collator(args, tokenizer)

    print("Loading model...")

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_fpath,
    #     config=config,
    #     attn_implementation="flash_attention_2",
    # )

    trainer = SFTTrainer(
        model=args.model_fpath,
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        formatting_func=lambda x: formatting_function(
            x, tokenizer, args.input_column, args.output_column, args.prompt
        ),
    )
    
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
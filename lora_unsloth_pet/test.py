from unsloth import FastLanguageModel
import torch
local_model_path = '/root/sft/Qwen3-14B'
dataset_path = "/root/train/unsloth/data/pet/distill"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=local_model_path,
    max_seq_length=4096,  # 支持32K+长上下文
    device_map="auto",
    dtype=None,  # 自动选择最优精度
    load_in_4bit=True,  # 4bit量化节省70%显存
    load_in_8bit=False,
    full_finetuning=False
)

from datasets import load_dataset
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

def generate_conversation(examples):
    problems  = examples["problem"]
    solutions = examples["generated_solution"]
    conversations = []
    for problem, solution in zip(problems, solutions):
        conversations.append([
            {"role" : "user",      "content" : problem},
            {"role" : "assistant", "content" : solution},
        ])
        print(len(conversations))
    return { "conversations": conversations, }


reasoning_dataset = reasoning_dataset.map(generate_conversation, batched = True),

print(reasoning_dataset["conversations"])
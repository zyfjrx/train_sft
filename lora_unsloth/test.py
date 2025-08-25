from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
from unsloth.chat_templates import standardize_sharegpt

"""
nohup python train.py >20250818_1.log 2>&1 &
"""


# ============== 1、加载模型、tokenizer ====================================
local_model_path = '/root/sft/pretrained/unsloth/Qwen3-8B-unsloth-bnb-4bit'
dataset_path = "/root/train/unsloth/data/keywords_data_train.jsonl"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=local_model_path,
    max_seq_length=2048,  # 支持32K+长上下文
    device_map="auto",
    dtype=None,  # 自动选择最优精度
    load_in_4bit=True,  # 4bit量化节省70%显存
    load_in_8bit=False,
    full_finetuning=False
)
print(model)

# ===================== 2.数据加载与格式转换 ==========================
def convert_to_qwen_format(example):
    """
    {"conversation_id": 612, "category": "", "conversation": [{"human": "", "assistant": ""}], "dataset": ""}
    :return:
    """
    conversations = []
    for conv_list in example['conversation']:
        for conv in conv_list:
            conversations.append([
                {"role": "user", "content": conv['human'].strip()},
                {"role": "assistant", "content": conv['assistant'].strip()},

            ]
            )
    return {"conversations": conversations}


def format_func(example):
    formatted_texts = []
    for conv in example['conversations']:
        formatted_texts.append(
            tokenizer.apply_chat_template(
                conv,
                tokenize=False,         # 训练时不分词，true返回的是张量
                add_generation_prompt=False,    # 训练期间要关闭，如果是推理则设为True
            )
        )

    return {"text":formatted_texts}
dataset = load_dataset("json", data_files=dataset_path,split="train")

dataset = dataset.map(
    convert_to_qwen_format,
    batched=True,
    remove_columns=dataset.column_names
)
dataset = dataset.map(format_func, batched=True,remove_columns=dataset.column_names)
print(dataset[1:5])
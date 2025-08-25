from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

"""
nohup bash train_all.sh >20250818_1.log 2>&1 &
"""

# ============== 1、加载模型、tokenizer ====================================
local_model_path = '/root/sft/pretrained/Qwen3-8B'
dataset_path = "/root/train/unsloth/data/keywords_data_train.jsonl"



# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    quantization_config=None,
    # device_map="auto",
    trust_remote_code=True
)
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    local_model_path,
    trust_remote_code=True
)


# print(model)

# # ===================== 2.数据加载与格式转换 ==========================
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
                tokenize=False,  # 训练时部分词，true返回的是张量
                add_generation_prompt=False,  # 训练期间要关闭，如果是推理则设为True
            )
        )

    return {"text": formatted_texts}


dataset = load_dataset("json", data_files=dataset_path, split="train")
# dataset = dataset.shuffle(seed=43).select(range(100))
dataset = dataset.map(
    convert_to_qwen_format,
    batched=True,
    remove_columns=dataset.column_names
)
# print(dataset[0])

formatted_dataset = dataset.map(
    format_func,
    batched=True,
    remove_columns=dataset.column_names
)
# print(formatted_dataset[0])


# ==================== 3.使用trl库的训练器 ====================
trainer = SFTTrainer(
    model = model,
    processing_class=tokenizer,# 新写法
    train_dataset = formatted_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        deepspeed = "./ds_config/deepspeed_stage_3_config.json",  #添加deepspeed配置文件
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 64, # Use GA to mimic batch size!
        warmup_steps = 100,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = 3e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0,
        lr_scheduler_type = "constant_with_warmup",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

trainer = SFTTrainer(
    model=model,
    processing_class = tokenizer,
    # tokenizer=tokenizer,
    train_dataset=formatted_dataset,
    eval_dataset=None,  # Can set up evaluation!
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=64,  # Use GA to mimic batch size!
        warmup_steps=5,
        num_train_epochs=1,  # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate=2e-4,  # Reduce to 2e-5 for long training runs
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        save_total_limit=1, # 只保存一个检查点，节省磁盘空间
        seed=3407,
        report_to="none",  # Use this for WandB etc
    ),
)


trainer_stats = trainer.train()

# ==================== 4.保存训练结果 ====================================
# 完整保存model和tokenizer
trainer.save_model("/root/autodl-tmp/outputs/Qwen3-8B-sft-all")


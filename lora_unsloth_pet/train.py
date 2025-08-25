from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

"""
nohup python train.py >20250818_1.log 2>&1 &
llamafactory 统计token方法
python scripts/stat_utils/length_cdf.py --model_name_or_path /root/sft/Qwen3-14B --dataset distill_pet_dataset_3 --template qwen3
"""


# ============== 1、加载模型、tokenizer ====================================
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


model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=32,  # Best to choose alpha = rank or rank*2
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # rank stabilized LoRA
    loftq_config=None,  # LoftQ

)
print(model)

# ===================== 2.数据加载与格式转换 ==========================

def convert_to_qwen_format(examples):
    """
    输入格式示例:
    {
      "instruction": "你是？",
      "input": "啦啦啦啦啦啦啦啦",
      "output": "我是宠物助手？"  
    }
    """
    all_conversations = []
    instructions = examples['instruction']
    outputs = examples['output']
    
    for i in range(len(instructions)):
        conversations = [{
            "role": "user",
            "content": instructions[i].strip()
        }, {
            "role": "assistant",
            "content": outputs[i].strip()
        }]
        all_conversations.append(conversations)
    return {"conversations": all_conversations}



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

dataset = load_dataset("json", data_dir=dataset_path,split="train")
dataset = dataset.map(convert_to_qwen_format, batched=True,remove_columns=dataset.column_names)
# print(dataset[1:5])
formatted_dataset = dataset.map(format_func, batched=True,remove_columns=dataset.column_names)
# print(formatted_dataset[1:5])

# ==================== 3.使用trl库的训练器 ====================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = formatted_dataset,
    eval_dataset = None,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        warmup_steps = 30,
        num_train_epochs = 3, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        save_steps = 200,
        report_to = "tensorboard", # Use this for WandB etc
    ),
)
trainer_stats=trainer.train()

# ==================== 4.保存训练结果 ====================================
# 只保存lora适配器参数
model.save_pretrained("outputs/Qwen3-14B-sft-lora-adapter-unsloth-cosine")
tokenizer.save_pretrained("outputs/Qwen3-14B-sft-lora-adapter-unsloth-cosine")

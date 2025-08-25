import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig

# ===================== 1、加载模型与分词器======================
LOCAL_MODEL_PATH = "/root/autodl-tmp/pretrained/unsloth/Qwen3-8B-unsloth-bnb-4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 启用4-bit量化
    bnb_4bit_quant_type="nf4",      # 量化类型
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True  # 嵌套量化节省更多内存
)

model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True
)

# ====================== 2. 配置LoRA适配器 =========================
peft_config = LoraConfig(
    r=32,  # LoRA秩
    lora_alpha=32,  # 缩放因子
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,  # Dropout率
    bias="none",  # 偏置处理方式
    task_type="CAUSAL_LM"  # 任务类型
)

# 如果不用trl的SFTTrainer，需要手动包装模型
# model = get_peft_model(model, peft_config)

# ==================== 2、加载数据集 ======================
def convert_to_qwen_format(examples):
    conversations = []
    # 遍历每个对话样本,注意开启batch时，会自动套一层list
    for conv_list in examples["conversation"]:
        # 重建符合Qwen3标准的消息结构
        for conv in conv_list:
            conversations.append([
                {"role": "user", "content": conv['human'].strip()},
                {"role": "assistant", "content": conv['assistant'].strip()}
            ])

    return {"conversations": conversations}

def format_func(examples):
    """应用Qwen对话模板"""
    formatted_texts = []

    for conv in examples["conversations"]:
        # 使用分词器的内置模板
        formatted = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False,
        )
        formatted_texts.append(formatted)

    return {"text": formatted_texts}

dataset = load_dataset("json", data_files="data/keywords_data_train.jsonl", split="train")
dataset = dataset.shuffle(seed=42).select(range(100)) #抽取100条快速验证
# 格式化数据为 ChatML 格式
dataset = dataset.map(
    convert_to_qwen_format,
    batched=True,
    remove_columns=dataset.column_names
)

# 格式化成qwen模板
formatted_dataset = dataset.map(
    format_func,
    batched=True,
    remove_columns=["conversations"]
)

# ===================== 3.使用trl库的训练器 ====================
trainer = SFTTrainer(
    model = model,
    # tokenizer = tokenizer,   # ！标准写法不显示传递，通过模型关联
    processing_class=tokenizer,# 新写法
    peft_config=peft_config,   # 必须传
    train_dataset = formatted_dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

# ========== 启动训练 ==========
trainer_stats = trainer.train()

# ====================== 4.保存微调后的模型 ==========================
# 只保存 LoRA adapters，必须同时保存tokenizer
trainer.model.save_pretrained("/root/autodl-tmp/outputs/Qwen3-8B-sft-adapter")
tokenizer.save_pretrained("/root/autodl-tmp/outputs/Qwen3-8B-sft-adapter")

# 完整保存模型和tokenizer
# trainer.save_model("/root/autodl-tmp/outputs/Qwen3-8B-sft-lora")

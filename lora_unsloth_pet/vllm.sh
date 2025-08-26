vllm serve /root/sft/Qwen3-14B \
 --served-model-name Qwen3-14B-sft-all \
 --max-model-len 32K \
 --enable-lora \
 --max-lora-rank 32 \
 --lora-modules adapter_v1=/root/train/train_sft/lora_unsloth_pet/outputs/Qwen3-14B-sft-lora-adapter-unsloth


 # 后台启动
nohup vllm serve /root/sft/Qwen3-14B \
 --served-model-name Qwen3-14B-sft-all \
 --max-model-len 32K \
 --enable-lora \
 --max-lora-rank 32 \
 --lora-modules adapter_v1=/root/train/train_sft/lora_unsloth_pet/outputs/Qwen3-14B-sft-lora-adapter-unsloth >vllm.log 2>&1 &


 nohup vllm serve /root/sft/Qwen3-14B \
 --served-model-name Qwen3-14B \
 --max-model-len 32K \
 >vllm.log 2>&1 &
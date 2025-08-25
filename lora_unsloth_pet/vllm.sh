vllm serve /root/sft/Qwen3-14B \
 --served-model-name Qwen3-14B-sft-all \
 --max-model-len 32K \
 --enable-lora \
 --max-lora-rank 32 \
 --lora-modules adapter_v1=/root/train/unsloth/lora_unsloth_pet/outputs/Qwen3-14B-sft-lora-adapter-unsloth
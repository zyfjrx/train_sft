# 压测
evalscope perf \
--url "http://127.0.0.1:8000/v1/chat/completions" \
--parallel 5 \
--model Qwen3-14B-sft-all \
--number 20 \
--api openai \
--dataset openqa \
--stream
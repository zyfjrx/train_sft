from unsloth import FastLanguageModel
from transformers import TextStreamer
import json

local_model_path = '/root/autodl-tmp/pretrained/unsloth/Qwen3-8B-unsloth-bnb-4bit'
lora_adapter_path = '/root/autodl-tmp/outputs/Qwen3-8B-sft-lora-adapter-unsloth'

# 1、加载基座模型、tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=local_model_path,
    max_seq_length=2048,  # 支持32K+长上下文
    device_map="auto",
    dtype=None,  # 自动选择最优精度
    load_in_4bit=True,  # 4bit量化节省70%显存
)

# 2、注入lora适配器
model.load_adapter(lora_adapter_path)
# 启用unsloth推理加速
FastLanguageModel.for_inference(model)
model.eval()

# 3、构造数据  {text:<im_start>\nsystem....<im_end>} ==> tokenizer.apply_chat_template
with open('../data/keywords_data_test.jsonl', 'r', encoding='utf-8') as read_file, \
        open('../data/keywords_data_test_result.jsonl', 'w+', encoding='utf-8') as write_file:
    for line in read_file.readlines():
        line = json.loads(line.strip())
        # {"conversation_id": 1, "category": "dialogue", "conversation": [{"human": "抽取出文本中的关键词：\n标题：人工神经网络在猕猴桃种类识别上的应用\n文本：在猕猴桃介电特性研究的基础上,将人工神经网络技术应用于猕猴桃的种类识别.该种类识别属于模式识别,其关键在于提取样品的特征参数,在获得特征参数的基础上,选取合适的网络通过训练来进行识别.猕猴桃种类识别的研究为自动化识别果品的种类、品种和新鲜等级等提供了一种新方法,为进一步研究果品介电特性与其内在品质的关系提供了一定的理论与实践基础.", "assistant": "食品科学技术基础学科;猕猴桃;应用;人工神经网络;介电特性;识别"}], "dataset": "psychology"}
        # 取出每条数据的 human内容、assistant内容
        user_context = line['conversation'][0]['human']
        assistant_context = line['conversation'][0]['assistant']
        # 调用微调后的模型，生成对应的回复结果
        message = [
            {"role": "user", "content": user_context}
        ]
        format_messages = tokenizer.apply_chat_template(
            message,
            tokenize=False,  # 训练时部分词，true返回的是张量
            add_generation_prompt=True,  # 训练期间要关闭，如果是推理则设为True
            enable_thinking=False,  # 禁用思考
        )
        inputs = tokenizer(format_messages, return_tensors='pt').to(model.device)
        # 将{输入、回复结果、标准答案} 写出到文件
        # 调用model.generate()  ======== 一次性输出
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
        )
        # 只输出回答部分
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        # print(response)

        write_file.write(json.dumps({"input":user_context,"predict":response,"target":assistant_context}, ensure_ascii=False)+'\n')


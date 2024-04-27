from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda'

model_path = 'Qwen/Qwen1.5-14B-Chat'
qwen14B = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype='auto', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = '回答以下选择题，给出选项'
messages = [
    {
        "role": "system",
        "content": "以下是关于Dota2游戏知识的单项选择题，请选出其中的正确答案。"
    },
    {
        "role": "user",
        "content": "以下背景故事属于哪个物品？位居中央的宝石依然能反射出其制造者的景象。\nA. 魔杖\nB. 莲花\nC. A杖\nD. 天鹰之戒\n答案："
    },
    {
        "role": "assistant",
        "content": "A"
    },
    {
        "role": "user",
        "content": "亚巴顿的初始移动速度是多少？\nA. 305\nB. 310\nC. 295\nD. 290\n答案："
    },
    {
        "role": "assistant",
        "content": "B"
    },
    {
        "role": "user",
        "content": "梦境缠绕的冷却时间是多久？\nA. 100\nB. 6\nC. 20/15/10/5\nD. 70/65/60\n答案："
    },
    {
        "role": "assistant",
        "content": "D"
    },
    {
        "role": "user",
        "content": "以下背景故事属于哪个物品？这个有魔力的球体曾经保护了一位历史上最有名的英雄\nA. 狂战斧\nB. 死灵书\nC. 恢复指环\nD. 林肯法球\n答案："
    },
    {
        "role": "assistant",
        "content": ""
    }
]

text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors='pt').to(device)
generated_ids = qwen14B.generate(model_inputs.input_ids, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

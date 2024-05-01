import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch.utils.cpp_extension

device = 'cuda'

model_path = 'Qwen/Qwen1.5-14B-Chat'
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype='auto', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path)

torch.utils.cpp_extension.CUDA_HOME


def format_example(exam, include_answer=True, cot=False, add_prompt=''):
    chat = []
    CHOICES = 'ABCDEFGHIJK'
    example = add_prompt + exam['question']
    for i, choice in enumerate(exam['choices']):
        example += f'\n{CHOICES[i]}. {choice}'
    
    chat_answer = {"role": "assistant"}
    example += '\n答案：'
    chat.append({"role": "user", "content": example})
    if include_answer:
        if cot:
            ans = "让我们一步一步思考，\n" + \
                exam["explanation"] + f"\n所以答案是{exam['answer']}。"
        else:
            ans = exam["answer"]
        chat_answer["content"] = f'\n{ans}'
    chat.append(chat_answer)
    return chat


def generate_few_shot_prompt(few_shot=[], cot=False):
    chat = [
        {"role": "system", "content": "以下是关于游戏Dota2知识考试的单项选择题，请选出其中正确的答案。\n\n"}
    ]
    for exam in few_shot:
        chat.extend(format_example(exam, cot=cot))
    return chat

def judge(outputs: str, answer: str):
    outputs = outputs.strip()
    return outputs == answer


def dota2_eval(file, model, tokenizer):
    with open(file, 'r') as f:
        exams = json.load(f)
    few_shot = exams[0: 4]
    chat = generate_few_shot_prompt(few_shot, cot=False)
    for c in chat:
        print(c['role'])
        print(c['content'], end='')
    return
    right = 0
    for i in range(4, len(exams)):
        chat_one = format_example(exams[i], include_answer=False, cot=False)
        chat_tmp = chat
        chat_tmp.append(chat_one)
        formatted_chat = tokenizer.apply_chat_template(chat_tmp, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted_chat, return_tensors="pt",add_special_tokens=False)
        inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
        outputs = model.generate(**inputs, max_new_tokens=512)
        decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
        if judge(decoded_output, exams[i]):
            right += 1
    print("acc=", right / (len(exams) - 3))

dota2_eval('/data/yinxiaoln/datasets/dota2eval/data2-eval.json', model, tokenizer)
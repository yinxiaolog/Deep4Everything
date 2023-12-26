import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig,snapshot_download
import json
import string
import time
# tokenizer_path='/data/usr/jy/Baichuan2/fine-tune/output/choose_train_v2'
# model_path="/data/usr/jy/Baichuan2/fine-tune/output/choose_train_v2"
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat", revision='v1.0.2')
tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="cuda:0", 
                              trust_remote_code=True, torch_dtype=torch.float16, 
                              cache_dir='/data/usr/jy/Baichuan2/tokenizer')
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda:0", 
                              trust_remote_code=True, torch_dtype=torch.float16,
                              cache_dir='/data/usr/jy/Baichuan2/model')
model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-13B-Chat")
def predict(model,tokenizer,message):
    response = model.chat(tokenizer, message)
    return response
def judge(a,b):
    a=process_string(a)
    b=process_string(b)
    return a==b
def process_string(s):
    for punctuation in string.punctuation:
        s = s.replace(punctuation, '')
    return s.replace(" ", "").upper()
    
with open('/data/usr/jy/Baichuan2/fine-tune/data/converted_choose_validation_v2.json', 'r', encoding="utf8") as file:
    val_data = json.load(file)
cor_cnt=err_cnt=0
tot_cnt=len(val_data)
for item in val_data:
    for conversation in item['conversations']:
        if conversation['from'] == 'human':
            message=[{"role": "user", "content": conversation['value']}]
            response=predict(model,tokenizer,message)
        if conversation['from'] == 'gpt':
            answer=conversation['value']
    item['response']=response
    judge_res=judge(response,answer)
    if judge_res:
        cor_cnt += 1
    else:
        err_cnt += 1
    print(judge_res,response,answer)
print("correct:%d error:%d total:%d rate:%f "%(cor_cnt,err_cnt,tot_cnt,cor_cnt/tot_cnt))
with open('/data/usr/jy/Baichuan2/fine-tune/test/predict/converted_choose_validation_v2"+str(int(time.time()))+".json', 'w', encoding="utf8") as file:
    json.dump(val_data, file, indent=4,ensure_ascii=False)

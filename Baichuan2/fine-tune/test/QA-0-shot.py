import os
import torch
from modelscope import snapshot_download, AutoModelForCausalLM, AutoTokenizer,GenerationConfig
import json
from tqdm import tqdm
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from simhash import Simhash
import argparse

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--predictions_output_file', default=None, type=str)
parser.add_argument('--model_name', default=None, type=str)
parser.add_argument('--vaidation_file', default=None, type=str)
args = parser.parse_args()
base_model = args.base_model
predictions_output_file = args.predictions_output_file
model_name = args.model_name
vaidation_file = args.vaidation_file

llama2_prompt = '[INST] <<SYS>>\n You are a helpful, respectful and honest ' \
            'assistant. Always answer as helpfully as possible, while being ' \
            'safe. Your answers should not include any harmful, unethical, ' \
            'racist, sexist, toxic, dangerous, or illegal content. Please ' \
            'ensure that your responses are socially unbiased and positive in ' \
            'nature. \n<</SYS>>\n\n{} [/INST]'
alpace_prompt = 'Below is an instruction that describes a task. ' \
            'Write a response that appropriately completes the request.\n\n' \
            '### Instruction:\n{}\n\n' \
            '### Response: '
prompt_input = (
    "指令：{}。\n"
    "问题：{}。\n"
    "请根据指令回答问题，不需要做解释。\n"
    "回答："
)
prompt_muju = ('你现在是一名专业的模具领域问答模型，请根据问题给出准确的回答，不需要解释，只需要根据指令返回准确答案。'
                '\n\n### 问题: {}\n### 回答: ')
## model tokenizer init
tokenizer = AutoTokenizer.from_pretrained(base_model, device_map=device, 
                              trust_remote_code=True, torch_dtype=torch.bfloat16, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device, 
                              trust_remote_code=True, torch_dtype=torch.bfloat16,)
model.eval()
with open(vaidation_file, 'r', encoding='utf-8') as f:
    vaidation_json = json.load(f)
prediction_output = []
print("validation len: {}".format(len(vaidation_json)))
print("print 5 exmaple of validation {}".format(vaidation_json[:5]))
# vaidation_json = vaidation_json[:10]
with torch.no_grad():
    dataloader = torch.utils.data.DataLoader(vaidation_json, batch_size=1)
    for batch in tqdm(dataloader):
        instruction_, input_, output_ = batch['instruction'], batch['input'], batch['output']
        # queries = [prompt.format(ins + query) for ins, query in zip(instruction_, input_)]
        # queries = [prompt_input.format(ins + query) for ins, query in zip(instruction_, input_)]
        queries = [prompt_muju.format(ins+query) for ins, query in zip(instruction_, input_)]
        inputs = tokenizer(queries, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(device)
        outputs = model.generate(**inputs,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                do_sample=False, max_new_tokens=512)
        for idx in range(len(outputs)):
            output = outputs[idx][len(inputs["input_ids"][idx]):]
            response = tokenizer.decode(output, skip_special_tokens=True)
            prediction_output.append({
                "instruction": instruction_[idx],
                "intput": input_[idx],
                "output": output_[idx],
                model_name: response
            })
with open(predictions_output_file, 'w', encoding='utf-8') as f:
    json.dump(prediction_output, f, ensure_ascii=False, indent=4)

print("finish")

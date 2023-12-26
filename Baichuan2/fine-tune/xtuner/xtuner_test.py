import sys
import torch, jieba
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig, snapshot_download
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json, string, time, os, re
import numpy as np

time_str = str(int(time.time()))

verbose = False
# 验证数据集
val_data_path = "/data/yinxiaoln/datasets/muju/question-ans-v2/validation/question-ans-validation-v2.json"
# 结果输出路径
result_path = sys.argv[1]
# 微调后的模型路径
model_dir = '/data/yinxiaoln/save/'
# 微调后的模型名
model_name = 'bc2_7b_chat_qav2_e5'
model_path = sys.argv[2]
print(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="cuda:0",
                                          trust_remote_code=True, torch_dtype=torch.float16, )
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0",
                                             trust_remote_code=True, torch_dtype=torch.float16, )
model.generation_config = GenerationConfig.from_pretrained(model_path)


# model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")

def predict(model, tokenizer, message):
    return model.chat(tokenizer, message)


def judge(a, b):
    return process_string(a) == process_string(b)


def process_string(s):
    for punctuation in string.punctuation:
        s = s.replace(punctuation, "")
    zh_punctuation = ['。', '，', '；', '：', '”', '“']
    for zp in zh_punctuation:
        s = s.replace(zp, "")
    s = s.replace(" ", "").upper()
    if s in ['正确', '对', '是', '是的', 'TRUE', 'YES']:
        s = "TRUE"
    elif s in ['错误', '不正确', '不对', '否', 'FALSE', 'NO']:
        s = 'FALSE'
    return s


# Metric
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # if data_args.ignore_pad_token_for_loss:
    #     # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
        result = scores[0]
        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))
        bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))
    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict


def main():
    with open(val_data_path, "r", encoding="utf8") as file:
        val_data = json.load(file)
    tot_cnt = len(val_data)
    print("tot_cnt", tot_cnt)
    cnt = 0
    sum_score = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bleu-4': 0}
    for item in val_data:
        # del item['bc2_7b_chat_qav2_e1-score']
        message = [{"role": "system", "content": item["instruction"]},
                   {"role": "user", "content": item["input"]}]
        response = predict(model, tokenizer, message)
        answer = item["output"]
        encoded_reponse = tokenizer.encode(response, return_tensors="pt")
        encoded_answer = tokenizer.encode(answer, return_tensors="pt")
        if verbose or cnt % 100 == 0:
            print(cnt, '/', tot_cnt, time.asctime())
            # print(judge_res)
            print(response, answer)
            print(encoded_reponse, encoded_answer)
        judge_res = compute_metrics([encoded_reponse, encoded_answer])
        sum_score['rouge-1'] += judge_res['rouge-1']
        sum_score['rouge-2'] += judge_res['rouge-2']
        sum_score['rouge-l'] += judge_res['rouge-l']
        sum_score['bleu-4'] += judge_res['bleu-4']
        item[model_name] = response
        # item[model_name+"-score"]=judge_res
        cnt += 1
    sum_score['rouge-1'] /= tot_cnt
    sum_score['rouge-2'] /= tot_cnt
    sum_score['rouge-l'] /= tot_cnt
    sum_score['bleu-4'] /= tot_cnt
    val_data[0][model_name + '-avg_score'] = sum_score
    print(sum_score)
    with open(result_path, "w", encoding="utf8") as file:
        json.dump(val_data, file, indent=4, ensure_ascii=False)
    print("saved to ", result_path)


if __name__ == '__main__':
    main()

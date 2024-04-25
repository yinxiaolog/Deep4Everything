import os
import json
from datasets import concatenate_datasets, load_dataset
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertForMaskedLM, BertConfig, DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from itertools import chain
import torch


bookcorpus = load_dataset("bookcorpus", split="train")
wiki = load_dataset("wikipedia", "20220301.en",
                    split='train', trust_remote_code=True)
wiki = wiki.remove_columns([col for col in wiki.column_names if col != 'name'])
dataset = concatenate_datasets([bookcorpus, wiki])
d = dataset.train_test_split(test_size=0.1)


def dataset_to_text(dataset, output_filename='data.txt'):
    if os.path.exists(output_filename):
        return
    with open(output_filename, 'w') as f:
        for t in dataset['text']:
            print(t, file=f)


dataset_to_text(d['train'], '/data/yinxiaoln/datasets/train.txt')
dataset_to_text(d['test'], '/data/yinxiaoln/datasets/test.txt')

special_tokens = [
    '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '<S>', '<T>'
]

files = ['/data/yinxiaoln/datasets/train.txt']
vocab_size = 30_522
max_length = 512
truncate_longer_samples = False
tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=files, vocab_size=vocab_size,
                special_tokens=special_tokens)
tokenizer.enable_truncation(max_length=max_length)
model_path = 'pretrained-bert'

if not os.path.isdir(model_path):
    os.mkdir(model_path)

tokenizer.save_model(model_path)

with open(os.path.join(model_path, 'config.json'), 'w') as f:
    tokenizer_cfg = {
        'do_lower_case': True,
        'unk_token': '[UNK]',
        'sep_token': '[SEP]',
        'pad_token': '[PAD]',
        'cls_token': '[CLS]',
        'mask_token': '[MASK]',
        'model_max_length': max_length,
        'max_len': max_length
    }
    json.dump(tokenizer_cfg, f)

tokenizer = BertTokenizerFast.from_pretrained(model_path)


def encode_with_truncation(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length',
                     max_length=max_length, return_special_tokens_mask=True)


def encode_without_truncation(examples):
    return tokenizer(examples['text'], return_special_tokens_mask=True)


encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation
train_dataset = d['train'].map(encode, batched=True)
test_dataset = d['test'].map(encode, batched=True)

if truncate_longer_samples:
    train_dataset.set_format(type='torch', columns=[
                             'input_ids', 'attention_mask'])
    test_dataset.set_format(type="torch", columns=[
                            "input_ids", "attention_mask"])
else:
    train_dataset.set_format(
        columns=["input_ids", "attention_mask", "special_tokens_mask"])
    test_dataset.set_format(
        columns=["input_ids", "attention_mask", "special_tokens_mask"])


def group_texts(examples):
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length
    result = {
        k: [t[i: i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

if not truncate_longer_samples:
    train_dataset = train_dataset.map(group_texts, batched=True,
                                      desc=f"Grouping texts in chunks of {max_length}")
    test_dataset = test_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts in chunks of {max_length}")
    # convert them from lists to torch tensors
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")


model_config = BertConfig(vocab_size=vocab_size,
                          max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)
# initialize the data collator, randomly masking 20% (default is 15%) of the tokens
# for the Masked Language Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.2
)
training_args = TrainingArguments(
    output_dir=model_path,  # output directory to where save model checkpoint
    evaluation_strategy="steps",  # evaluate each `logging_steps` steps
    overwrite_output_dir=True,
    num_train_epochs=10,  # number of training epochs, feel free to tweak
    # the training batch size, put it as high as your GPU memory fits
    per_device_train_batch_size=10,
    # accumulating the gradients before updating the weights
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=64,  # evaluation batch size
    logging_steps=1000,  # evaluate, log and save model checkpoints every 1000 step
    save_steps=1000,
    # load_best_model_at_end=True, # whether to load the best model (in terms of loss)
    # at the end of training
    # save_total_limit=3, # whether you don't have much space so you
    # let only 3 model weights saved in the disk
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
# train the model
trainer.train()

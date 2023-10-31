import os
import re
from collections import defaultdict

import pandas as pd
from d2l import torch as d2l
from torch.utils import data

import utils.bert_dataset_pre as pre
from log import logger
from utils.constant import *

LOG = logger.Logger().get_logger()


def read_hdfs_log(config):
    data_path = config[DATA][DATA_PATH]
    with open(os.path.join(data_path, 'HDFS.log'), 'r') as f:
        lines = f.readlines()
    blk_id_2_lines = defaultdict(list)
    for line in lines:
        blk_id = re.findall(r'(blk_-?\d+)', line)
        blk_id_2_lines[blk_id[0]].append(line)

    blk_label_df = pd.read_csv(os.path.join(data_path, 'anomaly_label.csv'))
    blk_label_dict = {}
    for _, row in blk_label_df.iterrows():
        blk_label_dict[row['BlockId']] = row['Label']
    train_dict = defaultdict(list)
    test_abnormal_dict = defaultdict(list)

    for k, v in blk_id_2_lines.items():
        if blk_label_dict[k] == 'Anomaly':
            train_dict[k] = v
        else:
            test_abnormal_dict[k] = v

    keys = [key for key in train_dict]
    train_normal_dict = defaultdict(list)
    test_normal_dict = defaultdict(list)
    valid_num = int(len(keys) * 0.2)
    for i in range(valid_num):
        test_normal_dict[keys[i]] = train_dict[keys[i]]
    for i in range(valid_num, len(keys), 1):
        train_normal_dict[keys[i]] = train_dict[keys[i]]

    train_normal = []
    for v in train_normal_dict.values():
        train_normal.append([line for line in v])
    test_normal = []
    for v in test_normal_dict.values():
        test_normal.append([line for line in v])
    test_abnormal = []
    for v in test_abnormal_dict.values():
        test_abnormal.append([line for line in v])
    return train_normal, test_normal, test_abnormal


class _HdfsLogDataset(data.Dataset):
    def __init__(self, config: dict, paragraphs, ):
        self.config = config
        self.data_path = config[DATA][DATA_PATH]
        self.window_size = config[HYPER_PARAMS][WINDOW_SIZE]
        self.max_len = config[HYPER_PARAMS][MAX_LEN]
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in paragraphs]
        words = [word for paragraph in paragraphs for word in paragraph]
        self.vocab = d2l.Vocab(words, min_freq=0, reserved_tokens=[
            PAD, MASK, CLS, SEP
        ])
        self.masked_ratio = config[HYPER_PARAMS][MASKED_RATIO]
        examples = []
        for paragraph in paragraphs:
            examples.extend(pre.get_nsp_data(paragraph, paragraphs, self.max_len))
        examples = [(pre.get_mlm_data(tokens, self.vocab, self.masked_ratio) + (segments, is_next))
                    for tokens, segments, is_next in examples]
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = pre.pad_bert_inputs(
            examples, self.max_len, self.vocab)
        LOG.info(len(self.all_token_ids))

    def __len__(self):
        return len(self.all_token_ids)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])


def _get_data_loader(dataset_type, conf):
    train_normal, test_normal, test_abnormal = read_hdfs_log(conf)
    if dataset_type == 'train_normal':
        dataset = _HdfsLogDataset(conf, paragraphs=train_normal)
    elif dataset_type == 'test_normal':
        dataset = _HdfsLogDataset(conf, paragraphs=test_normal)
    elif dataset_type == 'test_abnormal':
        dataset = _HdfsLogDataset(conf, paragraphs=test_abnormal)
    else:
        LOG.error('unknown dataset_type=%s', dataset_type)
        exit(1)
    batch_size = conf[HYPER_PARAMS][BATCH_SIZE]
    num_workers = os.cpu_count() if conf[HYPER_PARAMS][NUM_WORKERS] == AUTO else conf[HYPER_PARAMS][NUM_WORKERS]
    data_loader = data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)
    return data_loader, dataset.vocab


def get_hdfs_log_train_normal(config):
    return _get_data_loader('train_normal', config)


def get_hdfs_log_test_normal(config):
    return _get_data_loader('test_normal', config)


def get_hdfs_log_test_abnormal(config):
    return _get_data_loader('test_abnormal', config)

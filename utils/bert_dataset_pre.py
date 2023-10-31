import random

import torch

from log import logger
from utils.constant import *

LOG = logger.Logger().get_logger()


def get_next_sentence(sentence, next_sentence, paragraphs):
    """ Get random sentence from paragraphs
    paragraphs : [[["token_a", "token_b"], ["token_c", "token_d"]]]]
    """
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False

    return sentence, next_sentence, is_next


def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = [CLS] + tokens_a + [SEP]
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + [SEP]
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def get_nsp_data(paragraph, paragraphs, max_len):
    nsp = []
    for i in range(len(paragraph) - 1):
        tokens_pre, tokens_next, is_next = get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs
        )
        if len(tokens_pre) + len(tokens_next) + 3 > max_len:
            LOG.warning('nsp data exceed max_len=%d: token_pre=%s, token_next=%s', max_len, tokens_pre, tokens_next)
            continue
        tokens, segments = get_tokens_and_segments(tokens_pre, tokens_next)
        nsp.append((tokens, segments, is_next))
    return nsp


def mask_tokens(vocab, tokens, token_idx_without_reserved_tokens, num_mask):
    mlm_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(token_idx_without_reserved_tokens)
    for i in range(num_mask):
        masked_token_id = token_idx_without_reserved_tokens[i]
        if random.random() < 0.8:
            masked_token = MASK
        else:
            if random.random() < 0.5:
                masked_token = tokens[masked_token_id]
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_tokens[masked_token_id] = masked_token
        pred_positions_and_labels.append((masked_token_id, tokens[masked_token_id]))
    return mlm_tokens, pred_positions_and_labels


def get_mlm_data(tokens, vocab, masked_ratio):
    tokens_idx_without_reserved_tokens = []
    for i, token in enumerate(tokens):
        if token not in [CLS, SEP]:
            tokens_idx_without_reserved_tokens.append(i)
    num_mask = max(1, round(len(tokens) * masked_ratio))
    mlm_tokens, pred_positions_and_labels = mask_tokens(vocab, tokens, tokens_idx_without_reserved_tokens, num_mask)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    positions = [v[0] for v in pred_positions_and_labels]
    labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_tokens], positions, vocab[labels]


def pad_bert_inputs(examples, max_len, vocab):
    max_masked = round(max_len * 0.15)
    token_ids_list, segments_list, valid_lens = [], [], []
    pred_positions_list, mlm_weights_list, mlm_labels_list = [], [], []
    nsp_labels = []
    LOG.info(len(examples))
    cnt = 0
    for mlm_tokens, positions, mlm_labels, segments, is_next in examples:
        token_ids_list.append(torch.tensor(mlm_tokens + [vocab[PAD]] * (max_len - len(mlm_tokens)), dtype=torch.long))
        segments_list.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        valid_lens.append(torch.tensor(len(mlm_tokens), dtype=torch.float32))
        pred_positions_list.append(torch.tensor(positions + [0] * (max_masked - len(positions)), dtype=torch.long))
        mlm_weights_list.append(torch.tensor([1.0] * len(mlm_labels) + [0.0] * (max_masked - len(positions))))
        mlm_labels_list.append(torch.tensor(mlm_labels + [0] * (max_masked - len(mlm_labels)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
        cnt += 1
        if cnt > 10000:
            LOG.info(cnt)
            break
    return token_ids_list, segments_list, valid_lens, pred_positions_list, mlm_weights_list, mlm_labels_list, nsp_labels

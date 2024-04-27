import sys
import json
import numpy as np
from modelscope import snapshot_download
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import jieba
import editdistance
from loguru import logger as log
import torch
import torch.nn.functional as F
from torch import nn

from tqdm import tqdm

log.remove()
log.add(sys.stderr, level='INFO')

# model_dir = snapshot_download('tiansz/bert-base-chinese')
model_dir = '/home/yinxiaoln/.cache/modelscope/hub/tiansz/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_dir)
bert = BertModel.from_pretrained(model_dir)

config_path = '../../kg/bert/chinese_wwm_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../../kg/bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/yinxiaoln/.cache/modelscope/hub/tiansz/bert-base-chinese/vocab.txt'


class Config():
    epochs = 100
    batch_size = 64
    lr = 1e-5

    DEVICE = ['cuda:0', 'cuda:1']
    device = torch.device(DEVICE[0])
    devices = [torch.device(d) for d in DEVICE]

    max_len_bert = 512
    max_cols = 25
    max_question = 128
    checkpoint_path = '/data/data1/checkpoint'


OP = {0: '>', 1: '<', 2: '==', 3: '!=', 4: '<=', 5: '>=', 6: 'None'}
AGG = {0: '', 1: 'AVG', 2: 'MAX', 3: 'MIN', 4: 'COUNT', 5: 'SUM', 6: 'None'}
CONN = {0: '', 1: 'and', 2: 'or'}

loss_func = nn.CrossEntropyLoss(reduction='mean')
optim = None
scheduler = None
device = Config.device


def read_data(data_file, table_file):
    data, tables = [], {}
    with open(table_file) as f:
        for json_line in f:
            table = json.loads(json_line)
            tables[table['id']] = table
    with open(data_file) as f:
        for json_line in f:
            d = json.loads(json_line)
            table = tables[d['table_id']]
            d['table'] = table
            d['header2id'] = {j: i for i, j in enumerate(table['header'])}
            data.append(d)
    return data


def most_similar(s, slist):
    """从词表中找最相近的词（当无法全匹配的时候）
    """
    if len(slist) == 0:
        return s
    scores = [editdistance.eval(s, t) for t in slist]
    return slist[np.argmin(scores)]


def most_similar_2(w, s):
    """从句子s中找与w最相近的片段，
    借助分词工具和ngram的方式尽量精确地确定边界。
    """
    sw = jieba.lcut(s)
    sl = list(sw)
    sl.extend([''.join(i) for i in zip(sw, sw[1:])])
    sl.extend([''.join(i) for i in zip(sw, sw[1:], sw[2:])])
    return most_similar(w, sl)


def seq_padding(input, padding=0, max_len=None):
    if max_len is None:
        max_len = max([len(x) for x in input])
    tensors = []
    for x in input:
        if len(x) < max_len:
            x = torch.cat([x, torch.tensor([padding] * (max_len - (len(x))), dtype=torch.long)])
        tensors.append(x)
    return torch.stack(tensors)


class TrainDataset(Dataset):
    def __init__(self, data_file, table_file, max_len, max_cols, max_question) -> None:
        super().__init__()
        data = read_data(data_file, table_file)
        self.data = data
        self.max_len = max_len
        self.max_cols = max_cols
        self.max_question = max_question

    def __getitem__(self, index):
        d = self.data[index]
        table = d['table']
        headers = table['header']
        assert len(headers) < self.max_cols
        assert len(d['question']) < self.max_question
        # 编码question和列名到x
        x, attention_mask = [], []
        encoded_dict = tokenizer.encode_plus(d['question'])
        x.extend(encoded_dict['input_ids'])
        attention_mask.extend(encoded_dict['attention_mask'])
        question_token_len = len(x)
        x_mask = [0] * len(x)
        x_mask[1: -1] = [1] * (question_token_len - 2)

        cls_headers = []
        for header in headers:
            cls_headers.append(len(x))
            encoded_dict = tokenizer.encode_plus(header)
            x.extend(encoded_dict['input_ids'])
            attention_mask.extend(encoded_dict['attention_mask'])

        cls_headers_mask = [1] * len(cls_headers)

        # 给每一列添加agg标识
        agg = [len(AGG) - 1] * len(headers)
        for _sel, _op in zip(d['sql']['sel'], d['sql']['agg']):
            agg[_sel] = _op

        # 条件连接符
        cond_conn_op = [d['sql']['cond_conn_op']]
        # 条件列和值
        cond_cols = torch.zeros(question_token_len, dtype=torch.long)
        cond_ops = torch.zeros(question_token_len, dtype=torch.long) + len(OP) - 1

        for cond in d['sql']['conds']:
            if cond[2] not in d['question']:
                cond[2] = most_similar_2(cond[2], d['question'])
            if cond[2] not in d['question']:
                log.error('cond not in question')
                continue
            cond_token_ids = tokenizer.encode(cond[2])

            start, end = 1, 1
            for i in range(1, question_token_len - 1):
                for j in range(1, len(cond_token_ids) - 1):
                    if x[i + j - 1] != cond_token_ids[j]:
                        if j - 1 > end - start:
                            start = i
                            end = i + j - 1
                        break
                    if j == len(cond_token_ids) - 2 and cond_token_ids[j] == x[i + j - 1]:
                        j = j + 1
                        if j - 1 > end - start:
                            start = i
                            end = i + j - 2
                        break

            cond_cols[start: end + 1] = cond[0]
            cond_ops[start: end + 1] = cond[1]

        x = torch.tensor(x, dtype=torch.int32)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        x_mask = torch.tensor(x_mask, dtype=torch.int32)
        cls_headers = torch.tensor(cls_headers, dtype=torch.long)
        cls_headers_mask = torch.tensor(cls_headers_mask, dtype=torch.long)
        agg = torch.tensor(agg, dtype=torch.long)
        cond_conn_op = torch.tensor(cond_conn_op, dtype=torch.long)
        return x, attention_mask, x_mask, cls_headers, cls_headers_mask, agg, cond_conn_op, cond_cols, cond_ops

    def __len__(self):
        return len(self.data)


def collate_fn(batch_data):
    x, attention_mask, x_mask, cls_headers, cls_headers_mask, agg, cond_conn_op, cond_cols, cond_ops = zip(
        *batch_data)
    x = seq_padding(x)
    attention_mask = seq_padding(attention_mask)
    x_mask = seq_padding(x_mask, max_len=x.shape[1])
    cls_headers = seq_padding(cls_headers)
    cls_headers_mask = seq_padding(cls_headers_mask)
    agg = seq_padding(agg)
    cond_conn_op = seq_padding(cond_conn_op)
    cond_cols = seq_padding(cond_cols, max_len=x.shape[1])
    cond_ops = seq_padding(cond_ops, max_len=x.shape[1])
    return x, attention_mask, x_mask, cls_headers, cls_headers_mask, agg, cond_conn_op, cond_cols, cond_ops


class Model(nn.Module):
    def __init__(self, hidden_size, agg_out, cond_conn_op_out, cond_cols_out, cond_ops_out) -> None:
        super().__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.agg_fnn = nn.Linear(hidden_size, agg_out)
        self.cond_conn_op_fnn = nn.Linear(hidden_size, cond_conn_op_out)
        self.cond_ops_fnn = nn.Linear(hidden_size, cond_ops_out)
        self.question_fnn = nn.Linear(hidden_size, 256)
        self.headers_fnn = nn.Linear(hidden_size, 256)
        self.similar_fnn = nn.Linear(256, 1)

    def forward(self, x, attention_mask, cls_headers, cls_headers_mask):
        x = x.to(device)
        attention_mask = attention_mask.to(device)
        cls_headers = cls_headers.to(device)
        cls_headers_mask = cls_headers_mask.to(device)
        bert_out = self.bert(input_ids=x, attention_mask=attention_mask)
        last_hidden_state = bert_out['last_hidden_state']
        pooler_output = bert_out['pooler_output']

        headers_encode = torch.gather(
            last_hidden_state,
            dim=1,
            index=cls_headers.unsqueeze(-1).expand(-1, -1, self.hidden_size))
        y_hat_cond_conn_op = self.cond_conn_op_fnn(pooler_output)
        y_hat_agg = self.agg_fnn(headers_encode)
        y_hat_cond_ops = self.cond_ops_fnn(last_hidden_state)

        last_hidden_state = last_hidden_state.unsqueeze(2)
        headers_encode = headers_encode.unsqueeze(1)
        y_hat_cond_cols = self.question_fnn(last_hidden_state) + self.headers_fnn(headers_encode)
        y_hat_cond_cols = torch.tanh(y_hat_cond_cols)
        y_hat_cond_cols = self.similar_fnn(y_hat_cond_cols)
        y_hat_cond_cols = y_hat_cond_cols[..., 0] - (1 - cls_headers_mask[:, None]) * 1e10
        return y_hat_agg, y_hat_cond_conn_op, y_hat_cond_cols, y_hat_cond_ops


def train_step(model: nn.Module, x, attention_mask, x_mask, cls_headers, cls_headers_mask, agg, cond_conn_op, cond_cols, cond_ops):
    model.train()
    optim.zero_grad()
    y_hat_agg, y_hat_cond_conn_op, y_hat_cond_cols, y_hat_cond_ops = model(
        x, attention_mask, cls_headers, cls_headers_mask)
    cm = torch.not_equal(cond_ops, len(OP) - 1).to(device)
    batch_size = len(x)
    x = x.to(device)
    cls_headers = cls_headers.to(device)
    cls_headers_mask = cls_headers_mask.to(device)
    x_mask = x_mask.to(device)
    agg = agg.to(device)
    cond_conn_op = cond_conn_op.to(device)
    cond_cols = cond_cols.to(device)
    cond_ops = cond_ops.to(device)
    # 计算agg的loss
    loss_agg = F.cross_entropy(y_hat_agg.view(-1, len(AGG)), agg.view(-1),
                               reduction='none').reshape(batch_size, -1)
    loss_agg = torch.sum(loss_agg * cls_headers_mask) / torch.sum(cls_headers_mask)
    # 计算条件连接符的loss
    loss_cond_conn_op = F.cross_entropy(y_hat_cond_conn_op, cond_conn_op.view(-1))
    # 计算条件值的loss
    loss_cond_cols = F.cross_entropy(y_hat_cond_cols.view(
        -1, y_hat_cond_cols.shape[-1]), cond_cols.view(-1), reduction='none').reshape(batch_size, -1)
    loss_cond_cols = torch.sum(loss_cond_cols * x_mask * cm) / torch.sum(x_mask * cm)
    # 计算条件运算符的loss
    loss_cond_ops = F.cross_entropy(y_hat_cond_ops.view(-1, len(OP)),
                                    cond_ops.view(-1), reduction='none').reshape(batch_size, -1)
    loss_cond_ops = torch.sum(loss_cond_ops * x_mask) / torch.sum(x_mask)
    total_loss = loss_agg + loss_cond_conn_op + loss_cond_ops + loss_cond_cols
    total_loss.backward()
    optim.step()
    return total_loss.item()


def train(model, train_data_loader, test_data_loader):
    model.to(device)
    for epoch in range(1, Config.epochs + 1):
        loss = 0
        for x, attention_mask, x_mask, cls_headers, cls_headers_mask, agg, cond_conn_op, cond_cols, cond_ops in train_data_loader:
            loss_step = train_step(model, x, attention_mask, x_mask, cls_headers,
                                   cls_headers_mask, agg, cond_conn_op, cond_cols, cond_ops)
            loss += loss_step
        train_acc = valid(model, train_data_loader)
        test_acc = valid(model, test_data_loader)
        log.info(
            f'epoch={epoch}, loss={loss / len(train_data_loader.dataset)} train_acc={train_acc:.4%}, test_acc={test_acc:.4%}')


def valid_one_sql(y_agg, y_hat_agg,
                  y_cond_conn_op, y_hat_cond_conn_op,
                  y_cond_cols, y_hat_cond_cols,
                  y_cond_ops, y_hat_cond_ops,
                  raw_data=None):
    y_hat_agg = torch.argmax(y_hat_agg, dim=1)
    y_hat_cond_conn_op = torch.argmax(y_hat_cond_conn_op)
    y_hat_cond_cols = torch.argmax(y_hat_cond_cols, dim=1)
    y_hat_cond_ops = torch.argmax(y_hat_cond_ops, dim=1)

    ok_agg = y_agg.equal(y_hat_agg)
    ok_cond_conn_op = y_cond_conn_op.equal(y_hat_cond_conn_op)
    ok_cond_cols = y_cond_cols.equal(y_hat_cond_cols)
    ok_cond_ops = y_cond_ops.equal(y_hat_cond_ops)
    log.info(
        f'ok_agg={ok_agg}, ok_cond_conn_op={ok_cond_conn_op}, ok_cond_ops={ok_cond_ops}, ok_cond_cols={ok_cond_cols}')
    return ok_agg, ok_cond_conn_op, ok_cond_ops, ok_cond_cols


@torch.no_grad
def valid(model, dataloader):
    model.eval()
    agg_cnt = 0
    cond_conn_op_cnt = 0
    cond_ops_cnt = 0
    cond_cols_cnt = 0
    total = 0
    for x, attention_mask, x_mask, cls_headers, cls_headers_mask, agg, cond_conn_op, cond_cols, cond_ops in dataloader:
        y_hat_agg, y_hat_cond_conn_op, y_hat_cond_cols, y_hat_cond_ops = model(
            x, attention_mask, cls_headers, cls_headers_mask)
        cls_headers = cls_headers.to(device)
        cls_headers_mask = cls_headers_mask.to(device)
        x_mask = x_mask.to(device)
        agg = agg.to(device)
        cond_conn_op = cond_conn_op.to(device)
        cond_cols = cond_cols.to(device)
        cond_ops = cond_ops.to(device)

        for i in range(len(x)):
            ok_agg, ok_cond_conn_op, ok_cond_ops, ok_cond_cols = valid_one_sql(
                torch.masked_select(agg[i], cls_headers_mask[i] == 1),
                torch.masked_select(y_hat_agg[i],
                cls_headers_mask[i].reshape(cls_headers_mask[i].shape[0], -1).expand(-1, y_hat_agg[i].shape[1]) == 1).reshape(-1, y_hat_agg[i].shape[1]),
                cond_conn_op[i],
                y_hat_cond_conn_op[i],
                torch.masked_select(cond_cols[i], x_mask[i] == 1),
                torch.masked_select(y_hat_cond_cols[i], x_mask[i].reshape(
                    x_mask[i].shape[0], -1).expand(-1, y_hat_cond_cols[i].shape[1]) == 1).reshape(-1, y_hat_cond_cols[i].shape[1]),
                torch.masked_select(cond_ops[i], x_mask[i] == 1),
                torch.masked_select(y_hat_cond_ops[i], x_mask[i].reshape(
                    x_mask[i].shape[0], -1).expand(-1, y_hat_cond_ops[i].shape[1]) == 1).reshape(-1, y_hat_cond_ops[i].shape[1]),
            )
            if ok_agg:
                agg_cnt += 1
            if ok_cond_conn_op:
                cond_conn_op_cnt += 1
            if ok_cond_ops:
                cond_ops_cnt += 1
            if ok_cond_cols:
                cond_cols_cnt += 1
            if ok_agg and ok_cond_conn_op and ok_cond_ops and ok_cond_cols:
                total += 1

    dataset_size = len(dataloader.dataset)
    log.info(f'agg_acc={agg_cnt}/{dataset_size}, cond_conn_op_acc={cond_conn_op_cnt}/{dataset_size}, cond_ops_acc={cond_ops_cnt}/{dataset_size}, cond_cols_acc={cond_cols_cnt}/{dataset_size}, total_acc={total}/{len(dataloader.dataset)}')
    log.info(f'agg_acc={agg_cnt / dataset_size:.4%}, cond_conn_op_acc={cond_conn_op_cnt / dataset_size:.4%}, cond_ops_acc={cond_ops_cnt / dataset_size:.4%}, cond_cols_acc={cond_cols_cnt / dataset_size:.4%}, total_acc={total / len(dataloader.dataset):.4%}')
    return total / len(dataloader.dataset)


def main():
    train_dataset = TrainDataset('/data/yinxiaoln/datasets/TableQA/train/train.json',
                                 '/data/yinxiaoln/datasets/TableQA/train/train.tables.json',
                                 Config.max_len_bert,
                                 Config.max_cols,
                                 Config.max_question)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataset = TrainDataset('/data/yinxiaoln/datasets/TableQA/val/val.json',
                               '/data/yinxiaoln/datasets/TableQA/val/val.tables.json',
                               Config.max_len_bert,
                               Config.max_cols,
                               Config.max_question)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    model = Model(768, len(AGG), len(CONN), Config.max_cols, len(OP))
    # model = nn.DataParallel(model, device_ids=devices)
    global optim
    global scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
    train(model, train_dataloader, val_dataloader)


# a = tokenizer.encode("我是殷小龙")
# print(a)
main()

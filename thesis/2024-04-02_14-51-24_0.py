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

loss_func = nn.CrossEntropyLoss(reduction='sum')
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
        x, x_mask = [], []
        encoded_dict = tokenizer.encode_plus(d['question'])
        x.extend(encoded_dict['input_ids'])
        x_mask.extend(encoded_dict['attention_mask'])
        question_token_len = len(x)
        log.debug(tokenizer.convert_ids_to_tokens(x))
        cls_headers = []
        for header in headers:
            cls_headers.append(len(x))
            encoded_dict = tokenizer.encode_plus(header)
            x.extend(encoded_dict['input_ids'])
            x_mask.extend(encoded_dict['attention_mask'])

        # 给每一列添加agg标识
        sel = [len(AGG) - 1] * len(headers)
        for _sel, _op in zip(d['sql']['sel'], d['sql']['agg']):
            sel[_sel] = _op

        # 条件连接符
        cond_conn_op = [d['sql']['cond_conn_op']]
        # 条件列和值
        cond_cols = torch.zeros(question_token_len, dtype=torch.long) + len(headers)
        cond_ops = torch.zeros(question_token_len, dtype=torch.long) + len(OP) - 1

        for cond in d['sql']['conds']:
            if cond[2] not in d['question']:
                cond[2] = most_similar_2(cond[2], d['question'])
            if cond[2] not in d['question']:
                log.error('cond not in question')
                continue
            cond_token_ids = tokenizer.encode(cond[2])
            log.debug(tokenizer.convert_ids_to_tokens(cond_token_ids))

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

            log.debug(tokenizer.convert_ids_to_tokens(x[start: end + 1]))
            cond_cols[start: end + 1] = cond[0]
            cond_ops[start: end + 1] = cond[1]

        x = torch.tensor(x, dtype=torch.int32)
        x_mask = torch.tensor(x_mask, dtype=torch.int32)
        x = torch.cat([x, torch.zeros(self.max_len - len(x), dtype=torch.int32)], dim=0)
        x_mask = torch.cat([x_mask, torch.zeros(
            self.max_len - len(x_mask), dtype=torch.int32)], dim=0)
        sel = torch.tensor(sel, dtype=torch.long)
        cond_conn_op = torch.tensor(cond_conn_op, dtype=torch.long)
        return x, x_mask, question_token_len, cls_headers, sel, cond_conn_op, cond_cols, cond_ops, d

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    batch_x = []
    batch_x_mask = []
    batch_q = []
    batch_cls_headers = []
    batch_sel = []
    batch_cond_conn_op = []
    batch_cond_cols = []
    batch_cond_ops = []
    batch_data = []

    for d in data:
        batch_x.append(d[0])
        batch_x_mask.append(d[1])
        batch_q.append(d[2])
        batch_cls_headers.append(d[3])
        batch_sel.append(d[4])
        batch_cond_conn_op.append(d[5])
        batch_cond_cols.append(d[6])
        batch_cond_ops.append(d[7])
        batch_data.append(d[8])

    batch_x = torch.stack(batch_x, dim=0)
    batch_x_mask = torch.stack(batch_x_mask, dim=0)
    batch_sel = torch.cat(batch_sel, dim=0)
    batch_cond_conn_op = torch.cat(batch_cond_conn_op, dim=0)
    batch_cond_cols = torch.cat(batch_cond_cols, dim=0)
    batch_cond_ops = torch.cat(batch_cond_ops, dim=0)
    return batch_x, batch_x_mask, batch_q, batch_cls_headers, batch_sel, batch_cond_conn_op, batch_cond_cols, batch_cond_ops, batch_data


class Model(nn.Module):
    def __init__(self, hidden_size, agg_out, cond_conn_op_out, cond_cols_out, cond_ops_out) -> None:
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout()
        self.agg_fnn = nn.Linear(hidden_size, agg_out)
        self.cond_conn_op_fnn = nn.Linear(hidden_size, cond_conn_op_out)
        self.cond_cols_fnn = nn.Linear(hidden_size, cond_cols_out)
        self.cond_ops_fnn = nn.Linear(hidden_size, cond_ops_out)
        self.question_fnn = nn.Linear(hidden_size, 256)
        self.headers_fnn = nn.Linear(hidden_size, 256)
        self.similar_fnn = nn.Linear(256, 1)

    def forward(self, x, mask, cls_headers, question_len):
        x = x.to(device)
        mask = mask.to(device)
        bert_out = self.bert(input_ids=x, attention_mask=mask)
        last_hidden_state = bert_out['last_hidden_state']
        pooler_output = bert_out['pooler_output']
        y_agg = []
        y_cond_cols = []
        y_cond_ops = []
        batches = len(x)
        for i in range(batches):
            y_agg.append(self.agg_fnn(last_hidden_state[i][cls_headers[i], :]))

        y_cond_conn_op = self.cond_conn_op_fnn(pooler_output)

        for i in range(batches):
            question_encode = last_hidden_state[i][0: question_len[i], :]
            y_cond_ops.append(self.cond_ops_fnn(question_encode))

        for i in range(batches):
            question_encode = last_hidden_state[i][0: question_len[i], :]
            headers_encode = last_hidden_state[i][cls_headers[i], :]
            question_encode_256 = self.question_fnn(question_encode)
            headers_encode_256 = self.headers_fnn(headers_encode)
            question_encode_256 = torch.unsqueeze(question_encode_256, dim=1)
            headers_encode_256 = torch.unsqueeze(headers_encode_256, dim=0)
            cat_q_h = question_encode_256 + headers_encode_256
            cat_q_h = self.similar_fnn(cat_q_h)
            cat_q_h = torch.squeeze(cat_q_h, dim=-1)
            padding = torch.zeros([cat_q_h.shape[0], Config.max_cols - cat_q_h.shape[1]]) - 1e9
            padding = padding.to(device)
            cat_q_h = torch.cat([cat_q_h, padding], dim=1)
            y_cond_cols.append(cat_q_h)

        y_agg = torch.cat(y_agg, dim=0)
        y_cond_cols = torch.cat(y_cond_cols, dim=0)
        y_cond_ops = torch.cat(y_cond_ops, dim=0)
        return y_agg, y_cond_conn_op, y_cond_cols, y_cond_ops


def train_step(model: nn.Module, x, x_mask, question_len, cls_headers, y_agg, y_cond_conn_op, y_cond_cols, y_cond_ops):
    model.train()
    optim.zero_grad()
    y_hat_agg, y_hat_cond_conn_op, y_hat_cond_cols, y_hat_cond_ops = model(
        x, x_mask, cls_headers, question_len)
    # 计算agg的loss
    y_agg = y_agg.reshape(-1).to(device)
    loss_agg = loss_func(y_hat_agg, y_agg)
    # 计算条件连接符的loss
    y_cond_conn_op = y_cond_conn_op.reshape(-1).to(device)
    loss_cond_conn_op = loss_func(y_hat_cond_conn_op, y_cond_conn_op)
    # 计算条件值的loss
    y_cond_cols = y_cond_cols.reshape(-1).to(device)
    y_hat_cond_cols = y_hat_cond_cols.reshape(-1, y_hat_cond_cols.shape[1])
    loss_cond_cols = loss_func(y_hat_cond_cols, y_cond_cols)
    # 计算条件运算符的loss
    y_cond_ops = y_cond_ops.reshape(-1).to(device)
    y_hat_cond_ops = y_hat_cond_ops.reshape(-1, y_hat_cond_ops.shape[1])
    loss_cond_ops = loss_func(y_hat_cond_ops, y_cond_ops)

    total_loss = loss_agg + loss_cond_conn_op + loss_cond_ops + loss_cond_cols
    total_loss.backward()
    optim.step()
    return total_loss.item()


def train(model, train_data_loader, test_data_loader):
    model.to(device)
    for epoch in range(1, Config.epochs + 1):
        loss = 0
        # x, x_mask, question_token_len, cls_headers, sel, cond_conn_op, cond_cols, cond_ops, d
        for (x, x_mask, question_len, cls_headers, sel, cond_conn_op, cond_cols, cond_ops, _) in train_data_loader:
            loss_step = train_step(model, x, x_mask, question_len, cls_headers,
                                   sel, cond_conn_op, cond_cols, cond_ops)
            loss += loss_step
        if epoch == 2:
            pass
        train_acc = valid(model, train_data_loader)
        test_acc = valid(model, test_data_loader)
        log.info(f'epoch={epoch}, loss={loss / len(train_data_loader.dataset)}')
        log.info(f'train_acc={train_acc:.4%}, test_acc={test_acc:.4%}')


def valid_one_sql(y_agg, y_hat_agg,
                  y_cond_conn_op, y_hat_cond_conn_op,
                  y_cond_cols, y_hat_cond_cols,
                  y_cond_ops, y_hat_cond_ops,
                  raw_data=None):
    y_hat_agg = torch.argmax(y_hat_agg, dim=1)
    y_hat_cond_conn_op = torch.argmax(y_hat_cond_conn_op, dim=1)
    y_hat_cond_cols = torch.argmax(y_hat_cond_cols, dim=1)
    y_hat_cond_ops = torch.argmax(y_hat_cond_ops, dim=1)

    ok_agg = y_agg.equal(y_hat_agg)
    ok_cond_conn_op = y_cond_conn_op.equal(y_hat_cond_conn_op)
    ok_cond_cols = y_cond_cols.equal(y_hat_cond_cols)
    ok_cond_ops = y_cond_ops.equal(y_hat_cond_ops)
    log.info(y_agg)
    log.info(y_hat_agg)
    log.info(f'ok_agg={ok_agg}, ok_cond_conn_op={ok_cond_conn_op}, ok_cond_ops={ok_cond_ops}, ok_cond_cols={ok_cond_cols}')
    return ok_agg and ok_cond_conn_op and ok_cond_ops and ok_cond_cols


@torch.no_grad
def valid(model, dataloader):
    model.eval()
    right = 0
    for (x, x_mask, question_len, cls_headers, agg, cond_conn_op, cond_cols, cond_ops, data) in dataloader:
        y_hat_agg, y_hat_cond_conn_op, y_hat_cond_cols, y_hat_cond_ops = model(
            x, x_mask, cls_headers, question_len)
        agg = agg.reshape(-1).to(device)
        cond_conn_op = cond_conn_op.reshape(-1).to(device)
        y_hat_cond_conn_op = y_hat_cond_conn_op.reshape(-1, y_hat_cond_conn_op.shape[1])
        cond_cols = cond_cols.reshape(-1).to(device)
        y_hat_cond_cols = y_hat_cond_cols.reshape(-1, y_hat_cond_cols.shape[1])
        cond_ops = cond_ops.reshape(-1).to(device)
        y_hat_cond_ops = y_hat_cond_ops.reshape(-1, y_hat_cond_ops.shape[1])
        idx_agg = 0
        idx_cond_cols = 0
        idx_cond_ops = 0
        for i in range(len(x)):
            res = valid_one_sql(
                agg[idx_agg: idx_agg + len(cls_headers[i])],
                y_hat_agg[idx_agg: idx_agg + len(cls_headers[i]), :],
                cond_conn_op[i: i + 1],
                y_hat_cond_conn_op[i: i + 1, :],
                cond_cols[idx_cond_cols + 1: idx_cond_cols + question_len[i] - 1],
                y_hat_cond_cols[idx_cond_cols + 1: idx_cond_cols + question_len[i] - 1, :],
                cond_ops[idx_cond_ops + 1: idx_cond_ops + question_len[i] - 1],
                y_hat_cond_ops[idx_cond_ops + 1: idx_cond_ops + question_len[i] - 1, :],
                data[i]
            )
            if res:
                right += 1
            idx_agg += len(cls_headers[i])
            idx_cond_cols += question_len[i] + 2
            idx_cond_ops += question_len[i]
    log.info(f'{right}/{len(dataloader.dataset)}')
    return right / len(dataloader.dataset)


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
    optim = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    train(model, train_dataloader, val_dataloader)


main()

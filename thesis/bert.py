import sys
import json
import re
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
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import codecs

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
    device = torch.device(DEVICE[1])
    devices = [torch.device(d) for d in DEVICE]

    max_len_bert = 512
    max_cols = 25
    max_question = 128
    checkpoint_path = '/data/data1/checkpoint'


OP = {0: '>', 1: '<', 2: '==', 3: '!=', 4: '<=', 5: '>=', 6: 'None'}
#OP = {0: '>', 1: '<', 2: '==', 3: '!=', 4: 'None'}
AGG = {0: '', 1: 'AVG', 2: 'MAX', 3: 'MIN', 4: 'COUNT', 5: 'SUM', 6: 'None'}
CONN = {0: '', 1: 'and', 2: 'or'}

loss_func = nn.CrossEntropyLoss(reduction='mean')
optim = None
scheduler = None
device = Config.device


def read_data1(data_file, table_file):
    data, tables = [], {}
    with open(data_file, 'r', encoding='utf-8') as f:
        for l in f:
            data.append(json.loads(l))
    with open(table_file, 'r', encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = {}
            d['headers'] = l['header']
            d['header2id'] = {j: i for i, j in enumerate(d['headers'])}
            d['content'] = {}
            d['all_values'] = set()
            rows = np.array(l['rows'])
            for i, h in enumerate(d['headers']):
                d['content'][h] = set(rows[:, i])
                d['all_values'].update(d['content'][h])
            d['all_values'] = set([i for i in d['all_values'] if hasattr(i, '__len__')])
            tables[l['id']] = d
    return data, tables


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


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """带warmup的schedule, 源自transformers包optimization.py中
    
    :param num_warmup_steps: 需要warmup的步数, 一般为 num_training_steps * warmup_proportion(warmup的比例, 建议0.05-0.15)
    :param num_training_steps: 总的训练步数, 一般为 train_batches * num_epoch
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

datadir = '/data/yinxiaoln/datasets/TableQA'
valid_data, valid_table = read_data1(f'{datadir}/val/val.json', f'{datadir}/val/val.tables.json')
test_data, test_table = read_data1(f'{datadir}/test/test.json', f'{datadir}/test/test.tables.json')


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
        self.conn = nn.Linear(hidden_size, 3)
        self.agg = nn.Linear(hidden_size, len(AGG))
        self.op = nn.Linear(hidden_size, len(OP))
        self.dense1 = nn.Linear(hidden_size, 256)
        self.dense2 = nn.Linear(hidden_size, 256)
        self.dense3 = nn.Linear(256, 1)

    def forward(self, x1, attention_mask, h, hm):
        x1 = x1.to(device)
        attention_mask = attention_mask.to(device)
        h = h.to(device)
        hm = hm.to(device)
        x = self.bert(x1, attention_mask)
        x = x['last_hidden_state']

        # cls判断条件连接符 {0:"", 1:"and", 2:"or"}
        x4conn = x[:, 0]  # [cls位]
        pconn = self.conn(x4conn)  # [btz, num_cond_conn_op]

        # 列的cls位用来判断列名的agg和是否被select {0:"", 1:"AVG", 2:"MAX", 3:"MIN", 4:"COUNT", 5:"SUM", 6:"不被select"}
        x4h = torch.gather(x, dim=1, index=h.unsqueeze(-1).expand(-1, -1, 768)
                           )  # [btz, col_len, hdsz]
        psel = self.agg(x4h)  # [btz, col_len, num_agg]

        # 序列标注conds的值和运算符
        pcop = self.op(x)  # [btz, seq_len, num_op]
        x = x.unsqueeze(2)  # [btz, seq_len, 1, hdsz]
        x4h = x4h.unsqueeze(1)  # [btz, 1, col_len, hdsz]

        pcsel_1 = self.dense1(x)  # [btz, seq_len, 1, 256]
        pcsel_2 = self.dense2(x4h)  # [btz, 1, col_len, 256]
        pcsel = pcsel_1 + pcsel_2
        pcsel = torch.tanh(pcsel)
        pcsel = self.dense3(pcsel)  # [btz, seq_len, col_len, 1]
        pcsel = pcsel[..., 0] - (1 - hm[:, None]) * 1e10  # [btz, seq_len, col_len]
        return psel, pconn, pcop, pcsel


def train_step(model: nn.Module, x, attention_mask, x_mask, cls_headers, cls_headers_mask, agg, cond_conn_op, cond_cols, cond_ops):
    model.train()
    optim.zero_grad()
    psel, pconn, pcop, pcsel = model(
        x, attention_mask, cls_headers, cls_headers_mask)
    sel_in, conn_in, csel_in, cop_in, xm, hm = agg, cond_conn_op, cond_cols, cond_ops, x_mask, cls_headers_mask
    cm = torch.not_equal(cop_in, len(OP) - 1)
    cm = cm.to(device)
    sel_in = sel_in.to(device)
    conn_in = conn_in.to(device)
    csel_in = csel_in.to(device)
    cop_in = cop_in.to(device)
    xm = xm.to(device)
    hm = hm.to(device)
    batch_size = psel.shape[0]
    psel_loss = F.cross_entropy(psel.view(-1, len(OP)), sel_in.view(-1),
                                    reduction='none').reshape(batch_size, -1)
    psel_loss = torch.sum(psel_loss * hm) / torch.sum(hm)
    pconn_loss = F.cross_entropy(pconn, conn_in.view(-1))
    pcop_loss = F.cross_entropy(pcop.view(-1, len(OP)), cop_in.view(-1),
                                    reduction='none').reshape(batch_size, -1)
    pcop_loss = torch.sum(pcop_loss * xm) / torch.sum(xm)
    pcsel_loss = F.cross_entropy(
        pcsel.view(-1, pcsel.shape[-1]), csel_in.view(-1), reduction='none').reshape(batch_size, -1)
    pcsel_loss = torch.sum(pcsel_loss * xm * cm) / torch.sum(xm * cm)
    loss = psel_loss + pconn_loss + pcop_loss + pcsel_loss
    loss.backward()
    optim.step()
    return loss.item()


def train(model, train_data_loader, test_data_loader):
    model.to(device)
    for epoch in range(1, Config.epochs + 1):
        loss = 0
        for x, attention_mask, x_mask, cls_headers, cls_headers_mask, agg, cond_conn_op, cond_cols, cond_ops in train_data_loader:
            loss_step = train_step(model, x, attention_mask, x_mask, cls_headers, cls_headers_mask, agg, cond_conn_op, cond_cols, cond_ops)
            loss += loss_step
        #train_acc = valid(model, train_data_loader)
        #test_acc = valid(model, test_data_loader)
        train_acc = 0
        test_acc = evaluate(model, valid_data, valid_table)
        log.info(
            f'epoch={epoch}, loss={loss / len(train_data_loader.dataset)} train_acc={train_acc:.4%}, test_acc={test_acc:.4%}')
        scheduler.step()

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


@torch.no_grad
def nl2sql(model, question, table):
    """输入question和headers，转SQL
    """
    x1 = tokenizer.encode(question)
    h = []
    for i in table['headers']:
        _x1 = tokenizer.encode(i)
        h.append(len(x1))
        x1.extend(_x1)
    hm = [1] * len(h)
    attm = [1] * len(x1)
    psel, pconn, pcop, pcsel = model(
        torch.tensor([x1], dtype=torch.long, device=device),
        torch.tensor([attm], dtype=torch.long, device=device),
        torch.tensor([h], dtype=torch.long, device=device),
        torch.tensor([hm], dtype=torch.long, device=device))
    pconn, psel, pcop, pcsel = pconn.cpu().numpy(), psel.cpu(
    ).numpy(), pcop.cpu().numpy(), pcsel.cpu().numpy()
    R = {'agg': [], 'sel': []}
    for i, j in enumerate(psel[0].argmax(1)):
        if j != len(AGG) - 1:  # num_agg-1类是不被select的意思
            R['sel'].append(i)
            R['agg'].append(int(j))
    conds = []
    v_op = -1
    for i, j in enumerate(pcop[0, :len(question)+1].argmax(1)):
        # 这里结合标注和分类来预测条件
        if j != len(OP) - 1:
            if v_op != j:
                if v_op != -1:
                    v_end = v_start + len(v_str)
                    csel = pcsel[0][v_start: v_end].mean(0).argmax()
                    conds.append((csel, v_op, v_str))
                v_start = i
                v_op = j
                v_str = question[i - 1]
            else:
                v_str += question[i - 1]
        elif v_op != -1:
            v_end = v_start + len(v_str)
            csel = pcsel[0][v_start: v_end].mean(0).argmax()
            conds.append((csel, v_op, v_str))
            v_op = -1
    R['conds'] = set()
    for i, j, k in conds:
        if re.findall('[^\d\.]', k):
            j = 2  # 非数字只能用等号
        if j == 2:
            if k not in table['all_values']:
                # 等号的值必须在table出现过，否则找一个最相近的
                k = most_similar(k, list(table['all_values']))
            h = table['headers'][i]
            # 然后检查值对应的列是否正确，如果不正确，直接修正列名
            if k not in table['content'][h]:
                for r, v in table['content'].items():
                    if k in v:
                        i = table['header2id'][r]
                        break
        R['conds'].add((int(i), int(j), str(k)))
    R['conds'] = list(R['conds'])
    if len(R['conds']) <= 1:  # 条件数少于等于1时，条件连接符直接为0
        R['cond_conn_op'] = 0
    else:
        R['cond_conn_op'] = 1 + int(pconn[0, 1:].argmax())  # 不能是0
    return R


def is_equal(R1, R2):
    """判断两个SQL字典是否全匹配
    """
    return (R1['cond_conn_op'] == R2['cond_conn_op']) &\
        (set(zip(R1['sel'], R1['agg'])) == set(zip(R2['sel'], R2['agg']))) &\
        (set([tuple(i) for i in R1['conds']]) == set([tuple(i) for i in R2['conds']]))


def evaluate(model, data, tables):
        right = 0.
        pbar = tqdm()
        F = open('evaluate_pred.json', 'w', encoding='utf-8')
        for i, d in enumerate(data):
            question = d['question']
            table = tables[d['table_id']]
            R = nl2sql(model, question, table)
            right += float(is_equal(R, d['sql']))
            pbar.update(1)
            pbar.set_description('< acc: %.5f >' % (right / (i + 1)))
            d['sql_pred'] = R
            try:
                s = json.dumps(d, ensure_ascii=False, indent=4)
            except:
                continue
            F.write(s + '\n')
        F.close()
        pbar.close()
        return right / len(data)


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
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(optim, len(train_dataloader), len(train_dataloader) * 15)
    train(model, train_dataloader, val_dataloader)


# a = tokenizer.encode("我是殷小龙")
# print(a)
main()

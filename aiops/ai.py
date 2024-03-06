import os
import math
import json
import datetime
import inspect
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from loguru import logger as log
import aiops.config as config
from aiops.pre_process import build_train_dataset, build_test_dataset

path = '/data/yinxiaoln/datasets/aiops2023/processed/train_dataset/'


class AiOpsTrainDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.size = 1462944
        self.step = 100
        self.path = path
        self.seq_len = 256
        # self.data = build_train_dataset()
        self.data = [1]

    def __getitem__(self, index):
        data = self.data[index]
        data = torch.tensor(data, dtype=torch.float32)
        x = data[:, 0:32]
        y = data[:, 32:33]
        if len(data) < self.seq_len:
            x = torch.cat(
                [x, torch.zeros(self.seq_len - len(data), 32)], dim=0)
            y = torch.cat([y, torch.zeros(self.seq_len - len(data), 1)], dim=0)
        y = y.reshape(1, -1)
        p = self.seq_len - len(data)
        return x, y, p

    def __len__(self):
        return len(self.data)


class AiOpsTestDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.size = 1462944
        self.step = 100
        self.path = path
        self.seq_len = 256
        self.data = build_test_dataset()

    def __getitem__(self, index):
        data = self.data[index]
        fault_name = data[0]
        x = data[1]
        y = data[2]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if len(x) < self.seq_len:
            x = torch.cat([x, torch.zeros(self.seq_len - len(x), 32)], dim=0)
        y = y.reshape(1, -1)
        p = self.seq_len - len(x)
        return x, y, p, fault_name

    def __len__(self):
        return len(self.data)


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000) -> None:
        super(PositionEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = self.pe[:, :x.size(1)]
        return x


class Model(nn.Module):
    def __init__(self, d_model=32, max_len=1000) -> None:
        super(Model, self).__init__()
        self.pe = PositionEmbedding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=12)
        self.reg = nn.Linear(32, 1)

    def forward(self, x, mask):
        x = self.pe(x) + x
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.reg(x)
        x = x.reshape(len(x), 1, -1)
        return x

    @property
    def loss_func(self):
        return nn.MSELoss()

    @property
    def optim(self):
        return torch.optim.AdamW(self.parameters())


def collate_fn(data):
    x = [e[0] for e in data]
    y = [e[1] for e in data]
    return {'x': torch.stack(x), 'y': torch.stack[y]}


DEVICE = config.device[0]
device = torch.device(DEVICE)
date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def train_step(model: nn.Module, x, y, mask):
    model.train()
    model.optim.zero_grad()
    pred = model(x, mask)
    loss_func = model.loss_func(pred, y)
    loss_func.backward()
    model.optim.step()
    return loss_func.item()


@torch.no_grad
def valid(model, dataloader):
    model.eval()
    # return model.metric_func(dataloader)
    device = next(iter(model.parameters())).device
    for i, (x, y, p, fault_name) in enumerate(dataloader):
        ans = x[0][-1].tolist()
        for j in range(0, y.shape[2]):
            x = x.to(device)
            y = y.to(device)
            mask = cal_mask(p)
            mask = mask.to(device)
            pred = model(x, mask)
            ans.append(pred[0][0][-1].item())
            seq = ans[len(ans) - 32:len(ans)]
            seq = torch.tensor(seq, dtype=torch.float32)
            seq = seq.reshape(1, 1, -1)
            seq = seq.to(device)
            x = torch.cat((x, seq), dim=1)
            x = x[:, len(x[0]) - 256: len(x[0]), :]
            p = [max(p[0] - 1, 0)]
        pred = ans[len(ans) - y.shape[2]: len(ans)]
        pred = torch.tensor(pred, dtype=torch.float32)
        pred = pred.reshape(1, -1)
        pred = pred.to(device)
        cosine = F.cosine_similarity(y[0], pred)
        if cosine > 0.5:
            print(fault_name)


def cal_mask(p):
    masks = []
    for l in p:
        mask_p = torch.tensor([False] * (256 - l))
        mask_r = torch.tensor([True] * l)
        masks.append(torch.cat([mask_p, mask_r]))
    return torch.stack(masks)


def train(model, train_data_loader, test_data_loader):
    epochs = config.hyper_params["epochs"]
    model.to(device)
    for epoch in range(1, epochs + 1):
        loss = 0
        valid(model, test_data_loader)
        for (x, y, p) in train_data_loader:
            x = x.to(device)
            y = y.to(device)
            mask = cal_mask(p)
            mask = mask.to(device)
            l = train_step(model, x, y, mask)
            loss += l
            log.info(f'epoch={epoch}, loss={l}')
        log.info(f'epoch={epoch} loss={loss}')

        valid(model, test_data_loader)


def main():
    config_path = inspect.getfile(config)
    log.info(config)
    with open(config_path, 'r') as f:
        log.info(f.read())

    train_dataset = AiOpsTrainDataset(path)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=8,
        timeout=3600
    )

    test_dataset = AiOpsTestDataset(path)
    test_dataset[0]
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        timeout=3600
    )

    model = Model()
    train(model, train_dataloader, test_dataloader)


def parse_data():
    data = []
    files = os.listdir(path)
    for file in tqdm(files):
        file = os.path.join(path, file)
        with open(file, 'r', encoding='utf-8') as f:
            sub = json.load(f)
            data.extend(sub)
    return data


if __name__ == '__main__':
    main()
    # parse_data()

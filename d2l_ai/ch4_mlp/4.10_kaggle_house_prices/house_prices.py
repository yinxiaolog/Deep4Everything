import datetime

import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from d2l import torch as d2l

import config


DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
train_url = DATA_URL + 'kaggle_house_pred_train.csv'
test_url = DATA_URL + 'kaggle_house_pred_test.csv'

train_data = pd.read_csv(d2l.download(url=train_url, sha1_hash='585e9cc93e70b39160e7921475f9bcd7d31219ce'))
test_data = pd.read_csv(d2l.download(url=test_url, sha1_hash='fa19780a7b011d9b009e8bff8e99922a8ee2eb90'))

print(train_data.shape)
print(test_data.shape)

print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True, dtype=int)
print(all_features.shape)

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)


class TrainDataset(Dataset):
    def __init__(self, data):
        self.features = data[0]
        self.labels = data[1]

    def __getitem__(self, item):
        return self.features[item], self.labels[item]

    def __len__(self):
        return self.features.shape[0]


class Model(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.lr = config.hyper_params["lr"]
        self.weight_decay = config.hyper_params["weight_decay"]
        self.net = nn.Sequential(
            nn.Linear(features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, features: torch.Tensor):
        return self.net(features)

    @property
    def loss_func(self):
        return nn.MSELoss()

    @property
    def optim(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def metric_func(self, dataloader):
        mse = torch.zeros((1, 1), device=torch.device('cpu'))
        for (features, labels) in dataloader:
            features = features.to(device=torch.device('cpu'))
            labels = labels.to(device=torch.device('cpu'))
            pred = self.net(features)
            pred = torch.clamp(pred, 1, float('inf'))
            loss = nn.MSELoss(reduction='sum')
            mse += loss(torch.log(pred), torch.log(labels))

        mse = torch.sqrt(mse / len(dataloader.dataset))
        return mse.item()


def train_step(model: nn.Module, features, labels):
    model.train()
    model.optim.zero_grad()
    pred = model(features)
    loss_func = model.loss_func(pred, labels)
    loss_func.backward()
    model.optim.step()
    return loss_func.item()


@torch.no_grad
def valid(model, dataloader):
    model.eval()
    return model.metric_func(dataloader)


def train(model, train_data_loader, test_data_loader):
    train_loss = []
    valid_loss = []
    for epoch in range(1, config.hyper_params["epochs"] + 1):
        for _, (features, labels) in enumerate(train_data_loader):
            features = features.to(device=torch.device('cpu'))
            labels = labels.to(device=torch.device('cpu'))
            train_step(model, features, labels)

        train_metric = valid(model, train_data_loader)

        for _, (features, labels) in enumerate(test_data_loader):
            features = features.to(device=torch.device('cpu'))
            labels = labels.to(device=torch.device('cpu'))
            train_step(model, features, labels)

        val_metric = valid(model, test_data_loader)
        if epoch % 10 == 0:
            #print(f'epoch={epoch} train_metric={train_metric} valid_metric={val_metric}')
            pass

        train_loss.append(train_metric)
        valid_loss.append(val_metric)

    return train_loss, valid_loss


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def train_k_fold():
    batch_size = config.hyper_params["batch_size"]
    k = 5
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        train_features_k, train_labels_k, valid_features, valid_labels = get_k_fold_data(k, i, train_features, train_labels)
        model = Model(train_features.shape[1])
        device = torch.device('cpu')
        model = model.to(device=device)
        train_dataset = TrainDataset((train_features_k, train_labels_k))
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)
        valid_dataset = TrainDataset((valid_features, valid_labels))
        valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size)
        train_ls, valid_ls = train(model, train_dataloader, valid_dataloader)

        print(f'折{i + 1}, 训练rmse={train_ls[-1]}, 验证rmse={valid_ls[-1]}')
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

    print(f'train_avg_rmse={train_l_sum / k}, valid_avg_rmse={valid_l_sum / k}')


def main():
    model = Model(train_features.shape[1])
    batch_size = config.hyper_params["batch_size"]
    train_dataset = TrainDataset((train_features, train_labels))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)

    train(model, train_dataloader, train_dataloader)
    preds = model(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    #main()
    print(datetime.datetime.now())
    train_k_fold()
    plt.show()
    print(datetime.datetime.now())
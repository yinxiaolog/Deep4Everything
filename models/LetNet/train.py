import datetime
import inspect

import torch
from torch import nn
from d2l import torch as d2l
import utils.common as common
from model import Model
import config
from log import log

LOG = log.Logger.get_logger()
DEVICE = config.device[0]
device = torch.device(DEVICE)
date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def train_step(model: nn.Module, features, labels):
    model.train()
    model.optim.zero_grad()
    pred = model(features)
    loss_func = model.loss_func(pred, labels)
    loss_func.backward()
    model.optim.step()
    acc = common.accuracy(pred, labels)
    return loss_func.item(), acc


@torch.no_grad
def valid(model, dataloader):
    model.eval()
    return model.metric_func(dataloader)


def train(model, train_data_loader, test_data_loader):
    epochs = config.hyper_params["epochs"]
    timer = d2l.Timer()
    model.to(device)
    metric = d2l.Accumulator(3)
    for epoch in range(1, epochs + 1):
        for i, (features, labels) in enumerate(train_data_loader):
            timer.start()
            features = features.to(device)
            labels = labels.to(device)
            loss, acc = train_step(model, features, labels)
            metric.add(loss * features.shape[0], acc, features.shape[0])
            timer.stop()

        val_metric = valid(model, test_data_loader)
        if epoch % config.step == 0:
            LOG.info(f'epoch={epoch} loss={(metric[0] / metric[2]):.5f}, train acc={(metric[1] / metric[2] * 100):.5f} test acc={val_metric * 100:.3f}')

    LOG.info(f'{metric[2] * epochs / timer.sum():.1f} examples/sec on {str(device)}')


def main():
    log.Logger.set(config, date)
    config_path = inspect.getfile(config)
    LOG.info(config)
    with open(config_path, 'r') as f:
        LOG.info(f.read())

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=config.hyper_params['batch_size'])
    model = Model()
    train(model, train_iter, test_iter)


if __name__ == '__main__':
    main()

import os

import torch
import yaml
from torch import nn

from d2l import torch as d2l
from log import logger
from models import LeNet5

logger = logger.Logger().get_logger()
base_dir = os.path.dirname((os.path.abspath(__file__)))
TRAIN = 'train'
config = None


def parse_yml_config(model_name):
    config_path = os.path.join(base_dir, 'modelsConfig', model_name + '.yml')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        logger.error(f'open config file error, file={config_path}')
        logger.error(e)

    return None


def evaluate_accuracy_gpu(model, data_iter, device=None):
    if isinstance(model, nn.Module):
        model.eval()
        if not device:
            device = next(iter(model.parameters())).device

    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(model(X), y), y.numel())

    return metric[0] / metric[1]


def train_epoch(epoch, model, timer, train_iter, optimizer, device, loss_fun, test_iter):
    # 训练损失之和，训练准确率之和，样本数
    metric = d2l.Accumulator(3)
    model.train()
    timer.start()
    for i, (X, y) in enumerate(train_iter):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        y_hat = model(X)
        loss = loss_fun(y_hat, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            metric.add(loss * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])

    timer.stop()
    train_loss = metric[0] / metric[2]
    train_acc = metric[1] / metric[2]
    test_acc = evaluate_accuracy_gpu(model, test_iter)
    logger.info(
        f'epoch={epoch + 1} time={timer.times[-1]:.2f}s train_loss={train_loss:.2f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}')


def train(model):
    """用GPU训练模型"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    batch_size = config[TRAIN]['batch_size']
    epochs = config[TRAIN]['epochs']
    device = config[TRAIN]['device']
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    model.apply(init_weights)
    logger.info(f'training on {device}')
    model.to(device)
    optimizer = model.optim()
    loss_fun = model.loss_func()
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        model.train()
        timer.start()
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = loss_fun(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])

        timer.stop()
        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(model, test_iter)
        logger.info(
            f'epoch={epoch + 1} time={timer.times[-1]:.2f}s train_loss={train_loss:.2f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}')

    logger.info(f'total time: {timer.sum():.1f}s')
    logger.info(f'{metric[2] * epochs / timer.sum():.1f} examples/sec on {str(device)}')


if __name__ == '__main__':
    config = parse_yml_config(LeNet5.LeNet5.__name__)
    logger.info(config)
    leNet5 = LeNet5.LeNet5(config)
    train(leNet5)

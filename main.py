import argparse
import datetime
import importlib
import json
import os

import torch
import yaml
from torch import nn

from d2l_ai import torch as d2l
from log import logger

LOG = logger.Logger().get_logger()
base_dir = os.path.dirname((os.path.abspath(__file__)))
TRAIN = 'train'
config = None
date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
checkpoint_path = None
model_name = None


def parse_yml_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    except Exception as e:
        LOG.error(f'open config file error, file={config_path}')
        LOG.error(e)

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


def save_checkpoint(epoch, net: nn.Module, optimizer, loss):
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    torch.save({
        'epoch': epoch,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_func': loss,
    }, checkpoint_path + '/' + str(epoch) + '.pt')


def train(model):
    """用GPU训练模型"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    total = sum([param.nelement() for param in model.parameters()])
    LOG.info(f'model: {model_name}, total_params={total / 1e6:.2f}M')
    batch_size = config[TRAIN]['batch_size']
    epochs = config[TRAIN]['epochs']
    device = config[TRAIN]['device']
    train_iter = None
    test_iter = None
    if model_name == 'LeNet5':
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    elif model_name == 'AlexNet' or model_name == 'VGG11' or model_name == 'NiN':
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    elif model_name == 'GoogLeNet' or model_name == 'ResNet':
        train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)

    model.apply(init_weights)
    model.to(device)
    optimizer = model.optim()
    loss_fun = model.loss_func()
    timer, num_batches = d2l.Timer(), len(train_iter)
    best_test_acc = 0.0
    best = ''
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
        mm, ss = divmod(int(timer.times[-1]), 60)
        hh, mm = divmod(mm, 60)
        LOG.info(
            f'epoch={epoch + 1}/{epochs} time={hh}h {mm}m {ss}s '
            f'train_loss={train_loss:.2f} train_acc={train_acc * 100:.3f}% test_acc={test_acc * 100:.3f}%')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best = f'epoch={epoch + 1} train_loss={train_loss:.2f} ' \
                   f'train_acc={train_acc * 100:.3f}% test_acc={test_acc * 100:.3f}%'
        save_checkpoint(epoch + 1, model, optimizer, loss_fun)
    mm, ss = divmod(int(timer.sum()), 60)
    hh, mm = divmod(mm, 60)
    LOG.info(f'best: {best}')
    LOG.info(f'total time: time={hh}h {mm}m {ss}s')
    LOG.info(f'{metric[2] * epochs / timer.sum():.1f} examples/sec on {str(device)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-conf', '--config', required=True, help='config file')
    args = parser.parse_args()
    print(args)
    config_file = args.config
    config = parse_yml_config(config_file)
    model_name = config['name']
    logger.Logger.set(model_name, date)
    checkpoint_path = os.path.join(config['checkpoint_path'], model_name, date)
    LOG.info('\n' + json.dumps(config, indent=4))
    module = importlib.import_module('models.' + model_name)
    model = getattr(module, 'Train')
    train = model(config)
    train.train_bert()

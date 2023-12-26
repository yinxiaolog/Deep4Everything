import re
from collections import defaultdict

import torch
import torch.nn.functional as F
from d2l_ai import torch as d2l
from torch import nn

from log import logger
from mk_dataset import hdfs_log_dataset
from models.base_model import bert_encoder
from utils.constant import *

LOG = logger.Logger().get_logger()


class BertModel(nn.Module):
    def __init__(self, net_parameters):
        super(BertModel, self).__init__()
        self.encoder = bert_encoder.BertEncoder(net_parameters)
        self.vocab_size = net_parameters[VOCAB_SIZE]
        self.mlm_in_features = net_parameters[EMBEDDING_DIM]
        self.mlm_hidden = net_parameters[MLM_HIDDEN]
        self.nsp_hidden = net_parameters[NSP_HIDDEN]
        self.nsp_in_features = net_parameters[EMBEDDING_DIM]
        self.mlm = bert_encoder.Mlm(self.vocab_size, self.mlm_in_features, self.mlm_hidden)
        self.nsp = bert_encoder.Nsp(self.nsp_in_features, self.nsp_hidden)

    def forward(self, tokens, segments, valid_lens=None, positions=None):
        encoded_x = self.encoder(tokens, segments, valid_lens)
        if positions is not None:
            mlm_y_hat = self.mlm(encoded_x, positions)
        else:
            mlm_y_hat = None
        nsp_y_hat = self.nsp(encoded_x[:, 0, :])
        return encoded_x, mlm_y_hat, nsp_y_hat


class Train:
    def __init__(self, config):
        self.config = config
        self.batch_size = config[HYPER_PARAMS][BATCH_SIZE]
        self.max_len = config[HYPER_PARAMS][MAX_LEN]
        #self.train_iter, self.vocab = hdfs_log_dataset.get_hdfs_log_test_abnormal(config)
        self.train_iter, self.vocab = hdfs_log_dataset.get_hdfs_log_train_normal(config)
        #self.train_iter, self.vocab = hdfs_log_dataset.get_hdfs_log_test_normal(config)
        config[HYPER_PARAMS][VOCAB_SIZE] = len(self.vocab)
        LOG.error('net before')
        self.net = BertModel(config[HYPER_PARAMS])
        LOG.error('net after')
        self.devices = config[HYPER_PARAMS][DEVICE]
        self.epochs = config[HYPER_PARAMS][EPOCHS]

    def calculate_loss(self, tokens_x, segments_x, valid_lens_x, positions, mlm_weights_x, mlm_y, nsp_y):
        loss_func = nn.CrossEntropyLoss()
        _, mlm_y_hat, nsp_y_hat = self.net(tokens_x, segments_x, valid_lens_x.reshape(-1), positions)
        vocab_size = len(self.vocab)
        mlm_loss = loss_func(mlm_y_hat.reshape(-1, vocab_size), mlm_y.reshape(-1)) * mlm_weights_x.reshape(-1, 1)
        mlm_loss = mlm_loss.sum() / (mlm_weights_x.sum() + 1e-8)
        nsp_loss = loss_func(nsp_y_hat, nsp_y)
        return mlm_loss, nsp_loss, mlm_loss + nsp_loss

    def find_blk_id(self, token_ids, vocab):
        for token_id in token_ids:
            blk_ids = re.findall(r'(blk_-?\d+)', vocab.idx_to_token[token_id])
            if len(blk_ids) > 0:
                return blk_ids[0]
        return ''

    def evaluate(self, net: nn.Module, dataset_iter, vocab):
        top_g = 5
        r = 3
        devices = self.devices
        net.eval()
        log_seq_anomaly_count = defaultdict(int)
        with torch.no_grad():
            for tokens_x, segments_x, valid_lens_x, pred_positions_x, mlm_weights_x, mlm_Y, nsp_y in dataset_iter:
                tokens_x = tokens_x.to(devices[0])
                segments_x = segments_x.to(devices[0])
                valid_lens_x = valid_lens_x.to(devices[0])
                pred_positions_x = pred_positions_x.to(devices[0])
                mlm_y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
                _, mlm_y_hat, nsp_y_hat = net(tokens_x, segments_x, valid_lens_x.reshape(-1), pred_positions_x)
                mlm_y_hat = F.softmax(mlm_y_hat, dim=2)
                top_g_idx = torch.topk(mlm_y_hat, k=top_g, dim=-1, sorted=True)
                top_g_idx = top_g_idx[1]
                for i in range(len(top_g_idx)):
                    blk_id = self.find_blk_id(tokens_x[i], vocab)
                    if blk_id == '':
                        continue
                    for j in range(len(top_g_idx[0])):
                        if mlm_y[i][j] not in top_g_idx[i][j]:
                            log_seq_anomaly_count[blk_id] += 1

        true_count, false_count = 0, 0
        for _, v in log_seq_anomaly_count.items():
            if v >= r:
                false_count += 1
            else:
                true_count += 1
        return true_count, false_count

    def eval(self, net: nn.Module):
        data_iter, vocab = hdfs_log_dataset.get_hdfs_log_train_normal(self.config)
        tp, fn = self.evaluate(net, data_iter, vocab)
        data_iter, vocab = hdfs_log_dataset.get_hdfs_log_test_normal(self.config)
        fp, tn = self.evaluate(net, data_iter, vocab)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        LOG.info(f'normal: precision={precision} recall={recall} f1={f1}')

    def train_bert(self):
        net = self.net
        train_iter = self.train_iter
        devices = self.devices
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
        trainer = torch.optim.Adam(net.parameters(), lr=0.01)
        step, timer = 0, d2l.Timer()
        animator = d2l.Animator(xlabel='step', ylabel='loss',
                                xlim=[1, self.epochs], legend=['mlm', 'nsp'])
        # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
        metric = d2l.Accumulator(4)
        num_steps_reached = False
        LOG.error("train_bert error")
        while step < self.epochs and not num_steps_reached:
            for tokens_x, segments_x, valid_lens_x, pred_positions_x, mlm_weights_x, mlm_Y, nsp_y in train_iter:
                tokens_x = tokens_x.to(devices[0])
                segments_x = segments_x.to(devices[0])
                valid_lens_x = valid_lens_x.to(devices[0])
                pred_positions_x = pred_positions_x.to(devices[0])
                mlm_weights_x = mlm_weights_x.to(devices[0])
                mlm_y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
                trainer.zero_grad()
                timer.start()
                mlm_loss, nsp_loss, loss = self.calculate_loss(
                    tokens_x, segments_x, valid_lens_x,
                    pred_positions_x, mlm_weights_x, mlm_y, nsp_y)
                loss.backward()
                trainer.step()
                metric.add(mlm_loss, nsp_loss, tokens_x.shape[0], 1)
                timer.stop()
                animator.add(step + 1, (metric[0] / metric[3], metric[1] / metric[3]))
                step += 1
                net.train()
                self.eval(net)
                if step == self.epochs:
                    num_steps_reached = True
                    break

                LOG.info(f'MLM loss {metric[0] / metric[3]:.3f}, NSP loss {metric[1] / metric[3]:.3f}')
                LOG.info(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on {str(devices)}')
        d2l.plt.show()

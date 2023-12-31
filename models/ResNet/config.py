name = 'ResNet'

hyper_params = {
    'batch_size': 512,
    'epochs': 300,
    'lr': 0.2,
    'weight_decay': 0
}
device = ['cuda:1']
step = 1
log = {
    'log_base_dir': '/data/yinxiaoln/log/',
    'desc': 'file/console',
    'file': 'log.txt'
}
data = {}

model = {}

checkpoint_path = '/data/data1/checkpoint'

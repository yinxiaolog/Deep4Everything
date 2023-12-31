"""aiops dataset pre processing"""
import os
import csv
import json
import datetime
import matplotlib.pyplot as plt


DATA_PATH = '/data/yinxiaoln/code/Deep4Everything/datasets/aiops2023'
fault_data = os.path.join(DATA_PATH, 'fault_data')
new_fault_data = os.path.join(DATA_PATH, 'new_fault_data')
no_fault_data = os.path.join(DATA_PATH, 'no_fault_data')


class Metric:
    def __init__(self, cmdb_id, kpi_name, device, value, timestamp):
        self.cmdb_id = cmdb_id
        self.kpi_name = kpi_name
        self.value = value
        self.device = device
        self.timestamp = timestamp

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class Transaction:
    def __init__(self, tran_code,
                 timestamp,
                 amount,
                 bus_success_rate,
                 sys_success_rate,
                 avg_rsp_time,
                 stall_amount,
                 avg_proc_time,
                 stall_rate,
                 apdex):
        self.apdex = apdex
        self.stall_rate = stall_rate
        self.avg_proc_time = avg_proc_time
        self.stall_amount = stall_amount
        self.avg_rsp_time = avg_rsp_time
        self.sys_success_rate = sys_success_rate
        self.bus_success_rate = bus_success_rate
        self.amount = amount
        self.timestamp = timestamp
        self.tran_code = tran_code

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class MetricEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


def metric_decoder(dct):
    if 'cmdb_id' in dct:
        return Metric(dct['cmdb_id'],
                      dct['kpi_name'],
                      dct['device'],
                      dct['value'],
                      dct['timestamp'])
    return dct


class TransactionEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


def transaction_decoder(dct):
    if 'tran_code' in dct:
        return Transaction(
            dct['tran_code'],
            dct['timestamp'],
            dct['amount'],
            dct['bus_success_rate'],
            dct['sys_success_rate'],
            dct['avg_rsp_time'],
            dct['stall_amount'],
            dct['avg_proc_time'],
            dct['stall_rate'],
            dct['apdex']
        )
    return dct


def read_monitor():
    monitor = 'monitor'
    # monitor_path = [os.path.join(fault_data, monitor),
    #                 os.path.join(new_fault_data, monitor),
    #                 os.path.join(no_fault_data, monitor)]
    monitor_path = [os.path.join(fault_data, monitor)]

    linux_metrics = {}
    if os.path.isfile('/data/yinxiaoln/code/Deep4Everything/aiops/linux_metrics.json'):
        with open('/data/yinxiaoln/code/Deep4Everything/aiops/linux_metrics.json',
                  'r', encoding='utf-8') as f:
            json_str = f.read()
            linux_metrics = json.loads(json_str, object_hook=metric_decoder)
            return linux_metrics
    data_size = 0
    for path in monitor_path:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and 'csv' in file:
                size = read_one_monitor_csv(linux_metrics, file_path)
                data_size += size
                print(file_path, size, data_size)

    cnt = 0
    for cmdb_id in linux_metrics:
        for kpi_name in linux_metrics[cmdb_id]:
            for device in linux_metrics[cmdb_id][kpi_name]:
                sorted(linux_metrics[cmdb_id][kpi_name][device])
                cnt += len(linux_metrics[cmdb_id][kpi_name][device])

    assert cnt == data_size
    json_str = json.dumps(linux_metrics, indent=4, cls=MetricEncoder)
    with open('/data/yinxiaoln/code/Deep4Everything/aiops/linux_metrics.json', 'w', encoding='utf-8') as f:
        f.write(json_str)

    return linux_metrics


def read_one_monitor_csv(linux_metrics, path):
    cnt = 0
    with open(path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 去掉标题
        next(reader)
        for row in reader:
            timestamp = row[0]
            cmdb_id = row[1]
            kpi_name = row[2]
            value = row[3]
            device = row[4]
            if isinstance(timestamp, str):
                timestamp = int(timestamp)
            if isinstance(value, str):
                value = float(value)
            if len(device) == 0:
                device = 'unknown'
            assert len(row) == 5
            metric = Metric(cmdb_id, kpi_name, device, value, timestamp)
            add_one_metric(linux_metrics, metric)
            cnt += 1

    return cnt


def add_one_metric(linux_metrics: {}, metric: Metric):
    timestamp = metric.timestamp
    cmdb_id = metric.cmdb_id
    kpi_name = metric.kpi_name
    value = metric.value
    device = metric.device
    metric = Metric(cmdb_id, kpi_name, device, value, timestamp)
    if cmdb_id not in linux_metrics:
        linux_metrics[cmdb_id] = {}

    if kpi_name not in linux_metrics[cmdb_id]:
        linux_metrics[cmdb_id][kpi_name] = {}

    if device not in linux_metrics[cmdb_id][kpi_name]:
        linux_metrics[cmdb_id][kpi_name][device] = []

    linux_metrics[cmdb_id][kpi_name][device].append(metric)


def read_tc():
    tc = 'tc'
    tc_path = [os.path.join(fault_data, tc),
               os.path.join(new_fault_data, tc),
               os.path.join(no_fault_data, tc)]

    tc_metrics = []
    if os.path.isfile('/data/yinxiaoln/code/Deep4Everything/aiops/tc.json'):
        with open('/data/yinxiaoln/code/Deep4Everything/aiops/tc.json', 'r', encoding='utf-8') as f:
            json_str = f.read()
            tc_metrics = json.loads(json_str, object_hook=transaction_decoder)
            return tc_metrics
    for path in tc_path:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and 'csv' in file:
                with open(file_path, mode='r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    # 去掉标题
                    next(reader)
                    for row in reader:
                        tran = Transaction(row[0], int(row[1]), int(row[2]), float(row[3]), float(row[4]),
                                           float(row[5]),
                                           int(row[6]), float(
                                               row[7]), float(row[8]),
                                           float(row[9]))
                        tc_metrics.append(tran)

    json_str = json.dumps(tc_metrics, indent=4, cls=TransactionEncoder)
    with open('tc.json', 'w', encoding='utf-8') as f:
        f.write(json_str)
    return tc_metrics


def plot_linux_metric(start: int, end: int):
    linux_metrics = read_monitor()
    print('get linux_m')
    for cmdb_id in linux_metrics:
        for kpi_name in linux_metrics[cmdb_id]:
            for device in linux_metrics[cmdb_id][kpi_name]:
                metrics = linux_metrics[cmdb_id][kpi_name][device]
                x = []
                y = []
                for metric in metrics:
                    print(metric)
                    ts = metric.timestamp
                    if start <= ts <= end:
                        x.append(datetime.datetime.fromtimestamp(metric.timestamp))
                        y.append(metric.value)

                if len(x) > 0:
                    plt.figure(figsize=(40, 20))
                    plt.plot(x, y)
                    plt.savefig(f'{cmdb_id}-{kpi_name}-{device}.pdf')
                    plt.show()
                    print('end')


#read_monitor()
plot_linux_metric(1694620860, 1694620860 + 3600 * 24 * 7)

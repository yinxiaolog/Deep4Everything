import os
import csv

data_path = '/data/yinxiaoln/code/Deep4Everything/datasets/aiops2023'
fault_data = os.path.join(data_path, 'fault_data')
new_fault_data = os.path.join(data_path, 'new_fault_data')
no_fault_data = os.path.join(data_path, 'no_fault_data')


class Metric:
    def __init__(self, cmdb_id, kpi_name, device, value, timestamp):
        self.cmdb_id = cmdb_id
        self.kpi_name = kpi_name
        self.value = value
        self.device = device
        self.timestamp = timestamp


def read_monitor():
    monitor = 'monitor'
    monitor_path = [os.path.join(fault_data, monitor),
                    os.path.join(new_fault_data, monitor),
                    os.path.join(no_fault_data, monitor)]

    linux_metrics = {}
    for path in monitor_path:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and 'csv' in file:
                read_one_monitor_csv(linux_metrics, path)

    print("a")


def read_one_monitor_csv(linux_metrics, path):
    with open(path, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 去掉标题
        next(reader)
        for row in reader:
            timestamp = row[0]
            cmdb_id = row[1]
            kpi_name = row[2]
            value = row[3]
            print(len(row), row)
            device = row[4]
            if isinstance(timestamp, str):
                timestamp = int(timestamp)
            if isinstance(value, float):
                value = float(value)
            assert len(row) == 5
            metric = Metric(cmdb_id, kpi_name, device, value, timestamp)
            add_one_metric(linux_metrics, metric)


def add_one_metric(linux_metrics: {}, metric: Metric):
    timestamp = metric.timestamp
    cmdb_id = metric.cmdb_id
    kpi_name = metric.kpi_name
    value = metric.value
    device = metric.device
    metric = Metric(cmdb_id, kpi_name, device, value, timestamp)
    if cmdb_id in linux_metrics:
        if kpi_name in linux_metrics[cmdb_id]:
            if device in linux_metrics[cmdb_id][kpi_name]:
                linux_metrics[cmdb_id][kpi_name][device].append(metric)
            else:
                linux_metrics[cmdb_id][kpi_name][device] = []
        else:
            linux_metrics[cmdb_id][kpi_name] = {}
    else:
        linux_metrics[cmdb_id] = {}


read_monitor()

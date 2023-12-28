import os
import csv
import json

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

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __str__(self):
        return  self.__dict__

    def __repr__(self):
        return self.__str__()


class MetricEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


def read_monitor():
    monitor = 'monitor'
    monitor_path = [os.path.join(fault_data, monitor),
                    os.path.join(new_fault_data, monitor),
                    os.path.join(no_fault_data, monitor)]

    linux_metrics = {}
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
    with open('linux_metrics.json', 'w') as f:
        f.write(json_str)


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
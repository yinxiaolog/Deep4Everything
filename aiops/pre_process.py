"""aiops dataset pre processing"""
import os
import csv
import json
import datetime
from tqdm import tqdm
from enum import Enum
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pymysql
import psycopg2
import pandas as pd
from loguru import logger as LOG

DATA_PATH = '/data/yinxiaoln/code/Deep4Everything/datasets/aiops2023'
SAVE_PATH = '/data/yinxiaoln/datasets/aiops2023/processed'
SAVE_MONITOR = os.path.join(SAVE_PATH, 'monitor')
SAVE_TC = os.path.join(SAVE_PATH, 'tc')
MONITOR = 'monitor'
TC = 'tc'

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_MONITOR, exist_ok=True)
os.makedirs(SAVE_TC, exist_ok=True)


class AiOpsEnum(Enum):
    FAULT_DATA = 'fault_data'
    FAULT_DATA_MONITOR = 'fault_data_monitor_%s' % MONITOR
    FAULT_DATA_TC = 'fault_data_monitor_%s' % TC
    NEW_FAULT_DATA = 'new_fault_data'
    NEW_FAULT_DATA_MONITOR = 'new_fault_data_%s' % MONITOR
    NEW_FAULT_DATA_TC = 'new_fault_data_%s' % TC
    NO_FAULT_DATA = 'no_fault_data'
    NO_FAULT_DATA_MONITOR = 'no_fault_data_%s' % MONITOR
    NO_FAULT_DATA_TC = 'no_fault_data_%s' % TC
    ALL_FAULT_DATA = 'all_fault_data'
    MONITOR_DATA = 'monitor'
    TC_DATA = 'tc'
    ALL = 'all'
    T_FAULT_MONITOR = 'fault_monitor'
    T_FAULT_TC = 'fault_tc'
    T_NEW_FAULT_MONITOR = 'new_fault_monitor'
    T_NEW_FAULT_TC = 'new_fault_tc'
    T_NO_FAULT_MONITOR = 'no_fault_monitor'
    T_NO_FAULT_TC = 'no_fault_tc'


fault_data_path = os.path.join(DATA_PATH, AiOpsEnum.FAULT_DATA.value)
new_fault_data_path = os.path.join(DATA_PATH, AiOpsEnum.NEW_FAULT_DATA.value)
no_fault_data_path = os.path.join(DATA_PATH, AiOpsEnum.NO_FAULT_DATA.value)

success_files = [

]


class DataPoint:
    def __init__(self, metric: str, tags: dict, timestamp: int, value: float):
        self.metric = metric
        self.tags = tags
        self.timestamp = timestamp
        self.value = value

    def __lt__(self, other):
        return self.timestamp < other.timestamp

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


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
                 timestamp: int,
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


class DataPointEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


def read_monitor(monitor_type=AiOpsEnum.ALL):
    monitor = 'monitor'
    monitor_path = []
    if monitor_type == AiOpsEnum.ALL:
        monitor_path = [os.path.join(fault_data_path, monitor),
                        os.path.join(new_fault_data_path, monitor),
                        os.path.join(no_fault_data_path, monitor)]
    elif monitor_type == AiOpsEnum.FAULT_DATA:
        monitor_path = [os.path.join(fault_data_path, monitor)]
    elif monitor_type == AiOpsEnum.NEW_FAULT_DATA:
        monitor_path = [os.path.join(new_fault_data_path, monitor)]
    elif monitor_type == AiOpsEnum.NO_FAULT_DATA:
        monitor_path = [os.path.join(no_fault_data_path, monitor)]
    elif monitor_type == AiOpsEnum.ALL_FAULT_DATA:
        monitor_path = [os.path.join(fault_data_path, monitor),
                        os.path.join(new_fault_data_path, monitor)]

    linux_metrics = {}
    linux_metrics_file = os.path.join(SAVE_PATH, 'linux_metrics.json')
    if os.path.isfile(linux_metrics_file):
        with open(linux_metrics_file, 'r', encoding='utf-8') as f:
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
                json_str = json.dumps(linux_metrics[cmdb_id][kpi_name][device],
                                      indent=4, cls=MetricEncoder)
                device = device.replace('/', 'unk')
                file_path = os.path.join(
                    SAVE_MONITOR, f'groupby_{cmdb_id}-{kpi_name}-{device}')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                    print(f'write {file_path} done')

    assert cnt == data_size
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


def read_tc(tc_type: AiOpsEnum.ALL):
    files = csv_files(tc_type)
    tc_metrics = []
    tc_file = os.path.join(SAVE_TC, '%s.json' % tc_type.value)
    if os.path.isfile(tc_file):
        with open(tc_file, 'r', encoding='utf-8') as f:
            json_str = f.read()
            tc_metrics = json.loads(json_str, object_hook=transaction_decoder)
            return tc_metrics
    for file in files:
        with open(file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            # 去掉标题
            next(reader)
            for row in reader:
                tran = Transaction(row[0],
                                   int(row[1]),
                                   int(row[2]),
                                   float(row[3]),
                                   float(row[4]),
                                   float(row[5]),
                                   int(row[6]),
                                   float(row[7]),
                                   float(row[8]),
                                   float(row[9]))
                tc_metrics.append(tran)
        print(file, len(tc_metrics))

    json_str = json.dumps(tc_metrics, indent=4, cls=TransactionEncoder)
    with open(tc_file, 'w', encoding='utf-8') as f:
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
                        x.append(datetime.datetime.fromtimestamp(
                            metric.timestamp))
                        y.append(metric.value)

                if len(x) > 0:
                    plt.figure(figsize=(40, 20))
                    plt.plot(x, y)
                    plt.savefig(f'{cmdb_id}-{kpi_name}-{device}.pdf')
                    plt.show()
                    print('end')


def metric_to_datapoint(monitor_type: AiOpsEnum.ALL):
    files = csv_files(monitor_type)
    for file in files:
        datapoints = read_datapoints_from_monitor_csv(file)
        json_str = json.dumps(datapoints, indent=4, cls=DataPointEncoder)
        name = os.path.basename(file).removesuffix('.csv')
        json_file_path = os.path.join(SAVE_MONITOR, f'sorted_{monitor_type.value}_{name}.json')
        with open(json_file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        print(f'write {json_file_path} done')


def read_datapoints_from_monitor_csv(file_path):
    datapoints = []
    with open(file_path, mode='r', encoding='utf-8') as f:
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
            metric = kpi_name.replace('.', '_')
            tags = {'cmdb_id': cmdb_id,
                    'device': device}
            timestamp = timestamp * 1000
            datapoints.append(DataPoint(metric, tags, timestamp, value))
    sorted(datapoints)
    return datapoints


def one_tc_to_datapoints(tc: Transaction):
    tags = {'tran_code': tc.tran_code}
    ts = tc.timestamp * 1000
    return [
        DataPoint('amount', tags, ts, tc.amount),
        DataPoint('bus_success_rate', tags, ts, tc.bus_success_rate),
        DataPoint('sys_success_rate', tags, ts, tc.sys_success_rate),
        DataPoint('avg_rsp_time', tags, ts, tc.avg_rsp_time),
        DataPoint('stall_amount', tags, ts, tc.stall_amount),
        DataPoint('avg_proc_time', tags, ts, tc.avg_proc_time),
        DataPoint('stall_rate', tags, ts, tc.stall_amount),
        DataPoint('apdex', tags, ts, tc.apdex)
    ]


def tc_to_datapoint(tc):
    sorted(tc)
    datapoints = []
    for transaction in tqdm(tc):
        dps = one_tc_to_datapoints(transaction)
        for dp in dps:
            datapoints.append(dp)
    json_str = json.dumps(datapoints, indent=4, cls=DataPointEncoder)
    with open(os.path.join(SAVE_TC, 'tc_datapoints.json'), 'w', encoding='utf-8') as f:
        f.write(json_str)
    return datapoints


def csv_files(data_type: AiOpsEnum.ALL):
    paths = []
    if data_type == AiOpsEnum.MONITOR_DATA:
        paths = [os.path.join(fault_data_path, MONITOR),
                 os.path.join(new_fault_data_path, MONITOR),
                 os.path.join(no_fault_data_path, MONITOR)]
    elif data_type == AiOpsEnum.TC_DATA:
        paths = [os.path.join(fault_data_path, TC),
                 os.path.join(new_fault_data_path, TC),
                 os.path.join(no_fault_data_path, TC)]
    elif data_type == AiOpsEnum.FAULT_DATA_MONITOR:
        paths = [os.path.join(fault_data_path, MONITOR)]
    elif data_type == AiOpsEnum.NEW_FAULT_DATA_MONITOR:
        paths = [os.path.join(new_fault_data_path, MONITOR)]
    elif data_type == AiOpsEnum.NO_FAULT_DATA_MONITOR:
        paths = [os.path.join(no_fault_data_path, MONITOR)]
    elif data_type == AiOpsEnum.FAULT_DATA_TC:
        paths = [os.path.join(fault_data_path, TC)]
    elif data_type == AiOpsEnum.NEW_FAULT_DATA_TC:
        paths = [os.path.join(new_fault_data_path, TC)]
    elif data_type == AiOpsEnum.NO_FAULT_DATA_TC:
        paths = [os.path.join(no_fault_data_path, TC)]
    elif data_type == AiOpsEnum.ALL:
        paths = [os.path.join(fault_data_path, MONITOR),
                 os.path.join(new_fault_data_path, MONITOR),
                 os.path.join(no_fault_data_path, MONITOR),
                 os.path.join(fault_data_path, TC),
                 os.path.join(new_fault_data_path, TC),
                 os.path.join(no_fault_data_path, TC)
                 ]
    else:
        print("error not support this type", data_type)

    ans = []
    for path in paths:
        files = os.listdir(path)
        for file in files:
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and '.csv' in file:
                ans.append(file_path)
    return sorted(ans)


def write_monitor_to_mysql(data_type: AiOpsEnum.ALL):
    files = csv_files(data_type)
    LOG.info("pid=%d" % os.getpid())
    for file in files:
        if file in success_files:
            continue
        sql = get_sql_by_data(file)
        data = []
        df = pd.read_csv(file)
        df.fillna('unk', inplace=True)
        for _, row in df.iterrows():
            data.append((row['cmdb_id'], row['kpi_name'], row['device'], row['value'], row['timestamp']))
        # conn = pymysql.connect(
        #     host='10.82.77.104',
        #     user='root',
        #     password='123456',
        #     db='aiops2023'
        # )
        conn = psycopg2.connect(
            database='aiops2023',
            host='10.82.77.104',
            user='yinxiaoln',
            password='123456'
        )
        cursor = conn.cursor()
        try:
            LOG.info(f'{file}, {sql}')
            cursor.executemany(sql, data)
            conn.commit()
            LOG.success(f'write {file}')
        except Exception as e:
            LOG.error('error: %s %s %s' % (file, sql, e))
        finally:
            cursor.close()
            conn.close()


def write_tc_to_mysql(data_type: AiOpsEnum.ALL):
    files = csv_files(data_type)
    LOG.info('pid=%d' % os.getpid())
    for file in files:
        if file in success_files:
            continue
        data = []
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            data.append((row['tran_code'], row['timestamp'], row['amount'], row['bus_success_rate'],
                         row['sys_success_rate'], row['avg_rsp_time'], row['stall_amount'], row['avg_proc_time'],
                         row['stall_rate'], row['apdex']))
        # conn = pymysql.connect(
        #     host='10.82.77.104',
        #     user='root',
        #     password='123456',
        #     db='aiops2023'
        # )
        conn = psycopg2.connect(
            database='aiops2023',
            host='10.82.77.104',
            user='yinxiaoln',
            password='123456'
        )
        sql = get_sql_by_data(file)
        LOG.info(f'{file}, {sql}')
        cursor = conn.cursor()
        try:
            LOG.info(f'{file}, {sql}')
            cursor.executemany(sql, data)
            conn.commit()
            LOG.success(f'write {file}')
        except Exception as e:
            LOG.error('%s %s %s' % (file, sql, e))
        finally:
            cursor.close()
            conn.close()


def get_sql_by_data(file):
    if MONITOR in file:
        if AiOpsEnum.NEW_FAULT_DATA.value in file:
            table_name = AiOpsEnum.T_NEW_FAULT_MONITOR.value
        elif AiOpsEnum.NO_FAULT_DATA.value in file:
            table_name = AiOpsEnum.T_NO_FAULT_MONITOR.value
        elif AiOpsEnum.FAULT_DATA.value in file:
            table_name = AiOpsEnum.T_FAULT_MONITOR.value
        else:
            LOG.error("table_name not match %s", file)
            return None
        #sql = """insert ignore into %s values(%s, %s, %s, %s, %s)""" % (table_name, '%s', '%s', '%s', '%s', '%s')
        sql = f'insert into {table_name} values(%s, %s, %s, %s, %s) on conflict do nothing'
        return sql
    elif TC in file:
        if AiOpsEnum.NEW_FAULT_DATA.value in file:
            table_name = AiOpsEnum.T_NEW_FAULT_TC.value
        elif AiOpsEnum.NO_FAULT_DATA.value in file:
            table_name = AiOpsEnum.T_NO_FAULT_TC.value
        elif AiOpsEnum.FAULT_DATA.value in file:
            table_name = AiOpsEnum.T_FAULT_TC.value
        else:
            LOG.error("table name not match %s", file)
            return None
        #sql = f'insert ignore into {table_name} %s values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict do nothing'
        sql = f'insert into {table_name} values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict do nothing'
        return sql


def write_mysql_multi_process():
    pool = Pool(8)
    # pool.apply_async(func=write_monitor_to_mysql, args=(AiOpsEnum.NO_FAULT_DATA_MONITOR,))
    # pool.apply_async(func=write_monitor_to_mysql, args=(AiOpsEnum.FAULT_DATA_MONITOR,))
    # pool.apply_async(func=write_monitor_to_mysql, args=(AiOpsEnum.NEW_FAULT_DATA_MONITOR,))
    pool.apply_async(func=write_tc_to_mysql, args=(AiOpsEnum.NO_FAULT_DATA_TC,))
    pool.apply_async(func=write_tc_to_mysql, args=(AiOpsEnum.FAULT_DATA_TC,))
    pool.apply_async(func=write_tc_to_mysql, args=(AiOpsEnum.NEW_FAULT_DATA_TC,))
    pool.close()
    pool.join()
    LOG.success("write all file done")


# read_monitor()
# plot_linux_metric(1694620860, 1694620860 + 3600 * 24 * 7)
# read_monitor(AiOpsEnum.ALL_FAULT_DATA)
# metric_to_datapoint(AiOpsEnum.ALL_FAULT_DATA)
# tc_to_datapoint(read_tc(AiOpsEnum.ALL_FAULT_DATA))
# write_monitor_to_mysql(AiOpsEnum.FAULT_DATA_MONITOR)
# write_monitor_to_mysql(AiOpsEnum.NEW_FAULT_DATA_MONITOR)
# write_monitor_to_mysql(AiOpsEnum.NO_FAULT_DATA_MONITOR)
# write_tc_to_mysql(AiOpsEnum.FAULT_DATA_TC)
# write_tc_to_mysql(AiOpsEnum.NEW_FAULT_DATA_TC)
# write_tc_to_mysql(AiOpsEnum.NO_FAULT_DATA_TC)

if __name__ == '__main__':
    write_mysql_multi_process()

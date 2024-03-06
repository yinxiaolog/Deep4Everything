"""aiops dataset pre processing"""
import os
import csv
import json
import datetime
import time
import sys
import pickle
import joblib
from tqdm import tqdm
from enum import Enum
from multiprocessing import Pool
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
from loguru import logger as LOG
import numpy as np
import pandas as pd

LOG.remove()
LOG.add(sys.stderr, level='INFO')

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

cmdb_id_back_list = [
    'Weblogic_40', 'Weblogic_41', 'Weblogic_42', 'Weblogic_43', 'Weblogic_44',
    'Weblogic_45', 'Weblogic_46', 'Weblogic_47', 'Weblogic_48', 'Weblogic_47',
    'Weblogic_50'
]

tc_metrics = [
    'amount',
    'bus_success_rate',
    'sys_success_rate',
    'avg_rsp_time',
    'stall_amount',
    'avg_proc_time',
    'stall_rate',
    'apdex'
]

seq_len = 256
window = 5 * 60
dim = 32 + 1


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


class Fault:
    def __init__(self, start: int, end: int, values: []):
        self.start = start
        self.end = end
        self.values = values
        self.faults = []

    def __lt__(self, other):
        return self.start <= other.start

    def __str__(self):
        return str(__dict__)


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


class FaultEncoder(json.JSONEncoder):
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
        json_file_path = os.path.join(
            SAVE_MONITOR, f'sorted_{monitor_type.value}_{name}.json')
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
            data.append((row['cmdb_id'], row['kpi_name'],
                        row['device'], row['value'], row['timestamp']))
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
        # sql = """insert ignore into %s values(%s, %s, %s, %s, %s)""" % (table_name, '%s', '%s', '%s', '%s', '%s')
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
        # sql = f'insert ignore into {table_name} %s values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict do nothing'
        sql = f'insert into {table_name} values(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) on conflict do nothing'
        return sql


def write_mysql_multi_process():
    pool = Pool(8)
    pool.apply_async(func=write_monitor_to_mysql,
                     args=(AiOpsEnum.NO_FAULT_DATA_MONITOR,))
    pool.apply_async(func=write_monitor_to_mysql,
                     args=(AiOpsEnum.FAULT_DATA_MONITOR,))
    pool.apply_async(func=write_monitor_to_mysql,
                     args=(AiOpsEnum.NEW_FAULT_DATA_MONITOR,))
    pool.apply_async(func=write_tc_to_mysql,
                     args=(AiOpsEnum.NO_FAULT_DATA_TC,))
    pool.apply_async(func=write_tc_to_mysql, args=(AiOpsEnum.FAULT_DATA_TC,))
    pool.apply_async(func=write_tc_to_mysql,
                     args=(AiOpsEnum.NEW_FAULT_DATA_TC,))
    pool.close()
    pool.join()
    LOG.success("write all file done")


# read_monitor()
# plot_linux_metric(1694620860, 1694620860 + 3600 * 24 * 7)
# read_monitor(AiOpsEnum.ALL_FAULT_DATA)
# metric_to_datapoint(AiOpsEnum.ALL_FAULT_DATA)
# tc_to_datapoint(read_tc(AiOpsEnum.ALL_FAULT_DATA))


def select_from_pg(sql):
    conn = psycopg2.connect(
        database='aiops2023',
        host='10.82.77.104',
        user='yinxiaoln',
        password='123456'
    )

    cursor = conn.cursor()
    ans = []
    try:
        cursor.execute(sql)
        ans = cursor.fetchall()
    except Exception as e:
        LOG.error('%s' % (e))
    finally:
        cursor.close()
        conn.close()
    # LOG.info(f'rows={len(ans)}, {sql}')
    return ans


def fault_per_timeseries(rows):
    faults = []
    i = 0
    while i < len(rows):
        if rows[i][1] >= 100:
            i += 1
            continue
        j = i
        while j < len(rows):
            if rows[j][1] < 100:
                j += 1
                if j > 0 and j < len(rows) and rows[j][2] > rows[j - 1][2] + 10:
                    break
                continue
            else:
                break
        fault = Fault(rows[i][2] - window, rows[j - 1][2] + window, rows[i: j])
        faults.append(fault)
        i = j
    return faults


def find_fault(table_name: AiOpsEnum.T_FAULT_TC):
    tran_code_sql = f'select distinct tran_code from {table_name.value} order by tran_code'
    tran_codes = select_from_pg(tran_code_sql)
    tran_code_to_bus_success_rate = {}
    for tran_code in tran_codes:
        sql = f"select tran_code, bus_success_rate, timestamp from {table_name.value} where tran_code = '{tran_code[0]}' order by timestamp"
        tran_code_to_bus_success_rate[tran_code[0]] = select_from_pg(sql)

    faults = []
    for k, v in tran_code_to_bus_success_rate.items():
        LOG.debug(k)
        faults.extend(fault_per_timeseries(v))

    faults = sorted(faults)
    file = os.path.join(SAVE_PATH, f'{table_name.value}.json')
    with open(file, 'w') as f:
        f.write(json.dumps(faults, indent=4, cls=FaultEncoder))
    start = faults[0].start
    end = faults[0].end
    new_faults = []
    values = []
    for fault in faults:
        if fault.start > end:
            values = sorted(values, key=lambda row: row[2])
            new_fault = Fault(start, end, values.copy())
            new_faults.append(new_fault)
            start_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(start))
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end))
            LOG.info(f'{start_time} {end_time}')
            start = fault.start
            end = fault.end
            values.clear()
            values.extend(fault.values)
        else:
            values.extend(fault.values)
            end = max(end, fault.end)
    new_faults.append(Fault(start, end, values))
    file = os.path.join(SAVE_PATH, f'{table_name.value}_range.json')
    with open(file, 'w') as f:
        f.write(json.dumps(new_faults, indent=4, cls=FaultEncoder))
    return new_faults


def fill_values(values):
    if len(values) <= 1:
        LOG.debug(f'len of values = {len(values)}')
        return []
    arr = []
    for value in values:
        arr.append(value[1])
    arr = np.asarray(arr)
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        LOG.debug("std is 0")
        arr = arr + np.random.normal(0, 0.1, len(arr))
        mean = arr.mean()
        std = arr.std()

    new_values = [(value[0], (value[1] - mean) / std) for value in values]
    return new_values


def build_seq_by_faulu_range(values: [], fault_range: []):
    ans = []
    if not fault_range:
        ans.append(fill_values(values))
        return ans

    for i in range(len(fault_range) + 1):
        if i == 0:
            end = fault_range[0].start
            new_values = []
            for value in values:
                if value[0] < end:
                    new_values.append(value)
            ans.append(fill_values(values=new_values))
        elif i == len(fault_range):
            start = fault_range[-1].end
            new_values = []
            for value in values:
                if value[0] > start:
                    new_values.append(value)
            ans.append(fill_values(values=new_values))
        else:
            start = fault_range[i - 1].end
            end = fault_range[i].start
            new_values = []
            for value in values:
                if start < value[0] < end:
                    new_values.append(value)
            ans.append(fill_values(values=new_values))

    return ans


def build_train_data_monitor(table_name: AiOpsEnum, fault_range):
    metrics_sql = f"select distinct kpi_name from {table_name.value}"
    metrics = select_from_pg(metrics_sql)
    dct = {}
    for metric in metrics:
        metric = metric[0]
        cmdb_ids_sql = f"select distinct cmdb_id from {table_name.value} where kpi_name = '{metric}'"
        cmdb_ids = select_from_pg(cmdb_ids_sql)
        for cmdb_id in cmdb_ids:
            if cmdb_id[0] in cmdb_id_back_list:
                continue
            cmdb_id = cmdb_id[0]
            device_sql = f"select distinct device from {table_name.value} where kpi_name = '{metric}' and cmdb_id = '{cmdb_id}'"
            devices = select_from_pg(device_sql)
            for device in devices:
                device = device[0]
                values_sql = f"select timestamp, value from {table_name.value} where kpi_name = '{metric}' and cmdb_id = '{cmdb_id}' and device = '{device}' order by timestamp"
                values = select_from_pg(values_sql)
                k = f'{table_name.value}_{metric}_{cmdb_id}_{device}'
                v = build_seq_by_faulu_range(
                    values=values, fault_range=fault_range)
                dct[k] = v
    file = os.path.join(SAVE_PATH, 'before_vec',
                        f'{table_name.value}.before_vec')
    with open(file, 'w') as f:
        f.write(json.dumps(dct, indent=4))


def build_train_data_tc(table_name: AiOpsEnum, fault_range):
    tran_code_sql = f"select distinct tran_code from {table_name.value}"
    tran_codes = select_from_pg(tran_code_sql)
    dct = {}
    for tran_code in tran_codes:
        tran_code = tran_code[0]
        for metric in tc_metrics:
            values_sql = f"select timestamp, {metric} from {table_name.value} where tran_code = '{tran_code}'"
            values = select_from_pg(values_sql)
            k = f'{table_name.value}_{metric}_{tran_code}'
            v = build_seq_by_faulu_range(
                values=values, fault_range=fault_range)
            dct[k] = v
    file = os.path.join(SAVE_PATH, 'before_vec',
                        f'{table_name.value}.before_vec')
    with open(file, 'w') as f:
        f.write(json.dumps(dct, indent=4))


def build_train_data():
    fault_range = find_fault(AiOpsEnum.T_FAULT_TC)
    new_fault_range = find_fault(AiOpsEnum.T_NEW_FAULT_TC)
    pool = Pool(8)
    pool.apply_async(func=build_train_data_monitor,
                     args=(AiOpsEnum.T_NO_FAULT_MONITOR, []))
    pool.apply_async(func=build_train_data_tc,
                     args=(AiOpsEnum.T_NO_FAULT_TC, []))
    pool.apply_async(func=build_train_data_monitor, args=(
        AiOpsEnum.T_FAULT_MONITOR, fault_range))
    pool.apply_async(func=build_train_data_tc, args=(
        AiOpsEnum.T_FAULT_TC, fault_range))
    pool.apply_async(func=build_train_data_monitor, args=(
        AiOpsEnum.T_NEW_FAULT_MONITOR, new_fault_range))
    pool.apply_async(func=build_train_data_tc, args=(
        AiOpsEnum.T_NEW_FAULT_TC, new_fault_range))
    pool.close()
    pool.join()
    LOG.success("build train data done")


def build_test_seq(values, fault_range):
    ans = []

    def is_fault(ts):
        for fault in fault_range:
            if fault.start <= ts <= fault.end:
                return True
        return False

    for fault in fault_range:
        fault_values = []
        start = fault.start
        end = fault.end
        for value in values:
            if start <= value[0] <= end:
                fault_values.append(value)
        seq = []
        for value in values:
            if value[0] < start and not is_fault(value[0]):
                seq.append(value)
        if len(seq) > seq_len:
            seq = seq[len(seq) - seq_len:]
        LOG.info(f'seq_len={len(seq)} fault_values={len(fault_values)}')
        seq = fill_values(seq)
        ans.append([seq, fault_values])

    return ans


def build_test_data_monitor(table_name: AiOpsEnum, fault_range):
    metrics_sql = f"select distinct kpi_name from {table_name.value}"
    metrics = select_from_pg(metrics_sql)
    dct = {}
    for metric in metrics:
        metric = metric[0]
        cmdb_ids_sql = f"select distinct cmdb_id from {table_name.value} where kpi_name = '{metric}'"
        cmdb_ids = select_from_pg(cmdb_ids_sql)
        for cmdb_id in cmdb_ids:
            if cmdb_id[0] in cmdb_id_back_list:
                continue
            cmdb_id = cmdb_id[0]
            device_sql = f"select distinct device from {table_name.value} where kpi_name = '{metric}' and cmdb_id = '{cmdb_id}'"
            devices = select_from_pg(device_sql)
            for device in devices:
                device = device[0]
                values_sql = f"select timestamp, value from {table_name.value} where kpi_name = '{metric}' and cmdb_id = '{cmdb_id}' and device = '{device}' order by timestamp"
                values = select_from_pg(values_sql)
                LOG.info(f'{metric} {cmdb_id} {device}')
                if metric == 'system.tcp.time_wait' and cmdb_id == 'Weblogic_12' and device == 'unk':
                    pass

                k = f'{table_name.value}_{metric}_{cmdb_id}_{device}'
                v = build_test_seq(values=values, fault_range=fault_range)
                LOG.info(f'faults={len(fault_range)} seqs={len(v)}')
                dct[k] = v
    file = os.path.join(SAVE_PATH, 'before_vec',
                        f'test_{table_name.value}.before_vec')
    with open(file, 'w') as f:
        f.write(json.dumps(dct, indent=4))


def build_test_data_tc(table_name: AiOpsEnum, fault_range):
    tran_code_sql = f"select distinct tran_code from {table_name.value}"
    tran_codes = select_from_pg(tran_code_sql)
    dct = {}
    for tran_code in tran_codes:
        tran_code = tran_code[0]
        for metric in tc_metrics:
            values_sql = f"select timestamp, {metric} from {table_name.value} where tran_code = '{tran_code}'"
            values = select_from_pg(values_sql)
            k = f'{table_name.value}_{metric}_{tran_code}'
            v = build_test_seq(values=values, fault_range=fault_range)
            dct[k] = v
    file = os.path.join(SAVE_PATH, 'before_vec',
                        f'test_{table_name.value}.before_vec')
    with open(file, 'w') as f:
        f.write(json.dumps(dct, indent=4))


def build_test_data():
    fault_range = find_fault(AiOpsEnum.T_FAULT_TC)
    new_fault_range = find_fault(AiOpsEnum.T_NEW_FAULT_TC)

    pool = Pool(5)
    pool.apply_async(func=build_test_data_monitor, args=(
        AiOpsEnum.T_FAULT_MONITOR, fault_range))
    pool.apply_async(func=build_test_data_tc, args=(
        AiOpsEnum.T_FAULT_TC, fault_range))
    pool.apply_async(func=build_test_data_monitor, args=(
        AiOpsEnum.T_NEW_FAULT_MONITOR, new_fault_range))
    pool.apply_async(func=build_test_data_tc, args=(
        AiOpsEnum.T_NEW_FAULT_TC, new_fault_range))
    pool.close()
    pool.join()
    LOG.success("build test data done")


def dump_dataset(data, file):
    with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    print(f'process {os.getpgid()} done')


def build_train_dataset():
    train_path = os.path.join(SAVE_PATH, 'train')
    files = os.listdir(train_path)
    dataset = []
    for file in files:
        path = os.path.join(train_path, file)
        with open(path, 'r') as f:
            dct = json.loads(f.read())
            for _, v in dct.items():
                for seq in v:
                    if len(seq) < dim:
                        continue
                    new_seq = []
                    for i in range(dim, len(seq)):
                        sub_seq = seq[i - dim: i]
                        sub_v = [ts[1] for ts in sub_seq]
                        new_seq.append(sub_v)
                    for j in range(0, len(new_seq), seq_len):
                        matrix = [new_seq[k] for k in range(
                            j, min(len(new_seq), j + seq_len))]
                        dataset.append(matrix)
    LOG.info(f'dataset size={len(dataset)}')
    dataset = dataset[0: 10000]
    return dataset
    # 1462944
    
    path = os.path.join(SAVE_PATH, 'train_dataset')
    os.makedirs(path, exist_ok=True)
    step = 100
    j = 0

    pool = Pool(50)
    for i in range(0, len(dataset), step):
        sub = dataset[i: i + step]
        file = os.path.join(path, f'train_dataset_{j:06d}.json')
        pool.apply_async(func=dump_dataset, args=(sub, file))
        j += 1
    
    pool.close()
    pool.join()
    LOG.success('dump all done')
    return dataset


def build_test_dataset():
    test_path = os.path.join(SAVE_PATH, 'test')
    files = os.listdir(test_path)
    dataset = []
    for file in files:
        path = os.path.join(test_path, file)
        with open(path, 'r') as f:
            dct = json.loads(f.read())
            for k, v in dct.items():
                for seq in v:
                    normal = seq[0]
                    fault = seq[1]
                    if len(normal) <= 0:
                        continue
                    new_seq = []
                    for i in range(dim - 1, len(normal)):
                        sub_seq = normal[i - dim + 1: i]
                        sub_v = [ts[1] for ts in sub_seq]
                        new_seq.append(sub_v)
                    for j in range(0, len(new_seq), seq_len):
                        x = [new_seq[k] for k in range(j, min(len(new_seq), j + seq_len))]
                        fault = fill_values(fault)
                        if len(fault) > 0:
                            y = [ts[1] for ts in fault]
                            fault_name = f'{k}_{fault[0][0]}_{fault[-1][0]}'
                            dataset.append([fault_name, x, y])
    LOG.info(f'dataset size={len(dataset)}')
    # 1462944
    
    # path = os.path.join(SAVE_PATH, 'test_dataset')
    # os.makedirs(path, exist_ok=True)
    # step = 100
    # j = 0

    # pool = Pool(50)
    # for i in range(0, len(dataset), step):
    #     sub = dataset[i: i + step]
    #     file = os.path.join(path, f'train_dataset_{j:06d}.json')
    #     pool.apply_async(func=dump_dataset, args=(sub, file))
    #     j += 1
    
    # pool.close()
    # pool.join()
    # LOG.success('dump all done')
    return dataset


if __name__ == '__main__':
    #build_train_dataset()
    build_test_dataset()

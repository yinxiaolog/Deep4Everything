create table fault_monitor
(
    cmdb_id   varchar(128),
    kpi_name  varchar(128),
    device    varchar(128),
    value     double precision,
    timestamp bigint
);

comment on column fault_monitor.cmdb_id is '主机名';

comment on column fault_monitor.kpi_name is '指标名';

comment on column fault_monitor.device is '主机的磁盘等设备';

comment on column fault_monitor.value is '时序数据值';

comment on column fault_monitor.timestamp is '时间戳';

alter table fault_monitor
    owner to yinxiaoln;

create unique index fault_monitor_kpi_name_cmdb_id_device_timestamp_uindex
    on fault_monitor (kpi_name, cmdb_id, device, timestamp);

comment on index fault_monitor_kpi_name_cmdb_id_device_timestamp_uindex is '唯一索引';

create table fault_tc
(
    tran_code        varchar(64),
    timestamp        bigint,
    amount           bigint,
    bus_success_rate double precision,
    sys_success_rate double precision,
    avg_rsp_time     double precision,
    stall_amount     double precision,
    avg_proc_time    double precision,
    stall_rate       double precision,
    apdex            double precision
);

comment on column fault_tc.tran_code is '交易码';

comment on column fault_tc.timestamp is '时间戳';

comment on column fault_tc.amount is '交易量';

comment on column fault_tc.bus_success_rate is '业务成功率';

comment on column fault_tc.sys_success_rate is '系统成功率';

comment on column fault_tc.avg_rsp_time is '平均响应时间';

comment on column fault_tc.stall_amount is '长交易数';

comment on column fault_tc.avg_proc_time is '平均处理时间';

comment on column fault_tc.stall_rate is '长交易率';

comment on column fault_tc.apdex is '性能指标';

alter table fault_tc
    owner to yinxiaoln;

create unique index fault_tc_tran_code_timestamp_uindex
    on fault_tc (tran_code, timestamp);

comment on index fault_tc_tran_code_timestamp_uindex is '唯一索引';

create table new_fault_tc
(
    tran_code        varchar(64),
    timestamp        bigint,
    amount           bigint,
    bus_success_rate double precision,
    sys_success_rate double precision,
    avg_rsp_time     double precision,
    stall_amount     double precision,
    avg_proc_time    double precision,
    stall_rate       double precision,
    apdex            double precision
);

comment on column new_fault_tc.tran_code is '交易码';

comment on column new_fault_tc.timestamp is '时间戳';

comment on column new_fault_tc.amount is '交易量';

comment on column new_fault_tc.bus_success_rate is '业务成功率';

comment on column new_fault_tc.sys_success_rate is '系统成功率';

comment on column new_fault_tc.avg_rsp_time is '平均响应时间';

comment on column new_fault_tc.stall_amount is '长交易数';

comment on column new_fault_tc.avg_proc_time is '平均处理时间';

comment on column new_fault_tc.stall_rate is '长交易率';

comment on column new_fault_tc.apdex is '性能指标';

alter table new_fault_tc
    owner to yinxiaoln;

create unique index new_fault_tc_tran_code_timestamp_uindex
    on new_fault_tc (tran_code, timestamp);

comment on index new_fault_tc_tran_code_timestamp_uindex is '唯一索引';

create table new_fault_monitor
(
    cmdb_id   varchar(128),
    kpi_name  varchar(128),
    device    varchar(128),
    value     double precision,
    timestamp bigint
);

comment on column new_fault_monitor.cmdb_id is '主机名';

comment on column new_fault_monitor.kpi_name is '指标名';

comment on column new_fault_monitor.device is '主机的磁盘等设备';

comment on column new_fault_monitor.value is '时序数据值';

comment on column new_fault_monitor.timestamp is '时间戳';

alter table new_fault_monitor
    owner to yinxiaoln;

create unique index new_fault_monitor_kpi_name_cmdb_id_device_timestamp_uindex
    on new_fault_monitor (kpi_name, cmdb_id, device, timestamp);

comment on index new_fault_monitor_kpi_name_cmdb_id_device_timestamp_uindex is '唯一索引';

create table no_fault_monitor
(
    cmdb_id   varchar(128),
    kpi_name  varchar(128),
    device    varchar(128),
    value     double precision,
    timestamp bigint
);

comment on column no_fault_monitor.cmdb_id is '主机名';

comment on column no_fault_monitor.kpi_name is '指标名';

comment on column no_fault_monitor.device is '主机的磁盘等设备';

comment on column no_fault_monitor.value is '时序数据值';

comment on column no_fault_monitor.timestamp is '时间戳';

alter table no_fault_monitor
    owner to yinxiaoln;

create unique index no_fault_monitor_kpi_name_cmdb_id_device_timestamp_uindex
    on no_fault_monitor (kpi_name, cmdb_id, device, timestamp);

comment on index no_fault_monitor_kpi_name_cmdb_id_device_timestamp_uindex is '唯一索引';

create table no_fault_tc
(
    tran_code        varchar(64),
    timestamp        bigint,
    amount           bigint,
    bus_success_rate double precision,
    sys_success_rate double precision,
    avg_rsp_time     double precision,
    stall_amount     double precision,
    avg_proc_time    double precision,
    stall_rate       double precision,
    apdex            double precision
);

comment on column no_fault_tc.tran_code is '交易码';

comment on column no_fault_tc.timestamp is '时间戳';

comment on column no_fault_tc.amount is '交易量';

comment on column no_fault_tc.bus_success_rate is '业务成功率';

comment on column no_fault_tc.sys_success_rate is '系统成功率';

comment on column no_fault_tc.avg_rsp_time is '平均响应时间';

comment on column no_fault_tc.stall_amount is '长交易数';

comment on column no_fault_tc.avg_proc_time is '平均处理时间';

comment on column no_fault_tc.stall_rate is '长交易率';

comment on column no_fault_tc.apdex is '性能指标';

alter table no_fault_tc
    owner to yinxiaoln;

create unique index no_fault_tc_tran_code_timestamp_uindex
    on no_fault_tc (tran_code, timestamp);

comment on index no_fault_tc_tran_code_timestamp_uindex is '唯一索引';


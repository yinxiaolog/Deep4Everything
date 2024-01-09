create table fault_monitor
(
    cmdb_id   varchar(100) null comment '主机名',
    kpi_name  varchar(128) null comment '指标名',
    device    varchar(128) null comment '主机的磁盘等设备',
    value     double       null comment '时序数据值',
    timestamp bigint       null comment '时间戳',
    constraint fault_monitor_uindex
        unique (kpi_name, cmdb_id, device, timestamp) comment '唯一索引'
);

create table fault_tc
(
    tran_code        varchar(64) null comment '交易码',
    timestamp        bigint      null comment '时间戳',
    amount           bigint      null comment '交易量',
    bus_success_rate double      null comment '业务成功率',
    sys_success_rate double      null comment '系统成功率',
    avg_rsp_time     double      null comment '平均响应时间',
    stall_amount     double      null comment '长交易数',
    avg_proc_time    double      null comment '平均处理时间',
    stall_rate       double      null comment '长交易率',
    apdex            double      null comment '性能指标',
    constraint fault_tc_uindex
        unique (tran_code, timestamp) comment '唯一索引'
);

create table new_fault_monitor
(
    cmdb_id   varchar(100) null comment '主机名',
    kpi_name  varchar(128) null comment '指标名',
    device    varchar(128) null comment '主机的磁盘等设备',
    value     double       null comment '时序数据值',
    timestamp bigint       null comment '时间戳',
    constraint new_fault_monitor_uindex
        unique (kpi_name, cmdb_id, device, timestamp) comment '唯一索引'
);

create table new_fault_tc
(
    tran_code        varchar(64) null comment '交易码',
    timestamp        bigint      null comment '时间戳',
    amount           bigint      null comment '交易量',
    bus_success_rate double      null comment '业务成功率',
    sys_success_rate double      null comment '系统成功率',
    avg_rsp_time     double      null comment '平均响应时间',
    stall_amount     double      null comment '长交易数',
    avg_proc_time    double      null comment '平均处理时间',
    stall_rate       double      null comment '长交易率',
    apdex            double      null comment '性能指标',
    constraint new_fault_tc_uindex
        unique (tran_code, timestamp) comment '唯一索引'
);

create table no_fault_monitor
(
    cmdb_id   varchar(100) null comment '主机名',
    kpi_name  varchar(128) null comment '指标名',
    device    varchar(128) null comment '主机的磁盘等设备',
    value     double       null comment '时序数据值',
    timestamp bigint       null comment '时间戳',
    constraint no_fault_monitor_uindex
        unique (kpi_name, cmdb_id, device, timestamp) comment '唯一索引'
);

create table no_fault_tc
(
    tran_code        varchar(64) null comment '交易码',
    timestamp        bigint      null comment '时间戳',
    amount           bigint      null comment '交易量',
    bus_success_rate double      null comment '业务成功率',
    sys_success_rate double      null comment '系统成功率',
    avg_rsp_time     double      null comment '平均响应时间',
    stall_amount     double      null comment '长交易数',
    avg_proc_time    double      null comment '平均处理时间',
    stall_rate       double      null comment '长交易率',
    apdex            double      null comment '性能指标',
    constraint no_fault_tc_uindex
        unique (tran_code, timestamp) comment '唯一索引'
);


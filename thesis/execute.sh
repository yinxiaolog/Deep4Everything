#!/bin/bash
cur_time=$(date +'%Y-%m-%d_%H-%M-%S')
log_file=${cur_time}_cuda:$1.log
cp bert.py ${cur_time}_cuda:$1.py
nohup python bert.py 2>&1 > ${log_file} &


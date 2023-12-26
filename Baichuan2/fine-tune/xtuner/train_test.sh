#!/bin/bash

BASE_DIR="/data/yinxiaoln"
SAVE_PATH=$(date +"%Y-%m-%d_%H-%M-%S")

function train() {
    work_dir=${BASE_DIR}/save/Baichuan2/"${SAVE_PATH}"
    CUDA_VISIBLE_DEVICES=0 xtuner train baichuan_xtuner_config.py  --work-dir "${work_dir}"
}

function test() {
    xtuner_config=${BASE_DIR}/code/Deep4Everything/Baichuan2/fine-tune/xtuner/baichuan_xtuner_config.py
    pth_model_path=${work_dir}/epoch_100.pth
    hf_model_path=${work_dir}/epoch_100.hf
    xtuner convert pth_to_hf ${xtuner_config} ${pth_model_path} ${hf_model_path}
    python merge.py ${BASE_DIR}/pre_models/Baichuan2-7B-Chat  ${hf_model_path} ${work_dir}/merged
    python xtuner_test.py ${work_dir}/output.json ${work_dir}/merged
}

function main() {
    train
    test
}

main
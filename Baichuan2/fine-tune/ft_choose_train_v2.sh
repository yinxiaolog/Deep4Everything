hostfile=""
deepspeed --hostfile=$hostfile /data/usr/jy/Baichuan2/fine-tune/fine-tune.py  \
    --report_to "none" \
    --data_path "/data/usr/jy/Baichuan2/fine-tune/data/converted_choose_train_v2.json" \
    --model_name_or_path "/data/usr/jy/Baichuan2/fine-tune/output/choose_train_v2" \
    --output_dir "/data/usr/jy/Baichuan2/fine-tune/output/choose_train_v2" \
    --model_max_length 512 \
    --num_train_epochs 9 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --learning_rate 2e-5 \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight_decay 1e-4 \
    --warmup_ratio 0.0 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --deepspeed ds_config.json \
    --bf16 True \
    --tf32 True \
    --use_lora True
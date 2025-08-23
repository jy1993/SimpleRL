MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
OUTPUT_DIR=Checkpoints
TRAIN_PATH=data/train-logic.txt
TEST_PATH=data/test-logic.txt 
TASK_TYPE=logic
TB_PATH=logs/run_logic_r1_7b_20250729
LOG_PATH=logging/train_7b_20250729.log

nohup deepspeed --master_port 56789 --include localhost:0,1,2,3,4,5,6,7 main.py --group_size 8 --per_device_rollout_batch_size 8 --per_device_train_batch_size 8 --gradient_accumulation_steps 1 --tensorboard_path ${TB_PATH} --output_dir ${OUTPUT_DIR} --zero_stage 2 --model_path ${MODEL_PATH} --bf16 --gradient_checkpointing --debug --train_path ${TRAIN_PATH} --test_path ${TEST_PATH} --temperature 1.0 --save_steps 100 --eval_steps 50 --kl_coeff 0 --lr 1e-6 --warmup_steps 0 --epochs 3 --max_new_tokens 2048 --clip_lower 0.2 --clip_higher 0.28 --weight_decay 0.01 --task_type ${TASK_TYPE} > ${LOG_PATH} 2>&1 &

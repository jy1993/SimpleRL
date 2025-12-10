export NCCL_TIMEOUT=3600
nohup gunicorn --bind 0.0.0.0:8000 -w 32 --threads 2 --timeout 1 python_server:app > logging/gunicorn.log 2>&1 &
echo "start python-code server"

TENSORBOARD_PATH=your_tb_path
OUTPUT_DIR=your_output_dir
MODEL_PATH=your_model_path

deepspeed --include localhost:0,1,2,3,4,5,6,7 main.py --group_size 8 --per_device_rollout_batch_size 16 --per_device_train_batch_size 8 --gradient_accumulation_steps 2 --tensorboard_path ${TENSORBOARD_PATH} --output_dir ${OUTPUT_DIR} --zero_stage 2 --model_path ${MODEL_PATH} --bf16 --gradient_checkpointing --debug --train_path data/train/agent_math/train_dapo_clean.txt --test_path data/benchmark/math/aime.txt --temperature 1.0 --save_steps 100 --eval_steps 10000 --kl_coeff 0 --lr 1e-6 --warmup_steps 0 --epochs 1 --max_new_tokens 4800 --clip_lower 0.2 --clip_higher 0.28 --weight_decay 0.01 --task_type agent_math --mini_batch_size 2 --max_tool_calls 10 --unbiased_pass_k_n 16 --offload

pkill -f gunicorn
echo "end python-code server"

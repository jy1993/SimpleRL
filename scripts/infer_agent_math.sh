nohup gunicorn --bind 0.0.0.0:8000 -w 32 --threads 2 --timeout 1 python_server:app > logging/gunicorn.log 2>&1 &
echo "start python-code server"

MODEL_PATH=your_model_path
TEST_PATH=data/benchmark/math/aime.txt
SAVE_PATH=result/aime_result.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python agent_eval.py --model_path ${MODEL_PATH} --max_new_tokens 6000 --max_tool_calls 10 --test_path ${TEST_PATH} --max_test_samples 20000 --metric avg@k --k 32 --save_path ${SAVE_PATH} --n_gpus 4
pkill -f gunicorn
echo "end python-code server"

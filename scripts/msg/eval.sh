#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=dcu_htc
#SBATCH --nodes=1
#SBATCH --gres=dcu:8
#SBATCH --time=2-3:00:00
#SBATCH --ntasks-per-node=24
#SBATCH --mem=200G
#SBATCH --output=slurms/eval.out
#SBATCH --error=slurms/eval.err

ulimit -c 0

MODEL_PATH=./logs/msg-lr5e-5-cand-margin-0.1-rank-weight5-penalty1.4
NUM_BEAM=100
PORT=29400

echo "Running task with model path: ${MODEL_PATH} and num_beams: ${NUM_BEAM}"

accelerate launch --main_process_port $PORT src/eval_mp_post.py \
    --model_path $MODEL_PATH \
    --test_path ./data/MassSpecGym/test/CANOPUS_fps_selfies_threshold_0.2.tsv \
    --num_beams $NUM_BEAMS \
    --temperature 0.4 \
    --compute_mces

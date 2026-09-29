#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=dcu_htc
#SBATCH --nodes=1
#SBATCH --gres=dcu:8
#SBATCH --time=2-3:00:00
#SBATCH --ntasks-per-node=24
#SBATCH --mem=200G
#SBATCH --output=slurms/canopus-eval/out.out
#SBATCH --error=slurms/canopus-eval/err.err


ulimit -c 0

MODEL_PATH=./logs/canopus-lr5e-5-cand-margin-0.2-rank-weight5-penalty1.4-gen3
NUM_BEAM=100
TEMPERATURE=0.4
PORT=29400

echo "Model path: ${MODEL_PATH}"
echo "Num beams: ${NUM_BEAM}"
echo "Temperature: ${TEMPERATURE}"

accelerate launch --main_process_port $PORT src/eval_mp_post.py \
    --model_path $MODEL_PATH \
    --test_path ./data/CANOPUS/test/CANOPUS_fps_selfies_threshold_0.2.tsv \
    --num_beams $NUM_BEAM \
    --temperature $TEMPERATURE \
    --compute_mces

# sbatch --array=1 scripts/canopus/eval.sh

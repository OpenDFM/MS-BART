#!/bin/bash
#SBATCH --job-name=bart-base-selfies-pretrain-4M-ft-rank
#SBATCH --partition=gpu_debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=20:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=400G
#SBATCH --output=slurms/bart-base-selfies-pretrain-4M-ft-rank.out
#SBATCH --error=slurms/bart-base-selfies-pretrain-4M-ft-rank.err

LR=5e-5
CAND_MARGIN=0.1
RANK_WEIGHT=5
LENGTH_PENALTY=1.4

echo "Running Task with LR=$LR, MARGIN=$CAND_MARGIN, WEIGHT=$RANK_WEIGHT, PENALTY=$LENGTH_PENALTY"
ulimit -c 0


MODEL_NAME_OR_PATH=./log/msg-bart-base-selfies-pretrain-4M-ft
TOKENIZER_NAME=./log/msg-bart-base-selfies-pretrain-4M-ft
FINETUNE_PATH=./data/MassSpecGym/train/MassSpecGym_fps_selfies_threshold_0.11.tsv
VAL_PATH=./data/MassSpecGym/val/MassSpecGym_fps_selfies_threshold_0.11.tsv
SAVE_NAME=msg-lr$LR-cand-margin-$CAND_MARGIN-rank-weight$RANK_WEIGHT-penalty$LENGTH_PENALTY
PORT=$((9669 + SLURM_ARRAY_TASK_ID))
SAVE_STEPS=400
LOGGING_STEPS=200
EVAL_STEPS=400
EPOCHS=5
NUM_PROS=8

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

export WANDB_PROJECT="nips2025"
export WANDB_MODE="offline"


torchrun --nproc_per_node 2 --master_port $PORT src/rank_rl/main_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $TOKENIZER_NAME \
    --do_train \
    --do_eval \
    --bf16 \
    --finetune_path $FINETUNE_PATH \
    --val_path $VAL_PATH \
    --preprocessing_num_workers $NUM_PROS \
    --learning_rate $LR \
    --warmup_ratio 0.1 \
    --num_train_epochs $EPOCHS \
    --save_strategy steps \
    --save_steps  $SAVE_STEPS \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 2 \
    --report_to wandb \
    --run_name ms-$SAVE_NAME \
    --output_dir ./log/$SAVE_NAME \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --predict_with_generate \
    --max_source_length 256 \
    --save_safetensors False \
    --load_best_model_at_end \
    --metric_for_best_model top1_tanimoto_sim \
    --greater_is_better True \
    --cand_margin $CAND_MARGIN \
    --rank_weight $RANK_WEIGHT \
    --length_penalty $LENGTH_PENALTY \
    --freeze_encoder True

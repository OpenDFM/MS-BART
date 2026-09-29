#!/bin/bash
#SBATCH --job-name=canopus-bart-base-selfies-pretrain-4M-ft
#SBATCH --partition=gpu_llm_small
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mem=200G
#SBATCH --output=slurms/canopus/canopus-bart-base-selfies-pretrain-4M-ft.out
#SBATCH --error=slurms/canopus/canopus-bart-base-selfies-pretrain-4M-ft.err


ulimit -c 0

MODEL_NAME_OR_PATH=./logs/pretrained-model
TOKENIZER_NAME=./logs/pretrained-model
TRAIN_FILE=logs/datasets/CANOPUS/train/CANOPUS_fps_selfies_threshold_0.2.tsv
VALIDATION_FILE=logs/datasets/CANOPUS/val/CANOPUS_fps_selfies_threshold_0.2.tsv
SAVE_NAME=canopus-bart-base-selfies-pretrain-4M-ft
PORT=8669
SAVE_STEPS=200
LOGGING_STEPS=100
EVAL_STEPS=200
EPOCHS=20
NUM_PROS=8

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"


torchrun --nproc_per_node 1 --master_port $PORT src/ms_token/main_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $TOKENIZER_NAME \
    --do_train \
    --do_eval \
    --bf16 \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE\
    --preprocessing_num_workers $NUM_PROS \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --num_train_epochs $EPOCHS \
    --save_strategy steps \
    --save_steps  $SAVE_STEPS \
    --eval_strategy steps \
    --eval_steps $EVAL_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 2 \
    --report_to wandb \
    --run_name ms_$SAVE_NAME \
    --output_dir ./logs/$SAVE_NAME \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --predict_with_generate \
    --max_source_length 256 \
    --max_target_length 256 \
    --load_best_model_at_end \
    --metric_for_best_model top1_tanimoto_sim \
    --greater_is_better True \
    --early_stopping_patience 5

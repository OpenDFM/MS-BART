#!/bin/bash
#SBATCH --job-name=bart-base-selfies-pretrain-4M-ft
#SBATCH --partition=gpu_llm_small
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=20:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=400G
#SBATCH --output=slurms/bart-base-selfies-pretrain-4M-ft.out
#SBATCH --error=slurms/bart-base-selfies-pretrain-4M-ft.err

ulimit -c 0


MODEL_NAME_OR_PATH=./log/pretrained-model
TRAIN_FILE=./data/MassSpecGym/train/MassSpecGym_fps_selfies_threshold_0.11.tsv
VALIDATION_FILE=./data/MassSpecGym/val/MassSpecGym_fps_selfies_threshold_0.11.tsv
SAVE_NAME=msg-bart-base-selfies-pretrain-4M-ft
PORT=8669
SAVE_STEPS=400
LOGGING_STEPS=200
EVAL_STEPS=400
EPOCHS=10
NUM_PROS=8

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"


torchrun --nproc_per_node 2 --master_port $PORT src/ms_token/main_trainer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $MODEL_NAME_OR_PATH \
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
    --output_dir ./log/$SAVE_NAME \
    --gradient_accumulation_steps 1 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --predict_with_generate \
    --max_source_length 256 \
    --max_target_length 256 \
    --load_best_model_at_end \
    --early_stopping_patience 5 \
    --metric_for_best_model top1_tanimoto_sim \
    --greater_is_better True

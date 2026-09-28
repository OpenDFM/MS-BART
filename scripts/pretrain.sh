#!/bin/bash
#SBATCH --job-name=bart-base-selfies-pretrain-4M
#SBATCH --partition=gpu_llm_small
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2-3:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --mem=400G
#SBATCH --output=slurms/bart-base-selfies-pretrain-4M.out
#SBATCH --error=slurms/bart-base-selfies-pretrain-4M.err

ulimit -c 0

MODEL_NAME_OR_PATH=facebook/bart-base
TOKENIZER_NAME=./data/tokenizer/selfies-fps-tokenizer
PRETRAIN_FILE=./logs/datasets/MassSpecGym/pubchem-4M/train.tsv
VAL_FILE=./logs/datasets/MassSpecGym/pubchem-4M/val.tsv
SAVE_NAME=pretrained-model
PORT=8668
SAVE_STEPS=1000
LOGGING_STEPS=200
EPOCHS=3
NUM_PROS=8

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

torchrun --nproc_per_node 4 --master_port $PORT src/bart_pretrain_selfies_tokenizer.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --tokenizer_name $TOKENIZER_NAME \
    --do_train \
    --do_eval \
    --pretrain_path $PRETRAIN_FILE \
    --val_path $VAL_FILE \
    --preprocessing_num_workers $NUM_PROS \
    --log_level debug \
    --learning_rate 6e-4 \
    --lr_scheduler_type cosine_with_min_lr \
    --min_lr 1e-5 \
    --warmup_steps 10000 \
    --num_train_epochs $EPOCHS \
    --save_strategy steps \
    --save_steps  $SAVE_STEPS \
    --eval_strategy steps \
    --eval_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 3 \
    --report_to wandb \
    --run_name ms_$SAVE_NAME \
    --output_dir ./data/$SAVE_NAME \
    --max_seq_length 512 \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 96 \
    --per_device_eval_batch_size 96 \
    --load_best_model_at_end \
    --ddp_timeout 180000

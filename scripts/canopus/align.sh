ulimit -c 0

LR=5e-5
CAND_MARGIN=0.2
RANK_WEIGHT=5
LENGTH_PENALTY=1.4
NUM_GENERATION=3



MODEL_NAME_OR_PATH=./logs/canopus-bart-base-selfies-pretrain-4M-ft
TOKENIZER_NAME=./logs/canopus-bart-base-selfies-pretrain-4M-ft
FINETUNE_PATH=logs/datasets/CANOPUS/train/CANOPUS_fps_selfies_threshold_0.2.tsv
VAL_PATH=logs/datasets/CANOPUS/val/CANOPUS_fps_selfies_threshold_0.2.tsv
SAVE_NAME=canopus-lr$LR-cand-margin-$CAND_MARGIN-rank-weight$RANK_WEIGHT-penalty$LENGTH_PENALTY-gen$NUM_GENERATION
PORT=8669
SAVE_STEPS=50
LOGGING_STEPS=25
EVAL_STEPS=50
EPOCHS=20
NUM_PROS=8

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

export WANDB_PROJECT="nips2025"
export WANDB_MODE="offline"

torchrun --nproc_per_node 4 --master_port $PORT src/rank_rl/main_trainer.py \
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
    --output_dir ./logs/$SAVE_NAME \
    --gradient_accumulation_steps 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --predict_with_generate \
    --max_source_length 256 \
    --max_target_length 256 \
    --save_safetensors False \
    --load_best_model_at_end \
    --metric_for_best_model top1_tanimoto_sim \
    --greater_is_better True \
    --early_stopping_patience 5 \
    --cand_margin $CAND_MARGIN \
    --rank_weight $RANK_WEIGHT \
    --length_penalty $LENGTH_PENALTY \
    --num_generation $NUM_GENERATION \
    --freeze_encoder True

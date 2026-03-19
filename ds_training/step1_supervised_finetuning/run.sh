#!/bin/bash
OUTPUT_DIR="../../outputs/llama3.2-3b-yelp-raft-hf"
MODEL_PATH="meta-llama/Llama-3.2-3B"
TRAIN_FILE="../../raft_data/yelp/train.json"
EPOCHS=1
PER_DEVICE_BATCH=2
ACCUM_STEPS=8
LR=1e-5

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 python main.py \
    --model_name_or_path "$MODEL_PATH" \
    --train_file "$TRAIN_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH" \
    --gradient_accumulation_steps "$ACCUM_STEPS" \
    --learning_rate "$LR" \
    --num_train_epochs "$EPOCHS" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --target_modules "q_proj,v_proj,k_proj,o_proj" \
    --seed 1234 \
    --bf16 \
    --logging_steps 10 \
    --save_strategy epoch \
    --report_to none \
    &> "$OUTPUT_DIR/training.log"

echo "训练日志保存在 $OUTPUT_DIR/training.log"
echo "模型保存到 $OUTPUT_DIR"
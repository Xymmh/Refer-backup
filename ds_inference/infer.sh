#!/bin/bash
# 单卡推理脚本 - Llama-3.2-3B 基模 LoRA
# 注意：不使用 4bit 量化，避免 merge 后量化损失精度
MODEL_PATH="../outputs/llama3.2-3b-yelp-raft-hf"
BASE_MODEL_PATH="meta-llama/Llama-3.2-3B"
TEST_FILE="../raft_data/yelp/test.json"
SAVE_DIR="../generated_explanations/yelp"
BATCH_SIZE=4
MAX_TOKENS=256

mkdir -p "$SAVE_DIR"

CUDA_VISIBLE_DEVICES=0 python infer.py \
    --model_path "$MODEL_PATH" \
    --base_model_path "$BASE_MODEL_PATH" \
    --test_file "$TEST_FILE" \
    --save_dir "$SAVE_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_TOKENS" \
    &> "$SAVE_DIR/inference.log"

echo "生成结果保存在 $SAVE_DIR/generated_explanations.jsonl"
echo "日志: $SAVE_DIR/inference.log"
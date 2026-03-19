# infer.py - 修复问题1（prompt截取偏移）和问题2（base model路径硬编码）
# 保留 max_new_tokens=256

import json
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned RAFT model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to RAFT LoRA checkpoint")
    parser.add_argument("--base_model_path", type=str, default="meta-llama/Llama-3.2-3B",
                        help="Path to base model")  # 修复问题2：改为参数传入，不再硬编码
    parser.add_argument("--test_file", type=str, default="../../raft_data/yelp/test.json")
    parser.add_argument("--save_dir", type=str, default="../generated_explanations/yelp")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=256)  # 保持原始默认值
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--max_samples", type=int, default=1000)
    return parser.parse_args()


def load_model_and_tokenizer(model_path, base_model_path, use_4bit=False):
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading base model from: {base_model_path}")
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    model.eval()

    return model, tokenizer


def batch_generate(model, tokenizer, prompts, batch_size=4, max_new_tokens=256):
    generations = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(
            batch_prompts,
            padding=True,
            return_tensors="pt",
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # 修复问题1：用 batch 实际 input_ids 长度截取，而非逐条重新 tokenize
        # left padding 下所有样本等长，直接取 shape[1] 精确截掉 prompt+padding 部分
        input_len = inputs.input_ids.shape[1]
        for j in range(len(batch_prompts)):
            gen_text = tokenizer.decode(
                outputs[j][input_len:],
                skip_special_tokens=True
            )
            generations.append(gen_text.strip())

    return generations


def main():
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.base_model_path, args.use_4bit
    )

    print(f"Loading test data from: {args.test_file}")
    with open(args.test_file, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    if args.max_samples is not None:
        test_data = test_data[:args.max_samples]

    prompts = [item["prompt"] for item in test_data]
    print(f"Generating explanations for {len(prompts)} samples...")

    generated_exps = batch_generate(
        model, tokenizer, prompts, args.batch_size, args.max_new_tokens
    )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_file = Path(args.save_dir) / "generated_explanations.jsonl"
    print(f"Saving to: {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, (item, gen_exp) in enumerate(zip(test_data, generated_exps)):
            # 对齐作者输出格式：保留 index + source_data + input_str + output_str
            result = {
                "index": idx,
                "source_data": item,          # 完整原始数据（含 uid/iid/prompt/chosen/reject）
                "input_str": item["prompt"],   # 实际输入给模型的 prompt
                "output_str": gen_exp,          # 模型生成的内容
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Inference 完成！")


if __name__ == "__main__":
    main()
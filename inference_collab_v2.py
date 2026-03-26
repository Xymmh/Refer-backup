"""
inference_collab.py - 批量推理脚本

设计原则：
  - 逐条推理，完全无 padding，与训练格式一致
  - 只输入 prompt，不拼 chosen，模型自由生成
  - 输出格式与 G-Refer 一致，可直接用 metrics.py 评估
"""

import os, json, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',     default='meta-llama/Llama-3.2-3B')
    p.add_argument('--output_dir',     default='outputs/grpo_llama_yelp_v8/epoch1')
    p.add_argument('--test_path',      default='raft_data/yelp/test.json')
    p.add_argument('--save_path',      default='outputs/grpo_results_v8.jsonl')
    p.add_argument('--max_samples',    type=int, default=-1)
    p.add_argument('--max_new_tokens', type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[init] device={device}")

    # ── tokenizer ──────────────────────────────────────────────
    # 从 checkpoint 加载（含 <USER_EMB> <ITEM_EMB> 特殊 token）
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[tokenizer] vocab={len(tokenizer)}")

    # ── 模型：加载 base + 合并 LoRA ───────────────────────────
    print(f"[model] Loading base from {args.model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model.resize_token_embeddings(len(tokenizer))
    print(f"[model] Merging LoRA from {args.output_dir} ...")
    model = PeftModel.from_pretrained(model, args.output_dir)
    model = model.merge_and_unload()
    model.eval()
    print("[model] Ready.")

    # ── 加载测试集 ────────────────────────────────────────────
    samples = []
    with open(args.test_path) as f:
        for line in f:
            d = json.loads(line)
            samples.append({
                'uid':    d['uid'],
                'iid':    d['iid'],
                'prompt': d['prompt'],
                'chosen': d['chosen'],
            })
            if args.max_samples > 0 and len(samples) >= args.max_samples:
                break
    print(f"[data] Test samples: {len(samples)}")

    # ── 逐条推理（无 padding）────────────────────────────────
    results = []
    for i, s in enumerate(tqdm(samples, desc="Inferencing")):

        # 只 tokenize prompt，不拼 chosen
        input_ids = tokenizer(
            s['prompt'],
            add_special_tokens=False,
            return_tensors='pt',
        ).input_ids.to(device)

        input_len = input_ids.shape[1]

        with torch.no_grad():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 只取新生成的部分
        generated = tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True).strip()

        results.append({
            'uid':       s['uid'],
            'iid':       s['iid'],
            'prompt':    s['prompt'],
            'chosen':    s['chosen'],
            'generated': generated,
        })

        # 前3条打印对比
        if i < 3:
            print(f"\n[sample {i}]")
            print(f"  ground truth : {s['chosen'][:120]}")
            print(f"  generated    : {generated[:120]}")

    # ── 保存结果 ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        for idx, r in enumerate(results):
            record = {
                "index": idx,
                "source_data": {
                    "uid":    r["uid"],
                    "iid":    r["iid"],
                    "prompt": r["prompt"],
                    "chosen": r["chosen"],
                    "reject": "I DO NOT KNOW",
                },
                "input_str":  r["prompt"],
                "output_str": r["generated"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[done] Saved {len(results)} results → {args.save_path}")


if __name__ == '__main__':
    main()

# 运行:
#   python inference_collab.py \
#     --model_path meta-llama/Llama-3.2-3B \
#     --output_dir outputs/collab_llama_yelp_gnn_v3/epoch2 \
#     --test_path  raft_data/yelp/test.json \
#     --save_path  outputs/test_results.jsonl
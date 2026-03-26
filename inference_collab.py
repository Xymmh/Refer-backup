"""
批量推理脚本
- 对 raft_data/yelp/test.json 每条样本生成解释
- 输出格式与 G-Refer 一致，可直接用 metrics.py 评估
"""
import os, json, torch, argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',  default='meta-llama/Llama-3.2-3B')
    p.add_argument('--output_dir',  default='outputs/grpo_llama_yelp_v6_bart4/epoch1')
    p.add_argument('--test_path',   default='raft_data/yelp/test.json')
    p.add_argument('--save_path',   default='outputs/grpo_results_v6_bart4.jsonl')
    p.add_argument('--max_samples', type=int, default=-1,
                   help='调试用，-1表示全量')
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--max_length',  type=int, default=1200)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载 tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # 推理时左padding

    # 加载模型 + LoRA合并
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
    )
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(model, args.output_dir)
    model = model.merge_and_unload()
    model.eval()
    print("Model ready.")

    # 加载测试集
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
    print(f"Test samples: {len(samples)}")

    # 批量推理
    results = []
    for i in tqdm(range(0, len(samples), args.batch_size), desc="Inferencing"):
        batch = samples[i: i + args.batch_size]
        prompts = [s['prompt'] for s in batch]  # 引导模型以###开头续写

        enc = tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 只取新生成的部分
        input_len = enc['input_ids'].shape[1]
        for j, s in enumerate(batch):
            generated = tokenizer.decode(
                out[j][input_len:], skip_special_tokens=True).strip()
            results.append({
                'uid':       s['uid'],
                'iid':       s['iid'],
                'prompt':    s['prompt'],
                'chosen':    s['chosen'],      # ground truth
                'generated': generated,        # 模型输出
            })

    # 保存结果（与G-Refer格式一致）
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
                "output_str": r["generated"] if r["generated"].startswith("###") else "### " + r["generated"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(results)} results → {args.save_path}")

    # 打印前3条对比
    print("\n=== 样本预览 ===")
    for r in results[:3]:
        print(f"Ground truth: {r['chosen'][:100]}")
        print(f"Generated:    {r['generated'][:100]}")
        print()


if __name__ == '__main__':
    main()

# 运行命令:
# python inference_collab.py \
#   --model_path meta-llama/Llama-3.2-3B \
#   --output_dir outputs/collab_llama_yelp/epoch2 \
#   --test_path raft_data/yelp/test.json \
#   --save_path outputs/collab_results.jsonl \
#   --batch_size 4
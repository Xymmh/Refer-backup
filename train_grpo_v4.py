"""
train_grpo.py - 简化版 GRPO（无 Reference Model）

设计原则：
  - prompt 完整保留，绝不截断；无 padding，batch_size=1
  - 只使用 SFT 未见过的数据（--sft_used_samples 之后），且筛选含图路径的样本
  - 奖励：R_info（BARTScore generated→chosen）+ R_format（格式+长度）
  - R_info 为负数，越接近 0 越好，直接作为奖励
  
"""

import json, os, argparse, re, math, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation'))
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from bart_score import BARTScorer
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# 1. 参数
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sft_model_path',      default='outputs/collab_llama_yelp_gnn_v2/epoch2')
    p.add_argument('--base_model_path',     default='meta-llama/Llama-3.2-3B')
    p.add_argument('--raft_path',           default='raft_data/yelp/train.json')
    p.add_argument('--output_dir',          default='outputs/grpo_llama_yelp_v6_bart4')
    p.add_argument('--max_samples',         type=int,   default=5000)
    p.add_argument('--sft_used_samples',    type=int,   default=40000,
                   help='SFT 已使用的数据条数（从头部），GRPO 从剩余部分尾部往前取')
    p.add_argument('--num_epochs',          type=int,   default=1)
    p.add_argument('--G',                   type=int,   default=8,
                   help='每个 prompt 采样的输出数（组内归一化）')
    p.add_argument('--max_new_tokens',      type=int,   default=256)
    p.add_argument('--lr',                  type=float, default=1e-5)
    p.add_argument('--alpha',               type=float, default=0.8,  help='R_info(BARTScore) 权重')
    p.add_argument('--gamma',               type=float, default=0.2,  help='R_format 权重')
    p.add_argument('--format_penalty',      type=float, default=2.0,  help='缺少 ### 开头时的固定扣分')
    p.add_argument('--len_min',             type=int,   default=30,   help='长度惩罚下限（词数）')
    p.add_argument('--len_max',             type=int,   default=50,   help='长度惩罚上限（词数）')
    p.add_argument('--debug_every',         type=int,   default=1)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 2. 数据集
#    - 跳过 SFT 已使用的前 sft_used_samples 条
#    - 在剩余部分从尾部往前筛选含路径的样本，取 max_samples 条
# ─────────────────────────────────────────────────────────────
PATH_MARKER = "### For the given user-item pair, here are several related paths"
 
class GRPODataset(Dataset):
    def __init__(self, raft_path, max_samples=-1, sft_used_samples=0):
        all_samples = []
        with open(raft_path) as f:
            for line in f:
                d = json.loads(line)
                all_samples.append({'prompt': d['prompt'], 'chosen': d['chosen']})
 
        # 跳过 SFT 已用的前 sft_used_samples 条，只在剩余部分里选
        unseen = all_samples[sft_used_samples:]
 
        # 从尾部往前筛选含路径的样本
        candidates = [s for s in reversed(unseen) if PATH_MARKER in s['prompt']]
 
        if max_samples > 0:
            self.samples = candidates[:max_samples]
        else:
            self.samples = candidates
 
        print(f"[data] total={len(all_samples)} | "
              f"unseen(skip sft {sft_used_samples})={len(unseen)} | "
              f"has_path={len(candidates)} | "
              f"GRPO using={len(self.samples)}")
 
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
 
 
# ─────────────────────────────────────────────────────────────
# 3. 模型加载
# ─────────────────────────────────────────────────────────────
def load_model(base_path, sft_path, tokenizer):
    print(f"[model] Loading base from {base_path} ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16,          # 修复：torch_dtype → dtype
        device_map='auto', trust_remote_code=True)
    base.resize_token_embeddings(len(tokenizer))
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"[model] Loading SFT LoRA from {sft_path} (trainable) ...")
    model = PeftModel.from_pretrained(base, sft_path, is_trainable=True)
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    model.print_trainable_parameters()
    return model
 
 
# ─────────────────────────────────────────────────────────────
# 3b. BARTScorer（用于 R_info）
# ─────────────────────────────────────────────────────────────
def load_bart_scorer(device):
    """
    加载 BARTScorer（facebook/bart-large-cnn）用于计算 R_info。
    与评测脚本保持一致：batch_size=4，predictions→references 方向。
    """
    print("[bart] Loading BARTScorer (facebook/bart-large-cnn) ...")
    scorer = BARTScorer(device=str(device), checkpoint='facebook/bart-large-cnn')
    print("[bart] BARTScorer ready.")
    return scorer
 
 
# ─────────────────────────────────────────────────────────────
# 4. Tokenize（与 SFT 完全一致的格式，只截 response）
# ─────────────────────────────────────────────────────────────
def build_input_labels(tokenizer, prompt_text, response_text):
    """
    序列格式：[prompt_ids] [response_ids] <eos>
    labels  ：prompt 部分=-100，response+eos 参与 loss
    prompt / response 均完整保留，不做任何截断。
    """
    prompt_ids   = tokenizer(prompt_text,   add_special_tokens=False)['input_ids']
    response_ids = tokenizer(response_text, add_special_tokens=False)['input_ids']
 
    eos_id    = tokenizer.eos_token_id
    input_ids = prompt_ids + response_ids + [eos_id]
    labels    = [-100] * len(prompt_ids) + response_ids + [eos_id]
 
    assert len(input_ids) == len(labels)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels,    dtype=torch.long),
        len(prompt_ids),       # prompt_boundary
        len(response_ids),     # response_len
    )
 
 
# ─────────────────────────────────────────────────────────────
# 5. 采样 G 个输出
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_outputs(model, tokenizer, prompt, G, max_new_tokens, device):
    enc = tokenizer(prompt, add_special_tokens=False,
                    return_tensors='pt').to(device)
    input_len = enc.input_ids.shape[1]
 
    model.eval()
    texts = []
    for _ in range(G):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            out = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        t = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        texts.append(t)
    model.train()
    return texts
 
 
# ─────────────────────────────────────────────────────────────
# 6. 奖励函数（R_info=BARTScore + R_format=格式+长度）
# ─────────────────────────────────────────────────────────────
def compute_r_info(generated_texts, reference_texts, bart_scorer):
    """
    R_info = BARTScore(generated → chosen)，与评测脚本 BART_score() 完全一致。
    方向：generated 作为 src，chosen 作为 tgt。
    返回值为负数（log prob），越接近 0 说明生成质量越好。
    直接作为奖励使用，无需取反。
    典型范围：好的输出约 -3.0，差的输出约 -5.0 以下。
    """
    scores = bart_scorer.score(generated_texts, reference_texts, batch_size=4)
    return torch.tensor(scores, dtype=torch.float)
 
 
def extract_path(prompt_text):
    """
    截取图路径推理部分，跳过提示句，从 '1. ' 开始到 '\n### Explanation:' 为止。
    数据集已在加载时过滤，保证每条样本都有路径，此函数不需要兜底逻辑。
    """
    marker     = "### For the given user-item pair, here are several related paths"
    colon_skip = "interactions: "
    path_end   = "\n### Explanation:"
 
    start      = prompt_text.find(marker)
    end        = prompt_text.find(path_end)
    path_start = prompt_text.find(colon_skip, start) + len(colon_skip)
    return prompt_text[path_start:end].strip()
 
 
def compute_r_format(generated_texts, format_penalty=2.0,
                     target_min=25, target_max=50):
    """
    R_format = 格式检查 + 长度惩罚，合二为一。
 
    格式检查（固定扣分）：
      - 以 '### ' 开头：+0
      - 不以 '### ' 开头：-format_penalty
 
    长度惩罚（指数，无上限）：
      - 词数在 [target_min, target_max] 内：0
      - 偏差 n 词：-(exp(n * 0.1) - 1)
      示例：少/多5词 → -0.65，少/多10词 → -1.72，少/多20词 → -6.39
    """
    scores = []
    for t in generated_texts:
        score = 0.0
 
        # 格式检查
        if not t.startswith('###'):
            score -= format_penalty
 
        # 长度惩罚
        n = len(t.split())
        if n < target_min:
            score -= math.exp((target_min - n) * 0.1) - 1
        elif n > target_max:
            score -= math.exp((n - target_max) * 0.1) - 1
 
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float)
 
 
def compute_reward(generated_texts, reference_texts,
                   bart_scorer, device,
                   alpha, gamma, format_penalty, len_min, len_max):
    r_info   = compute_r_info(generated_texts, reference_texts, bart_scorer)
    r_format = compute_r_format(generated_texts, format_penalty, len_min, len_max)
    total    = alpha * r_info + gamma * r_format
    return total, r_info, r_format
 
 
# ─────────────────────────────────────────────────────────────
# 7. GRPO loss（精确边界，无 padding）
# ─────────────────────────────────────────────────────────────
def grpo_loss(model, tokenizer, prompt, generated_texts,
              advantages, device):
    """
    L = mean_i( advantage_i * NLL_i )
    NLL_i 只对 response 部分计算（prompt labels=-100）
    """
    model.train()
    loss_list = []
 
    for gen_text, adv in zip(generated_texts, advantages):
        if not gen_text.strip():
            continue
 
        input_ids, labels, _, response_len = build_input_labels(
            tokenizer, prompt, gen_text)
 
        if response_len == 0:
            continue
 
        input_ids = input_ids.unsqueeze(0).to(device)
        labels    = labels.unsqueeze(0).to(device)
 
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            nll_i = model(input_ids=input_ids, labels=labels).loss
 
        if torch.isnan(nll_i) or torch.isinf(nll_i):
            continue
 
        adv_t = torch.tensor(adv, device=device, dtype=nll_i.dtype)
        loss_list.append(adv_t * nll_i)
 
    if not loss_list:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(loss_list).mean()
 
 
# ─────────────────────────────────────────────────────────────
# 8. 主训练循环
# ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"[init] device={device}  G={args.G}  lr={args.lr}")
    print(f"{'='*60}\n")
 
    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    model = load_model(args.base_model_path, args.sft_model_path, tokenizer)
    bart_scorer = load_bart_scorer(device)
 
    dataset = GRPODataset(args.raft_path, args.max_samples, args.sft_used_samples)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)
 
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
 
    os.makedirs(args.output_dir, exist_ok=True)
 
    for epoch in range(args.num_epochs):
        total_loss, total_reward, steps = 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
 
        for batch in pbar:
            prompt = batch['prompt'][0]
            chosen = batch['chosen'][0]
 
 
            # ── 采样 G 个输出 ────────────────────────────────
            gen_texts = sample_outputs(
                model, tokenizer, prompt,
                args.G, args.max_new_tokens, device)
 
            # ── 计算奖励 ─────────────────────────────────────
            rewards, r_info, r_format = compute_reward(
                gen_texts, [chosen] * args.G,
                bart_scorer, device,
                args.alpha, args.gamma,
                args.format_penalty, args.len_min, args.len_max)
 
            # ── 组内归一化 → advantage ───────────────────────
            mean_r  = rewards.mean()
            std_r   = rewards.std() + 1e-8
            advantages = ((rewards - mean_r) / std_r).tolist()
 
            # ── 调试信息：展示所有G个结果，标注最优 ──────────
            if steps % args.debug_every == 0:
                best_idx = rewards.argmax().item()
                print(f"\n[debug step={steps}]")
                print(f"  chosen : {chosen[:100]!r}")
                print(f"  {'─'*56}")
                for i, (t, r, adv) in enumerate(
                        zip(gen_texts, rewards.tolist(), advantages)):
                    tag   = " ← BEST" if i == best_idx else ""
                    fmt_ok = "✓" if t.startswith("###") else "✗ no-###"
                    n_words = len(t.split())
                    print(f"  [{i}] reward={r:.4f}  adv={adv:+.4f}  "
                          f"len={n_words}w  fmt={fmt_ok}{tag}")
                    print(f"       {t[:100]!r}")
                print(f"  {'─'*56}")
                print(f"  r_info   : {[f'{r:.4f}' for r in r_info.tolist()]}")
                print(f"  r_format : {[f'{r:.4f}' for r in r_format.tolist()]}")
                # 拆解 r_format 细节：格式扣分 vs 长度扣分
                for i, t in enumerate(gen_texts):
                    fmt_pen  = -args.format_penalty if not t.startswith('###') else 0.0
                    n        = len(t.split())
                    if n < args.len_min:
                        len_pen = -(math.exp((args.len_min - n) * 0.1) - 1)
                    elif n > args.len_max:
                        len_pen = -(math.exp((n - args.len_max) * 0.1) - 1)
                    else:
                        len_pen = 0.0
                    print(f"    [{i}] fmt_pen={fmt_pen:.2f}  len_pen={len_pen:.4f}  "
                          f"words={n}")
                print(f"  mean_reward={mean_r.item():.4f}  std={std_r.item():.4f}")
 
            # ── loss & 反向传播 ──────────────────────────────
            optimizer.zero_grad()
            loss = grpo_loss(model, tokenizer, prompt, gen_texts,
                             advantages, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
 
            loss_val   = loss.item()
            reward_val = mean_r.item()
            bart_val   = r_info.mean().item()
            total_loss   += loss_val
            total_reward += reward_val
            steps += 1
 
            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             reward=f"{reward_val:.4f}",
                             bart=f"{bart_val:.4f}")
 
            del loss, gen_texts, rewards, r_info, r_format
            torch.cuda.empty_cache()
 
        print(f"\n[epoch {epoch+1}] avg_loss={total_loss/steps:.4f}  "
              f"avg_reward={total_reward/steps:.4f}")
 
        save_path = f"{args.output_dir}/epoch{epoch+1}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[save] → {save_path}")
 
    print("\nGRPO 训练完成！")
 
 
if __name__ == '__main__':
    train(parse_args())
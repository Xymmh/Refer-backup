"""
train_grpo.py - 简化版GRPO，去掉reference model
只保留 reward → advantage → loss 核心流程

运行命令:
python train_grpo.py \
  --sft_model_path outputs/collab_llama_yelp/epoch2 \
  --base_model_path meta-llama/Llama-3.2-3B \
  --raft_path raft_data/yelp/train.json \
  --output_dir outputs/grpo_llama_yelp \
  --max_samples 5000 --num_epochs 1 --G 4
"""

import json, math, os, argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ── 参数 ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sft_model_path',  default='outputs/collab_llama_yelp_gnn/epoch2')
    p.add_argument('--base_model_path', default='meta-llama/Llama-3.2-3B')
    p.add_argument('--raft_path',       default='raft_data/yelp/train.json')
    p.add_argument('--output_dir',      default='outputs/grpo_llama_yelp')
    p.add_argument('--max_samples',     type=int,   default=5000)
    p.add_argument('--num_epochs',      type=int,   default=1)
    p.add_argument('--G',               type=int,   default=4,   help='每个prompt采样几个输出')
    p.add_argument('--max_new_tokens',  type=int,   default=128)
    p.add_argument('--max_length',      type=int,   default=1200)
    p.add_argument('--lr',              type=float, default=1e-5)
    p.add_argument('--alpha',           type=float, default=0.5,  help='R_info权重')
    p.add_argument('--beta',            type=float, default=0.3,  help='R_recon权重')
    p.add_argument('--gamma',           type=float, default=0.2,  help='R_len权重')
    p.add_argument('--sbert_model',     default='all-MiniLM-L6-v2')
    return p.parse_args()


# ── 数据集 ───────────────────────────────────────────────────────────
class GRPODataset(Dataset):
    def __init__(self, raft_path, max_samples=-1):
        self.samples = []
        with open(raft_path) as f:
            for line in f:
                d = json.loads(line)
                self.samples.append({
                    'prompt': d['prompt'],
                    'chosen': d['chosen'],
                })
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break
        print(f"GRPO Dataset size: {len(self.samples)}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ── 模型加载（只加载一个模型）───────────────────────────────────────
def load_model(base_path, sft_path, tokenizer):
    print("Loading model (bfloat16 + gradient checkpointing)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map='auto',
        trust_remote_code=True,
    )
    base.resize_token_embeddings(len(tokenizer))

    # 开启gradient checkpointing节省激活值显存
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 直接加载SFT LoRA并设为可训练（不merge，避免峰值显存）
    print("Loading SFT LoRA as trainable adapter...")
    model = PeftModel.from_pretrained(base, sft_path, is_trainable=True)
    # 确保所有LoRA参数可训练
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
    model.print_trainable_parameters()
    return model


# ── 奖励函数 ─────────────────────────────────────────────────────────
def compute_r_info(generated_texts, reference_texts, sbert):
    """R_info = cosine_sim(φ(生成), φ(参考))，范围[-1, 1]"""
    gen_emb = sbert.encode(generated_texts, convert_to_tensor=True,
                            normalize_embeddings=True)
    ref_emb = sbert.encode(reference_texts, convert_to_tensor=True,
                            normalize_embeddings=True)
    return (gen_emb * ref_emb).sum(dim=-1).cpu().float()


def compute_r_recon(generated_texts, reference_texts, model, tokenizer, device):
    """
    R_recon = -exp(-mean log P(wj | w_{1:j-1}, W_I))
    论文原始含义：以生成文本为条件，计算重建参考文本（chosen）的log prob
    生成文本质量越高（信息丰富、无重复）→ 越容易重建chosen → NLL越低 → 惩罚越小
    重复词多的生成文本 → 信息量低 → 难以重建chosen → NLL高 → 惩罚大
    """
    results = []
    model.eval()
    with torch.no_grad():
        for gen_text, ref_text in zip(generated_texts, reference_texts):
            if not gen_text.strip() or not ref_text.strip():
                results.append(-1.0)
                continue

            # 拼接：生成文本作为条件，参考文本作为重建目标
            # 格式：[生成文本] [SEP] [参考文本]
            # 只对参考文本部分计算loss（mask掉生成文本部分）
            gen_enc = tokenizer(gen_text, return_tensors='pt',
                                truncation=True, max_length=200)
            gen_len = gen_enc.input_ids.shape[1]

            full_text = gen_text + " " + ref_text
            full_enc = tokenizer(full_text, return_tensors='pt',
                                 truncation=True, max_length=400).to(device)

            if full_enc.input_ids.shape[1] <= gen_len:
                results.append(-1.0)
                continue

            # 只对ref部分计算loss
            labels = full_enc.input_ids.clone()
            labels[0, :gen_len] = -100

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                nll = model(input_ids=full_enc.input_ids,
                            labels=labels).loss.item()

            results.append(-math.exp(-nll))
    return torch.tensor(results, dtype=torch.float)


def compute_r_repetition(generated_texts):
    """
    惩罚重复词：计算unique词占总词数的比例
    比例越低说明重复越多，惩罚越大
    范围：(0, 1]，1表示完全无重复
    """
    scores = []
    for text in generated_texts:
        words = text.lower().split()
        if not words:
            scores.append(-1.0)
            continue
        unique_ratio = len(set(words)) / len(words)
        # 映射到[-1, 0]：unique_ratio=1（无重复）→0，unique_ratio=0.5→-0.5
        scores.append(unique_ratio - 1.0)
    return torch.tensor(scores, dtype=torch.float)


def compute_r_len(generated_texts, target_min=25, target_max=60):
    """
    改进版长度奖励：
    - 词数在[target_min, target_max]之间给满分0
    - 过短（<target_min）线性惩罚，鼓励生成足够长的解释
    - 过长（>target_max）轻微惩罚，防止冗长
    """
    scores = []
    for t in generated_texts:
        length = len(t.split())
        if length < target_min:
            # 过短惩罚，最短为0词时得-1
            score = (length - target_min) / target_min
        elif length > target_max:
            # 过长轻微惩罚
            score = -(length - target_max) / target_max * 0.3
        else:
            score = 0.0
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float)


def compute_reward(generated_texts, reference_texts, model,
                   tokenizer, sbert, device, alpha, beta, gamma, delta=0.3):
    r_info  = compute_r_info(generated_texts, reference_texts, sbert)
    r_recon = compute_r_recon(generated_texts, reference_texts, model, tokenizer, device)
    r_len   = compute_r_len(generated_texts)
    r_rep   = compute_r_repetition(generated_texts)
    total   = alpha * r_info + beta * r_recon + gamma * r_len + delta * r_rep
    return total, r_info, r_recon, r_len, r_rep


# ── 采样G个输出 ──────────────────────────────────────────────────────
@torch.no_grad()
def sample_outputs(model, tokenizer, prompt, G, max_new_tokens, max_length, device):
    enc = tokenizer(prompt, return_tensors='pt',
                    truncation=True, max_length=max_length).to(device)
    input_len = enc.input_ids.shape[1]

    model.eval()
    texts = []
    # 串行生成，每次batch_size=1，避免显存爆炸
    for _ in range(G):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            output = model.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        t = tokenizer.decode(output[0][input_len:],
                             skip_special_tokens=True).strip()
        if not t.startswith('###'):
            t = '### ' + t
        texts.append(t)
    return texts, input_len


# ── GRPO loss ────────────────────────────────────────────────────────
def grpo_loss(model, tokenizer, prompt, generated_texts,
              advantages, device, max_length):
    """
    loss = -mean( advantage_i * mean_log_prob_i )
    只对生成部分的token计算log prob（prompt部分label设为-100）
    """
    model.train()
    prompt_len = tokenizer(prompt, return_tensors='pt',
                           truncation=True,
                           max_length=max_length).input_ids.shape[1]

    loss_list = []

    for gen_text, adv in zip(generated_texts, advantages):
        if not gen_text.strip():
            continue

        full_ids = tokenizer(
            prompt + gen_text, return_tensors='pt',
            truncation=True, max_length=max_length,
        ).input_ids.to(device)

        if full_ids.shape[1] <= prompt_len:
            continue

        # prompt部分mask掉，只对生成部分算loss
        labels = full_ids.clone()
        labels[0, :prompt_len] = -100

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            nll_i = model(input_ids=full_ids, labels=labels).loss

        adv_t = torch.tensor(adv, device=device, dtype=nll_i.dtype)
        loss_list.append(adv_t * nll_i)

    if not loss_list:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(loss_list).mean()


# ── 主训练循环 ───────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.base_model_path, args.sft_model_path, tokenizer)
    sbert = SentenceTransformer(args.sbert_model)

    dataset = GRPODataset(args.raft_path, args.max_samples)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        total_loss = 0.0
        total_reward = 0.0
        steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            prompt = batch['prompt'][0]
            chosen = batch['chosen'][0]

            # Step 1: 采样G个输出
            gen_texts, _ = sample_outputs(
                model, tokenizer, prompt,
                args.G, args.max_new_tokens, args.max_length, device)

            # Step 2: 计算reward
            rewards, r_info, r_recon, r_len, r_rep = compute_reward(
                gen_texts, [chosen] * args.G,
                model, tokenizer, sbert, device,
                args.alpha, args.beta, args.gamma)

            # Step 3: 组内归一化 → advantage
            mean_r = rewards.mean()
            std_r  = rewards.std() + 1e-8
            advantages = ((rewards - mean_r) / std_r).tolist()

            # Step 4: 计算loss并反向传播
            optimizer.zero_grad()
            loss = grpo_loss(model, tokenizer, prompt, gen_texts,
                             advantages, device, args.max_length)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val   = loss.item()
            reward_val = mean_r.item()
            rinfo_val  = r_info.mean().item()

            total_loss   += loss_val
            total_reward += reward_val
            steps += 1

            # 显式释放显存
            del loss, gen_texts, rewards, r_info, r_recon, r_len, r_rep
            torch.cuda.empty_cache()

            pbar.set_postfix({
                'loss':   f'{loss_val:.4f}',
                'reward': f'{reward_val:.4f}',
                'r_info': f'{rinfo_val:.4f}',
            })

        print(f"Epoch {epoch+1} "
              f"avg_loss={total_loss/steps:.4f} "
              f"avg_reward={total_reward/steps:.4f}")

        save_path = f"{args.output_dir}/epoch{epoch+1}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Saved → {save_path}")

    print("GRPO训练完成！")


if __name__ == '__main__':
    args = parse_args()
    train(args)
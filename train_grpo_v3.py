"""
train_grpo.py - 简化版 GRPO（无 Reference Model）v2

设计原则：
  - prompt 完整保留，绝不截断
  - 无 padding：build_input_labels 与 SFT 完全一致的格式
  - 奖励：R_info（语义相似度）+ R_recon（重建概率）+ R_len（长度）
  - 删除重复词惩罚（R_rep）

运行命令:
python train_grpo.py \
  --sft_model_path outputs/collab_llama_yelp_gnn_v2/epoch2 \
  --base_model_path meta-llama/Llama-3.2-3B \
  --raft_path raft_data/yelp/train.json \
  --output_dir outputs/grpo_llama_yelp_v6_rf3 \
  --max_samples 5000 --num_epochs 1 --G 4
"""

import json, os, argparse, re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# 1. 参数
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sft_model_path',      default='outputs/collab_llama_yelp_gnn_v2/epoch2')
    p.add_argument('--base_model_path',     default='meta-llama/Llama-3.2-3B')
    p.add_argument('--raft_path',           default='raft_data/yelp/train.json')
    p.add_argument('--output_dir',          default='outputs/grpo_llama_yelp_v6_rf3')
    p.add_argument('--max_samples',         type=int,   default=200)
    p.add_argument('--sft_used_samples',    type=int,   default=40000,
                   help='SFT 已使用的数据条数（从头部），GRPO 从剩余部分尾部往前取')
    p.add_argument('--num_epochs',          type=int,   default=1)
    p.add_argument('--G',                   type=int,   default=4,
                   help='每个 prompt 采样的输出数（组内归一化）')
    p.add_argument('--max_new_tokens',      type=int,   default=256)
    p.add_argument('--lr',                  type=float, default=1e-5)
    p.add_argument('--alpha',               type=float, default=0.4,  help='R_info 权重')
    p.add_argument('--beta',                type=float, default=0.5,  help='R_recon 权重')
    p.add_argument('--gamma',               type=float, default=0.1,  help='R_len 权重')
    p.add_argument('--sbert_model',         default='all-MiniLM-L6-v2')
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
# 3b. BERT 模型（用于 BERTScore F1 计算 R_recon）
# ─────────────────────────────────────────────────────────────
def load_bert_model(model_name='bert-base-uncased'):
    """
    加载 BERT 用于手动计算 BERTScore F1。
    常驻 CPU，计算时临时移到 GPU，计算完移回。
    """
    from transformers import BertModel, BertTokenizer
    print(f"[bert] Loading {model_name} for BERTScore ...")
    bert_tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_model     = BertModel.from_pretrained(model_name)
    bert_model.eval()
    for param in bert_model.parameters():
        param.requires_grad = False
    print(f"[bert] Ready on CPU, all params frozen.")
    return bert_model, bert_tokenizer


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
                temperature=1.2,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        t = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        texts.append(t)
    model.train()
    return texts


# ─────────────────────────────────────────────────────────────
# 6. 奖励函数（R_info + R_recon + R_len）
# ─────────────────────────────────────────────────────────────
def compute_r_info(generated_texts, reference_texts, sbert):
    """语义余弦相似度 ∈ [-1, 1]"""
    gen_emb = sbert.encode(generated_texts, convert_to_tensor=True,
                            normalize_embeddings=True)
    ref_emb = sbert.encode(reference_texts,  convert_to_tensor=True,
                            normalize_embeddings=True)
    return (gen_emb * ref_emb).sum(dim=-1).cpu().float()


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


# 停用词表（介词、冠词、代词、系动词等）
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "that",
    "this", "these", "those", "it", "its", "they", "them", "their", "who",
    "which", "what", "there", "here", "also", "not", "no", "so", "if",
    "than", "then", "when", "where", "how", "all", "any", "both", "each",
    "more", "most", "other", "some", "such", "into", "through", "during",
    "about", "above", "after", "before", "between", "i", "you", "he",
    "she", "we", "my", "your", "his", "her", "our", "us", "me", "him",
}


def remove_stopwords(text):
    """去除停用词，返回剩余词用空格拼接。"""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return ' '.join(w for w in words if w not in STOPWORDS)


def extract_middle_nodes(path_text):
    """
    从路径文本中提取中间节点的 Profile 内容，剔除首尾的目标 user 和目标 item。
    路径结构：User->Item_mid->User_mid->Item_target（每条路径4个节点）。
    首节点（目标user）和尾节点（目标item）往往复述 prompt 开头的 profile，
    剔除后只保留真正有区分度的中间推理节点。
    """
    profiles = re.findall(r'Profile:\s*(.*?)\)', path_text, re.DOTALL)
    profiles = [p.strip() for p in profiles]

    if len(profiles) <= 2:
        return ' '.join(profiles)

    # 每条路径4个节点，取中间两个（index 1, 2）
    middle = []
    i = 0
    while i < len(profiles):
        chunk = profiles[i:i+4]
        if len(chunk) == 4:
            middle.extend(chunk[1:3])
        elif len(chunk) > 2:
            middle.extend(chunk[1:-1])
        i += 4
    return ' '.join(middle) if middle else ' '.join(profiles)


def bertscore_f1(gen_texts, ref_texts, bert_model, bert_tokenizer, device):
    """
    手动实现 BERTScore F1（token 级语义匹配）：
      Precision = 对 gen 里每个 token，在 ref 里找最相似的，取均值
      Recall    = 对 ref 里每个 token，在 gen 里找最相似的，取均值
      F1        = 调和平均
    输入文本已去除停用词。
    """
    scores = []
    bert_model.eval()
    with torch.no_grad():
        for gen, ref in zip(gen_texts, ref_texts):
            if not gen.strip() or not ref.strip():
                scores.append(0.0)
                continue

            def encode(text):
                enc = bert_tokenizer(
                    text, return_tensors='pt',
                    truncation=True, max_length=128,
                    add_special_tokens=True).to(device)
                out = bert_model(**enc, output_hidden_states=True)
                # 取最后一层 hidden state，去掉 [CLS] 和 [SEP]
                emb = out.hidden_states[-1][0, 1:-1, :]
                # L2 归一化
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                return emb  # (seq_len, hidden)

            gen_emb = encode(gen)   # (G, H)
            ref_emb = encode(ref)   # (R, H)

            if gen_emb.shape[0] == 0 or ref_emb.shape[0] == 0:
                scores.append(0.0)
                continue

            # 相似度矩阵 (G, R)
            sim = torch.mm(gen_emb, ref_emb.T)

            precision = sim.max(dim=1).values.mean().item()
            recall    = sim.max(dim=0).values.mean().item()
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            scores.append(f1)
    return torch.tensor(scores, dtype=torch.float)


def compute_r_recon(generated_texts, middle_node_texts,
                    bert_model, bert_tokenizer, device):
    """
    R_recon = BERTScore F1(去停用词后的生成文本, 去停用词后的中间节点文本)
    衡量生成的解释与路径中间节点的语义重叠程度。
    中间节点是真正有区分度的推理信息（剔除了首尾user/item复述）。
    ∈ [0, 1]，越高越好。
    """
    # 对生成文本和中间节点文本都去除停用词
    gen_clean  = [remove_stopwords(t) for t in generated_texts]
    node_clean = [remove_stopwords(t) for t in middle_node_texts]

    bert_model.to(device)
    result = bertscore_f1(gen_clean, node_clean,
                          bert_model, bert_tokenizer, device)
    bert_model.to('cpu')
    torch.cuda.empty_cache()
    return result


def compute_r_len(generated_texts, target_min=30, target_max=50):
    """
    [target_min, target_max] 词数内得 0。
    指数惩罚，无上限：偏差越大惩罚指数级增大。
    penalty = exp(偏差词数 * 0.1) - 1
    示例：少/多5词 → -0.65，少/多10词 → -1.72，少/多20词 → -6.39
    """
    import math
    scores = []
    for t in generated_texts:
        n = len(t.split())
        if n < target_min:
            penalty = math.exp((target_min - n) * 0.1) - 1
        elif n > target_max:
            penalty = math.exp((n - target_max) * 0.1) - 1
        else:
            penalty = 0.0
        scores.append(-penalty)
    return torch.tensor(scores, dtype=torch.float)


def compute_reward(generated_texts, middle_node_texts, reference_texts,
                   bert_model, bert_tokenizer, sbert, device,
                   alpha, beta, gamma):
    r_info  = compute_r_info(generated_texts, reference_texts, sbert)
    r_recon = compute_r_recon(generated_texts, middle_node_texts,
                               bert_model, bert_tokenizer, device)
    r_len   = compute_r_len(generated_texts)
    total   = alpha * r_info + beta * r_recon + gamma * r_len
    return total, r_info, r_recon, r_len


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
    bert_model, bert_tokenizer = load_bert_model()
    sbert = SentenceTransformer(args.sbert_model)

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

            # ── 截取路径并提取中间节点 ────────────────────────
            path_text        = extract_path(prompt)
            middle_node_text = extract_middle_nodes(path_text)

            # ── 采样 G 个输出 ────────────────────────────────
            gen_texts = sample_outputs(
                model, tokenizer, prompt,
                args.G, args.max_new_tokens, device)

            # ── 计算奖励 ─────────────────────────────────────
            rewards, r_info, r_recon, r_len = compute_reward(
                gen_texts, [middle_node_text] * args.G, [chosen] * args.G,
                bert_model, bert_tokenizer, sbert, device,
                args.alpha, args.beta, args.gamma)

            # ── 组内归一化 → advantage ───────────────────────
            mean_r  = rewards.mean()
            std_r   = rewards.std() + 1e-8
            advantages = ((rewards - mean_r) / std_r).tolist()

            # ── 调试信息：展示所有G个结果，标注最优 ──────────
            if steps % args.debug_every == 0:
                best_idx = rewards.argmax().item()
                print(f"\n[debug step={steps}]")
                node_clean_dbg = remove_stopwords(middle_node_text)
                print(f"  path raw        : {path_text[:100]!r}")
                print(f"  middle nodes    : {middle_node_text[:120]!r}")
                print(f"  nodes(no stop)  : {node_clean_dbg[:120]!r}")
                print(f"  chosen          : {chosen[:80]!r}")
                print(f"  {'─'*56}")
                for i, (t, r, adv) in enumerate(
                        zip(gen_texts, rewards.tolist(), advantages)):
                    tag = " ← BEST" if i == best_idx else ""
                    print(f"  [{i}] reward={r:.4f}  adv={adv:+.4f}  "
                          f"len={len(t.split())}w{tag}")
                    print(f"       {t[:100]!r}")
                print(f"  {'─'*56}")
                print(f"  r_info  : {[f'{r:.4f}' for r in r_info.tolist()]}")
                print(f"  r_recon : {[f'{r:.4f}' for r in r_recon.tolist()]}")
                print(f"  r_len   : {[f'{r:.4f}' for r in r_len.tolist()]}")
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
            total_loss   += loss_val
            total_reward += reward_val
            steps += 1

            del loss, gen_texts, rewards
            torch.cuda.empty_cache()

            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             reward=f"{reward_val:.4f}",
                             r_info=f"{r_info.mean().item():.4f}")

        print(f"\n[epoch {epoch+1}] avg_loss={total_loss/steps:.4f}  "
              f"avg_reward={total_reward/steps:.4f}")

        save_path = f"{args.output_dir}/epoch{epoch+1}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[save] → {save_path}")

    print("\nGRPO 训练完成！")


if __name__ == '__main__':
    train(parse_args())
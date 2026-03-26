"""
train_grpo.py - 简化版 GRPO（无 Reference Model）

设计原则：
  - prompt 完整保留，绝不截断；无 padding，batch_size=1
  - 只使用 SFT 未见过的数据（--sft_used_samples 之后），且筛选含路径的样本
  - 奖励：R_bart + R_bert + R_format + R_rep + R_path（路径实体命中）+ R_entity（profile 实体命中）
  - KL 散度约束：ref model 常驻 CPU，每个 sample 计算 loss 时整体搬到 GPU 一次，算完移回
  - R_bart 为负数，越接近 0 越好；R_bert ∈ [0,1] 越高越好
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
    p.add_argument('--sft_model_path',     default='outputs/collab_llama_yelp_gnn_v2/epoch2')
    p.add_argument('--base_model_path',    default='meta-llama/Llama-3.2-3B')
    p.add_argument('--raft_path',          default='raft_data/yelp/train.json')
    p.add_argument('--output_dir',         default='outputs/grpo_llama_yelp_v6_beba6')
    p.add_argument('--max_samples',        type=int,   default=100)
    p.add_argument('--sft_used_samples',   type=int,   default=40000,
                   help='SFT 已使用的数据条数（从头部），GRPO 从剩余部分尾部往前取')
    p.add_argument('--num_epochs',         type=int,   default=1)
    p.add_argument('--G',                  type=int,   default=6,
                   help='每个 prompt 采样的输出数（组内归一化）')
    p.add_argument('--max_new_tokens',     type=int,   default=256)
    p.add_argument('--lr',                 type=float, default=2e-6)
    p.add_argument('--kl_coef',            type=float, default=0.1,
                   help='KL 散度惩罚系数，约束策略不偏离参考模型')
    # 奖励权重
    p.add_argument('--alpha',              type=float, default=0.5,  help='R_bart 权重')
    p.add_argument('--beta',               type=float, default=0.3,  help='R_bert 权重')
    p.add_argument('--gamma',              type=float, default=0.1,  help='R_format 权重')
    p.add_argument('--delta',              type=float, default=0.1,  help='R_rep 权重')
    p.add_argument('--epsilon',            type=float, default=0.3,  help='R_path 权重')
    p.add_argument('--zeta',               type=float, default=0.2,  help='R_entity 权重')
    # 格式 / 长度惩罚
    p.add_argument('--format_penalty',     type=float, default=2.0,
                   help='缺少 ### The user would 开头时的固定扣分')
    p.add_argument('--len_min',            type=int,   default=25,   help='长度惩罚下限（词数）')
    p.add_argument('--len_max',            type=int,   default=55,   help='长度惩罚上限（词数）')
    # 重复惩罚
    p.add_argument('--rep_penalty',        type=float, default=0.1,  help='每个重复词的扣分')
    p.add_argument('--prefix_penalty',     type=float, default=0.3,  help='组内前缀重复的指数惩罚速率')
    p.add_argument('--prefix_free_len',    type=int,   default=6,
                   help='组内前缀重复的免罚词数（默认前6词相同不扣分）')
    # 实体命中奖励
    p.add_argument('--path_reward',        type=float, default=0.3,  help='路径实体每命中一词的加分')
    p.add_argument('--entity_reward',      type=float, default=0.3,  help='profile 实体每命中一词的加分')
    p.add_argument('--debug_every',        type=int,   default=1)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 2. 数据集
# ─────────────────────────────────────────────────────────────
PATH_MARKER = "### For the given user-item pair, here are several related paths"

STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have',
    'had', 'it', 'its', 'this', 'that', 'i', 'you', 'he', 'she', 'we',
    'they', 'my', 'your', 'his', 'her', 'our', 'their', 'not', 'no', 'as',
    'by', 'from', 'up', 'out', 'so', 'if', 'do', 'did', 'will', 'would',
    'can', 'could', 'may', 'might', 'than', 'then', 'also', 'just', 'more',
    'very', 'all', 'any', 'each', 'both', 'about', 'into', 'through',
    'during', 'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why',
}


def _clean_words(text):
    """小写、只保留字母数字词、去除停用词和短词，通用预处理。"""
    return {w for w in re.findall(r'[a-z0-9]+', text.lower())
            if w not in STOPWORDS and len(w) > 2}


def _extract_path_entities(prompt):
    """
    从 PATH_MARKER 之后的路径段提取实体节点词。
    路径格式：EntityA -> relation -> EntityB -> ...
    偶数位置（0-indexed）为实体，奇数位置为关系边，只取实体。
    """
    start = prompt.find(PATH_MARKER)
    if start == -1:
        return set()
    path_text = prompt[start + len(PATH_MARKER):]
    tokens = re.split(r'\s*->\s*', path_text)
    entities = set()
    for i, tok in enumerate(tokens):
        if i % 2 == 0:  # 实体位置
            for w in _clean_words(tok.replace('_', ' ')):
                entities.add(w)
    return entities


def _extract_profile_entities(prompt):
    """
    从 Business profile 字段提取关键词（菜名、特色词等）。
    取从 'Business profile:' 到下一个换行符之间的内容。
    """
    start = prompt.find("Business profile:")
    if start == -1:
        return set()
    start += len("Business profile:")
    end = prompt.find("\n", start)
    segment = prompt[start:end] if end != -1 else prompt[start:]
    return _clean_words(segment)


class GRPODataset(Dataset):
    def __init__(self, raft_path, max_samples=-1, sft_used_samples=0):
        all_samples = []
        with open(raft_path) as f:
            for line in f:
                d = json.loads(line)
                all_samples.append({'prompt': d['prompt'], 'chosen': d['chosen']})

        unseen     = all_samples[sft_used_samples:]
        candidates = [s for s in reversed(unseen) if PATH_MARKER in s['prompt']]
        self.samples = candidates[:max_samples] if max_samples > 0 else candidates

        # 预计算路径实体和 profile 实体（sorted list，可被 DataLoader collate）
        for s in self.samples:
            s['path_entities']    = sorted(_extract_path_entities(s['prompt']))
            s['profile_entities'] = sorted(_extract_profile_entities(s['prompt']))

        pe_avg = sum(len(s['path_entities'])    for s in self.samples) / max(len(self.samples), 1)
        en_avg = sum(len(s['profile_entities']) for s in self.samples) / max(len(self.samples), 1)
        print(f"[data] total={len(all_samples)} | "
              f"unseen(skip sft {sft_used_samples})={len(unseen)} | "
              f"has_path={len(candidates)} | "
              f"GRPO using={len(self.samples)} | "
              f"avg path_entities={pe_avg:.1f}  avg profile_entities={en_avg:.1f}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ─────────────────────────────────────────────────────────────
# 3. 模型加载
# ─────────────────────────────────────────────────────────────
def load_model(base_path, sft_path, tokenizer):
    print(f"[model] Loading base from {base_path} ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_path, dtype=torch.bfloat16,
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


def load_ref_model(base_path, sft_path, tokenizer):
    """
    加载冻结的参考模型，常驻 CPU 节省显存。
    grpo_loss 调用时整体搬到 GPU，G 条 response 全算完后移回 CPU，每个 sample 只搬一次。
    """
    print(f"[ref] Loading reference model from {sft_path} (frozen, CPU) ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16,
        device_map='cpu', trust_remote_code=True)
    base.resize_token_embeddings(len(tokenizer))
    ref_model = PeftModel.from_pretrained(base, sft_path, is_trainable=False)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print("[ref] Reference model ready on CPU (all params frozen).")
    return ref_model


def load_bart_scorer(device):
    print("[bart] Loading BARTScorer (facebook/bart-large-cnn) ...")
    scorer = BARTScorer(device=str(device), checkpoint='facebook/bart-large-cnn')
    print("[bart] BARTScorer ready.")
    return scorer


def load_bert_model(model_name='bert-base-uncased'):
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
# 4. Tokenize
# ─────────────────────────────────────────────────────────────
def build_input_labels(tokenizer, prompt_text, response_text):
    """
    序列格式：[prompt_ids] [response_ids] <eos>
    labels  ：prompt 部分=-100，response+eos 参与 loss
    """
    prompt_ids   = tokenizer(prompt_text,   add_special_tokens=False)['input_ids']
    response_ids = tokenizer(response_text, add_special_tokens=False)['input_ids']
    eos_id       = tokenizer.eos_token_id
    input_ids    = prompt_ids + response_ids + [eos_id]
    labels       = [-100] * len(prompt_ids) + response_ids + [eos_id]
    assert len(input_ids) == len(labels)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels,    dtype=torch.long),
        len(prompt_ids),
        len(response_ids),
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
# 6. 奖励函数
# ─────────────────────────────────────────────────────────────
def compute_r_bart(generated_texts, reference_texts, bart_scorer):
    """BARTScore(generated → chosen)，负数，越接近 0 越好。"""
    scores = bart_scorer.score(generated_texts, reference_texts, batch_size=4)
    return torch.tensor(scores, dtype=torch.float)


def compute_r_bert(generated_texts, reference_texts, bert_model, bert_tokenizer, device):
    """BERTScore F1(generated, chosen)，∈ [0,1]，越高越好。"""
    scores = []
    bert_model.to(device)
    bert_model.eval()
    with torch.no_grad():
        for gen, ref in zip(generated_texts, reference_texts):
            if not gen.strip() or not ref.strip():
                scores.append(0.0)
                continue

            def encode(text):
                enc = bert_tokenizer(
                    text, return_tensors='pt',
                    truncation=True, max_length=128,
                    add_special_tokens=True).to(device)
                out = bert_model(**enc, output_hidden_states=True)
                emb = out.hidden_states[-1][0, 1:-1, :]
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
                return emb

            gen_emb = encode(gen)
            ref_emb = encode(ref)
            if gen_emb.shape[0] == 0 or ref_emb.shape[0] == 0:
                scores.append(0.0)
                continue

            sim       = torch.mm(gen_emb, ref_emb.T)
            precision = sim.max(dim=1).values.mean().item()
            recall    = sim.max(dim=0).values.mean().item()
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall > 0 else 0.0)
            scores.append(f1)
    bert_model.to('cpu')
    torch.cuda.empty_cache()
    return torch.tensor(scores, dtype=torch.float)


def _common_prefix_len(texts):
    """计算组内每条文本与其他文本的最大公共前缀长度（小写分词）。"""
    word_lists = [t.lower().split() for t in texts]
    n = len(word_lists)
    max_shared = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = word_lists[i], word_lists[j]
            k, min_len = 0, min(len(wi), len(wj))
            while k < min_len and wi[k] == wj[k]:
                k += 1
            if k > max_shared[i]: max_shared[i] = k
            if k > max_shared[j]: max_shared[j] = k
    return max_shared


def compute_r_rep(generated_texts, rep_penalty=0.1,
                  prefix_penalty=0.3, prefix_free_len=6):
    """
    词内重复惩罚 + 组内前缀重复惩罚。
    前缀重复超过 prefix_free_len 词后按指数函数扣分。
    """
    shared_lens = _common_prefix_len(generated_texts)
    scores = []
    for t, shared in zip(generated_texts, shared_lens):
        words    = t.lower().split()
        n_repeat = len(words) - len(set(words))
        score    = -n_repeat * rep_penalty
        excess   = shared - prefix_free_len
        if excess > 0:
            score -= math.exp(excess * prefix_penalty) - 1
        scores.append(score)
    return torch.tensor(scores, dtype=torch.float)


def compute_r_format(generated_texts, prompt_texts, target_min=25, target_max=50,
                     format_penalty=2.0):
    """
    格式惩罚（必须以 '### The user would' 开头）
    + 长度惩罚（指数，偏离 [target_min, target_max] 扣分）
    + 'business' 出现惩罚（每次 -0.5）
    + 逗号惩罚（每个 -0.1）
    + Business title 词命中奖励（每词 +1.0）
    """
    scores = []
    for t, prompt in zip(generated_texts, prompt_texts):
        score = 0.0 if t.startswith("### The user would") else -format_penalty

        n = len(t.split())
        if n < target_min:
            score += -(math.exp((target_min - n) * 0.1) - 1)
        elif n > target_max:
            score += -(math.exp((n - target_max) * 0.1) - 1)

        score -= t.lower().count('business') * 0.5
        score -= t.count(',') * 0.1

        try:
            s = prompt.find("Business title:") + len("Business title:")
            e = prompt.find(". Business profile:")
            if s > len("Business title:") - 1 and e > s:
                title_words = set(prompt[s:e].strip().lower().split())
                score += len(title_words & set(t.lower().split())) * 1.0
        except Exception:
            pass

        scores.append(score)
    return torch.tensor(scores, dtype=torch.float)


def compute_r_path(generated_texts, path_entities, path_reward=0.3):
    """
    R_path：生成文本命中 KG 路径实体节点词，每词 +path_reward。
    鼓励模型利用协同过滤路径中的具体信息（菜名、餐厅名、口味词等）。
    path_entities：set，从当前 prompt 路径中提取的实体词。
    """
    scores = []
    for t in generated_texts:
        hits = len(path_entities & _clean_words(t))
        scores.append(hits * path_reward)
    return torch.tensor(scores, dtype=torch.float)


def compute_r_entity(generated_texts, profile_entities, entity_reward=0.3):
    """
    R_entity：生成文本命中 Business profile 关键词，每词 +entity_reward。
    鼓励模型生成具体细节（菜名、食材、氛围词等），使输出更详细。
    profile_entities：set，从 Business profile 字段提取的关键词。
    """
    scores = []
    for t in generated_texts:
        hits = len(profile_entities & _clean_words(t))
        scores.append(hits * entity_reward)
    return torch.tensor(scores, dtype=torch.float)


def compute_reward(generated_texts, reference_texts, prompt_texts,
                   bart_scorer, bert_model, bert_tokenizer, device,
                   alpha, beta, gamma, delta, epsilon, zeta,
                   rep_penalty, len_min, len_max, format_penalty,
                   prefix_penalty, prefix_free_len,
                   path_entities, path_reward,
                   profile_entities, entity_reward):
    r_bart   = compute_r_bart(generated_texts, reference_texts, bart_scorer)
    r_bert   = compute_r_bert(generated_texts, reference_texts,
                               bert_model, bert_tokenizer, device)
    r_format = compute_r_format(generated_texts, prompt_texts,
                                 len_min, len_max, format_penalty)
    r_rep    = compute_r_rep(generated_texts, rep_penalty,
                              prefix_penalty, prefix_free_len)
    r_path   = compute_r_path(generated_texts, path_entities, path_reward)
    r_entity = compute_r_entity(generated_texts, profile_entities, entity_reward)
    total    = (alpha   * r_bart
                + beta  * r_bert
                + gamma * r_format
                + delta * r_rep
                + epsilon * r_path
                + zeta    * r_entity)
    return total, r_bart, r_bert, r_format, r_rep, r_path, r_entity


# ─────────────────────────────────────────────────────────────
# 7. GRPO loss + KL 约束
# ─────────────────────────────────────────────────────────────
def grpo_loss(model, ref_model, tokenizer, prompt, generated_texts,
              advantages, device, kl_coef=0.1):
    """
    L = mean_i( advantage_i * NLL_i  +  kl_coef * KL_i )

    KL_i ≈ mean_t( log π_θ(t) - log π_ref(t) ) = (-NLL_i) - logp_ref_i

    ref_model 在循环外整体搬到 GPU 一次，G 条 response 全算完后移回 CPU，
    避免每条都搬运整个模型导致卡顿。
    """
    model.train()
    loss_list = []
    kl_vals   = []

    # ── ref model 整体搬到 GPU，循环结束后移回 ───────────────
    ref_model.to(device)

    for gen_text, adv in zip(generated_texts, advantages):
        if not gen_text.strip():
            continue

        input_ids, labels, prompt_len, response_len = build_input_labels(
            tokenizer, prompt, gen_text)

        if response_len == 0:
            continue

        input_ids = input_ids.unsqueeze(0).to(device)
        labels    = labels.unsqueeze(0).to(device)

        # ── policy NLL ──────────────────────────────────────
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            nll_i = model(input_ids=input_ids, labels=labels).loss

        if torch.isnan(nll_i) or torch.isinf(nll_i):
            continue

        # ── ref log prob（token 级均值）──────────────────────
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            ref_logits = ref_model(input_ids=input_ids).logits
        shift_logits = ref_logits[0, prompt_len - 1 : prompt_len + response_len - 1, :]
        shift_tokens = input_ids[0, prompt_len : prompt_len + response_len]
        log_probs    = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        logp_ref     = log_probs[range(len(shift_tokens)), shift_tokens].mean().detach()

        # KL ≈ log π_θ - log π_ref；NLL = -log π_θ 所以 log π_θ = -NLL
        kl_i = (-nll_i) - logp_ref
        kl_vals.append(kl_i.item())

        adv_t = torch.tensor(adv, device=device, dtype=nll_i.dtype)
        loss_list.append(adv_t * nll_i + kl_coef * kl_i)

    # ── ref model 移回 CPU ───────────────────────────────────
    ref_model.to('cpu')
    torch.cuda.empty_cache()

    if not loss_list:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    mean_kl = sum(kl_vals) / len(kl_vals)
    return torch.stack(loss_list).mean(), mean_kl


# ─────────────────────────────────────────────────────────────
# 8. 主训练循环
# ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"[init] device={device}  G={args.G}  lr={args.lr}  kl_coef={args.kl_coef}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model       = load_model(args.base_model_path, args.sft_model_path, tokenizer)
    ref_model   = load_ref_model(args.base_model_path, args.sft_model_path, tokenizer)
    bart_scorer = load_bart_scorer(device)
    bert_model, bert_tokenizer = load_bert_model()

    dataset = GRPODataset(args.raft_path, args.max_samples, args.sft_used_samples)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.num_epochs):
        total_loss, total_reward, steps = 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for batch in pbar:
            prompt  = batch['prompt'][0]
            chosen  = batch['chosen'][0]
            # DataLoader collate 后是 list of list of str，取第0个转回 set
            path_entities    = set(batch['path_entities'][0])
            profile_entities = set(batch['profile_entities'][0])

            # ── 采样 G 个输出 ────────────────────────────────
            gen_texts = sample_outputs(
                model, tokenizer, prompt,
                args.G, args.max_new_tokens, device)

            # ── 计算奖励 ─────────────────────────────────────
            rewards, r_bart, r_bert, r_format, r_rep, r_path, r_entity = compute_reward(
                gen_texts, [chosen] * args.G, [prompt] * args.G,
                bart_scorer, bert_model, bert_tokenizer, device,
                args.alpha, args.beta, args.gamma, args.delta,
                args.epsilon, args.zeta,
                args.rep_penalty, args.len_min, args.len_max, args.format_penalty,
                args.prefix_penalty, args.prefix_free_len,
                path_entities, args.path_reward,
                profile_entities, args.entity_reward)

            # ── 组内归一化 → advantage ───────────────────────
            mean_r     = rewards.mean()
            std_r      = rewards.std() + 1e-8
            advantages = ((rewards - mean_r) / std_r).tolist()

            # ── loss & 反向传播 ──────────────────────────────
            optimizer.zero_grad()
            loss, mean_kl = grpo_loss(
                model, ref_model, tokenizer, prompt,
                gen_texts, advantages, device, kl_coef=args.kl_coef)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val   = loss.item()
            reward_val = mean_r.item()

            # ── 调试信息（在 grpo_loss 之后打印，KL 已有真实值）──
            if steps % args.debug_every == 0:
                best_idx = rewards.argmax().item()
                print(f"\n[debug step={steps}]")
                print(f"  chosen : {chosen[:100]!r}")
                print(f"  {'─'*56}")
                for i, (t, r, adv) in enumerate(
                        zip(gen_texts, rewards.tolist(), advantages)):
                    tag    = " ← BEST" if i == best_idx else ""
                    fmt_ok = "✓" if t.startswith("### The user would") else "✗"
                    print(f"  [{i}] reward={r:.4f}  adv={adv:+.4f}  "
                          f"len={len(t.split())}w  fmt={fmt_ok}{tag}")
                    print(f"       {t[:100]!r}")
                print(f"  {'─'*56}")
                print(f"  r_bart   : {[f'{v:.4f}' for v in r_bart.tolist()]}")
                print(f"  r_bert   : {[f'{v:.4f}' for v in r_bert.tolist()]}")
                print(f"  r_format : {[f'{v:.4f}' for v in r_format.tolist()]}")
                print(f"  r_rep    : {[f'{v:.4f}' for v in r_rep.tolist()]}")
                print(f"  r_path   : {[f'{v:.4f}' for v in r_path.tolist()]}  "
                      f"path_entities({len(path_entities)}): {sorted(path_entities)[:10]}")
                print(f"  r_entity : {[f'{v:.4f}' for v in r_entity.tolist()]}  "
                      f"profile_entities({len(profile_entities)}): {sorted(profile_entities)[:10]}")
                shared_lens = _common_prefix_len(gen_texts)
                for i, t in enumerate(gen_texts):
                    n       = len(t.split())
                    fmt_pen = 0.0 if t.startswith("### The user would") else -args.format_penalty
                    len_pen = (-(math.exp((args.len_min - n) * 0.1) - 1) if n < args.len_min
                               else -(math.exp((n - args.len_max) * 0.1) - 1) if n > args.len_max
                               else 0.0)
                    words   = t.lower().split()
                    n_rep   = len(words) - len(set(words))
                    excess  = shared_lens[i] - args.prefix_free_len
                    pfx_pen = -(math.exp(excess * args.prefix_penalty) - 1) if excess > 0 else 0.0
                    print(f"    [{i}] fmt={fmt_pen:.2f}  len={len_pen:.2f}  "
                          f"rep={n_rep}x({-n_rep*args.rep_penalty:.2f})  "
                          f"pfx={shared_lens[i]}w({pfx_pen:.2f})  words={n}")
                print(f"  mean_reward={mean_r.item():.4f}  std={std_r.item():.4f}  "
                      f"kl={mean_kl:.4f}  loss={loss_val:.4f}")

            total_loss   += loss_val
            total_reward += reward_val
            steps += 1

            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             reward=f"{reward_val:.4f}",
                             kl=f"{mean_kl:.4f}",
                             bart=f"{r_bart.mean().item():.4f}",
                             bert=f"{r_bert.mean().item():.4f}")

            del loss, gen_texts, rewards, r_bart, r_bert, r_format, r_rep, r_path, r_entity
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
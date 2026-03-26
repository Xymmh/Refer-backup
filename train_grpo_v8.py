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
    p.add_argument('--sft_model_path', default='outputs/collab_llama_yelp_gnn_v2/epoch2')
    p.add_argument('--base_model_path', default='meta-llama/Llama-3.2-3B')
    p.add_argument('--raft_path', default='raft_data/yelp/train.json')
    p.add_argument('--output_dir', default='outputs/grpo_llama_yelp_v8')
    p.add_argument('--max_samples', type=int, default=500)
    p.add_argument('--num_epochs', type=int, default=1)
    p.add_argument('--G', type=int, default=6,
                   help='每个 prompt 采样的输出数（组内归一化）')
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--lr', type=float, default=2e-6)
    p.add_argument('--kl_coef', type=float, default=5.0,
                   help='KL 散度惩罚系数，约束策略不偏离参考模型')
    # 奖励权重
    p.add_argument('--alpha', type=float, default=0.5, help='R_bart 权重')
    p.add_argument('--beta', type=float, default=1.0, help='R_bert 权重')
    p.add_argument('--gamma', type=float, default=0.05, help='R_format 权重')
    p.add_argument('--delta', type=float, default=0.15, help='R_rep 权重')
    p.add_argument('--epsilon', type=float, default=0.35, help='R_path 权重')
    p.add_argument('--zeta', type=float, default=0.35, help='R_entity 权重')
    # 格式 / 长度惩罚
    p.add_argument('--format_penalty', type=float, default=2.0,
                   help='缺少 ### The user would 开头时的固定扣分')
    p.add_argument('--len_min', type=int, default=40, help='长度惩罚下限（词数）')
    p.add_argument('--len_max', type=int, default=60, help='长度惩罚上限（词数）')
    # 重复惩罚
    p.add_argument('--rep_penalty', type=float, default=0.05, help='每个重复词的扣分')
    p.add_argument('--prefix_penalty', type=float, default=0.3, help='组内前缀重复的指数惩罚速率')
    p.add_argument('--prefix_free_len', type=int, default=10,
                   help='组内前缀重复的免罚词数（默认前6词相同不扣分）')
    # 实体命中奖励
    p.add_argument('--path_reward', type=float, default=0.3, help='路径实体每命中一词的加分')
    p.add_argument('--entity_reward', type=float, default=0.3, help='profile 实体每命中一词的加分')
    p.add_argument('--eta', type=float, default=0.15, help='R_faithful 权重')
    p.add_argument('--halluc_penalty', type=float, default=0.3,
                   help='每个幻觉专有名词的扣分（生成了但 prompt 里找不到）')
    p.add_argument('--debug_every', type=int, default=1)
    return p.parse_args()
# ─────────────────────────────────────────────────────────────
# 2. 数据集
# ─────────────────────────────────────────────────────────────
# 路径段标记（数据中实际使用的字符串）
PATH_MARKER = "### For the given user-item pair, here are several related paths"
STOPWORDS = {
    'a',
    'about',
    'activities',
    'affordable',
    'all',
    'along',
    'also',
    'ambiance',
    'amenities',
    'american',
    'an',
    'and',
    'any',
    'appealing',
    'appreciate',
    'are',
    'area',
    'as',
    'asian',
    'at',
    'atmosphere',
    'atmospheres',
    'attention',
    'attentive',
    'authentic',
    'back',
    'bar',
    'bars',
    'based',
    'bbq',
    'be',
    'been',
    'beer',
    'beers',
    'best',
    'beverages',
    'both',
    'breakfast',
    'brunch',
    'burgers',
    'business',
    'businesses',
    'but',
    'by',
    'byob',
    'cafe',
    'cafes',
    'cajun',
    'can',
    'casual',
    'chicken',
    'chinese',
    'choices',
    'classic',
    'clean',
    'cocktails',
    'coffee',
    'comfort',
    'comfortable',
    'community',
    'convenient',
    'could',
    'cozy',
    'craft',
    'cream',
    'creative',
    'creole',
    'cuisine',
    'cuisines',
    'customer',
    'customizable',
    'deals',
    'decor',
    'delicious',
    'desserts',
    'detail',
    'did',
    'different',
    'dining',
    'dishes',
    'diverse',
    'do',
    'drink',
    'drinks',
    'during',
    'each',
    'eastern',
    'efficient',
    'enjoy',
    'entertainment',
    'enthusiasts',
    'environment',
    'environments',
    'especially',
    'establishments',
    'events',
    'excellent',
    'exceptional',
    'experience',
    'experiences',
    'extensive',
    'families',
    'family',
    'fans',
    'fast',
    'find',
    'fine',
    'flavorful',
    'flavors',
    'focus',
    'food',
    'for',
    'free',
    'french',
    'fresh',
    'fried',
    'friendly',
    'from',
    'fun',
    'fusion',
    'games',
    'generous',
    'get',
    'give',
    'gluten',
    'good',
    'goods',
    'gourmet',
    'great',
    'grill',
    'had',
    'happy',
    'has',
    'have',
    'he',
    'healthy',
    'hearty',
    'her',
    'high',
    'his',
    'homemade',
    'hour',
    'house',
    'how',
    'i',
    'ice',
    'ideal',
    'if',
    'in',
    'including',
    'indian',
    'indianapolis',
    'individuals',
    'ingredients',
    'interactive',
    'intimate',
    'into',
    'is',
    'it',
    'italian',
    'item',
    'items',
    'its',
    'japanese',
    'just',
    'knowledgeable',
    'laid',
    'large',
    'latin',
    'like',
    'likely',
    'live',
    'lively',
    'local',
    'located',
    'location',
    'locations',
    'looking',
    'louis',
    'love',
    'lovers',
    'low',
    'made',
    'make',
    'may',
    'meals',
    'mediterranean',
    'menu',
    'menus',
    'mexican',
    'might',
    'mix',
    'modern',
    'money',
    'more',
    'music',
    'my',
    'nashville',
    'need',
    'neighborhood',
    'new',
    'night',
    'nightlife',
    'no',
    'not',
    'of',
    'offer',
    'offering',
    'offerings',
    'offers',
    'on',
    'options',
    'or',
    'orleans',
    'our',
    'out',
    'outdoor',
    'particularly',
    'pastries',
    'personalized',
    'philadelphia',
    'pizza',
    'place',
    'places',
    'portion',
    'portions',
    'price',
    'prices',
    'products',
    'profile',
    'pub',
    'quality',
    'quick',
    'range',
    'reasonable',
    'relaxed',
    'restaurant',
    'restaurants',
    'saint',
    'sandwiches',
    'seafood',
    'seating',
    'seeking',
    'selection',
    'selections',
    'service',
    'services',
    'setting',
    'settings',
    'she',
    'shops',
    'sizes',
    'small',
    'so',
    'southern',
    'special',
    'specials',
    'specialty',
    'sports',
    'spot',
    'spots',
    'staff',
    'street',
    'style',
    'such',
    'sushi',
    'tacos',
    'tampa',
    'tasty',
    'tea',
    'thai',
    'than',
    'that',
    'the',
    'their',
    'then',
    'they',
    'this',
    'those',
    'through',
    'to',
    'top',
    'toppings',
    'traditional',
    'treats',
    'trendy',
    'trying',
    'unique',
    'up',
    'upscale',
    'user',
    'users',
    'value',
    'variety',
    'vegan',
    'vegetarian',
    'venues',
    'very',
    'vibe',
    'vibrant',
    'vietnamese',
    'views',
    'want',
    'was',
    'we',
    'welcoming',
    'well',
    'were',
    'what',
    'when',
    'where',
    'which',
    'who',
    'whom',
    'why',
    'wide',
    'will',
    'willing',
    'wine',
    'with',
    'would',
    'yet',
    'you',
    'your',
}
def _clean_words(text):
    """小写、只保留字母词（≥3字符）、去除停用词，通用预处理。"""
    return {w for w in re.findall(r'[a-z]+', text.lower())
            if w not in STOPWORDS and len(w) >= 3}
def _extract_path_entities(prompt):
    """
    从路径段的所有 Item Profile 括号中提取关键词。
    实际路径格式：
      User (Profile: ...) -> buys -> Item (Profile: ...) -> bought by -> User (Profile: ...) -> buys -> Item (Profile: ...)
    策略：提取所有 "Item (Profile: ...)" 括号内的词——这些是协同过滤
    路径中途经的 item profile，包含具体菜名、餐厅特色等协同信息。
    只取 Item Profile（不取 User Profile），避免引入泛化的用户偏好描述。
    """
    start = prompt.find(PATH_MARKER)
    if start == -1:
        return set()
    path_text = prompt[start + len(PATH_MARKER):]
    # 提取所有 Item (Profile: ...) 中的内容
    item_profiles = re.findall(r'Item\s*\(Profile:\s*(.*?)\)', path_text, re.DOTALL)
    entities = set()
    for prof in item_profiles:
        entities.update(_clean_words(prof))
    return entities
def _extract_profile_entities(prompt):
    """
    从 prompt 开头的 Business profile 字段提取关键词。
    格式：'Business profile: <内容>. User profile:'
    取第一个 'Business profile:' 到 ' User profile:' 之间的内容。
    这段文字直接描述目标 item 的特色（菜系、口味、氛围等），
    是最直接的"应该出现在生成结果中"的词汇来源。
    """
    start = prompt.find("Business profile:")
    if start == -1:
        return set()
    start += len("Business profile:")
    # 结束位置：User profile: 或 \n### 之前
    end = prompt.find(" User profile:", start)
    if end == -1:
        end = prompt.find("\n###", start)
    segment = prompt[start:end].strip() if end != -1 else prompt[start:].strip()
    return _clean_words(segment)
class GRPODataset(Dataset):
    def __init__(self, raft_path, max_samples=-1):
        all_samples = []
        with open(raft_path) as f:
            for line in f:
                d = json.loads(line)
                all_samples.append({'prompt': d['prompt'], 'chosen': d['chosen']})
        candidates = [s for s in all_samples if PATH_MARKER in s['prompt']]
        import random
        if max_samples > 0:
            random.shuffle(candidates)
            self.samples = candidates[:max_samples]
        else:
            self.samples = candidates
        # 预计算路径实体和 profile 实体，存为空格拼接的字符串（避免 DataLoader 对 list 转置）
        for s in self.samples:
            s['path_entities'] = ' '.join(sorted(_extract_path_entities(s['prompt'])))
            s['profile_entities'] = ' '.join(sorted(_extract_profile_entities(s['prompt'])))
        pe_avg = sum(len(s['path_entities'].split()) for s in self.samples) / max(len(self.samples), 1)
        en_avg = sum(len(s['profile_entities'].split()) for s in self.samples) / max(len(self.samples), 1)
        print(f"[data] total={len(all_samples)} | "
              f"has_path={len(candidates)} | "
              f"GRPO using={len(self.samples)} (random sample, no SFT skip) | "
              f"avg path_entities={pe_avg:.1f} avg profile_entities={en_avg:.1f}")
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
    加载冻结的参考模型，始终在 CPU 上推理，不占任何显存。
    使用 float32（CPU 原生支持，bfloat16 在 CPU 上会回退到 float32 模拟，反而更慢）。
    grpo_loss 将 G 条 response 批量化后一次 forward，避免逐条推理。
    """
    print(f"[ref] Loading reference model from {sft_path} (frozen, CPU, float32) ...")
    base = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float32,
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
    bert_model = BertModel.from_pretrained(model_name)
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
    labels ：prompt 部分=-100，response+eos 参与 loss
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
    response_ids = tokenizer(response_text, add_special_tokens=False)['input_ids']
    eos_id = tokenizer.eos_token_id
    input_ids = prompt_ids + response_ids + [eos_id]
    labels = [-100] * len(prompt_ids) + response_ids + [eos_id]
    assert len(input_ids) == len(labels)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
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
            sim = torch.mm(gen_emb, ref_emb.T)
            precision = sim.max(dim=1).values.mean().item()
            recall = sim.max(dim=0).values.mean().item()
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
        words = t.lower().split()
        n_repeat = len(words) - len(set(words))
        score = -n_repeat * rep_penalty
        excess = shared - prefix_free_len
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
def compute_r_faithful(generated_texts, prompt_texts, halluc_penalty=0.3):
    """
    R_faithful：忠实度奖励，通过 content bigram 检测幻觉。
    对生成文本中相邻的 content word pair，若两端词都不在 prompt 词表里，
    视为幻觉词对，每对扣 halluc_penalty 分。
    原理：真实信息（菜名、地名、特色词）应来自 prompt，
    "al pastor"、"salsa verde" 等 prompt 里没有的词组合视为幻觉。
    大小写不敏感，覆盖所有词而非只看大写专有名词。
    """
    scores = []
    for t, prompt in zip(generated_texts, prompt_texts):
        prompt_words = set(re.findall(r'[a-z]+', prompt.lower()))
        gen_content = [w for w in re.findall(r'[a-z]+', t.lower())
                        if w not in STOPWORDS and len(w) >= 3]
        halluc = sum(
            1 for i in range(len(gen_content) - 1)
            if gen_content[i] not in prompt_words
            and gen_content[i+1] not in prompt_words
        )
        scores.append(-halluc * halluc_penalty)
    return torch.tensor(scores, dtype=torch.float)
def compute_reward(generated_texts, reference_texts, prompt_texts,
                   bart_scorer, bert_model, bert_tokenizer, device,
                   alpha, beta, gamma, delta, epsilon, zeta, eta,
                   rep_penalty, len_min, len_max, format_penalty,
                   prefix_penalty, prefix_free_len,
                   path_entities, path_reward,
                   profile_entities, entity_reward,
                   halluc_penalty):
    r_bart = compute_r_bart(generated_texts, reference_texts, bart_scorer)
    r_bert = compute_r_bert(generated_texts, reference_texts,
                                 bert_model, bert_tokenizer, device)
    r_format = compute_r_format(generated_texts, prompt_texts,
                                   len_min, len_max, format_penalty)
    r_rep = compute_r_rep(generated_texts, rep_penalty,
                                prefix_penalty, prefix_free_len)
    r_path = compute_r_path(generated_texts, path_entities, path_reward)
    r_entity = compute_r_entity(generated_texts, profile_entities, entity_reward)
    r_faithful = compute_r_faithful(generated_texts, prompt_texts, halluc_penalty)
    total = (alpha * r_bart
                  + beta * r_bert
                  + gamma * r_format
                  + delta * r_rep
                  + epsilon * r_path
                  + zeta * r_entity
                  + eta * r_faithful)
    return total, r_bart, r_bert, r_format, r_rep, r_path, r_entity, r_faithful
# ─────────────────────────────────────────────────────────────
# 7. GRPO loss + KL 约束（KL 加在 loss 上）
# ─────────────────────────────────────────────────────────────
def grpo_loss(model, ref_model, tokenizer, prompt, generated_texts,
              advantages, device, kl_coef=0.1, ref_ctx_len=64):
    """
    L = mean_i( advantage_i × NLL_i + kl_coef × KL_i )
    KL_i = mean_t[ log π_θ(t) - log π_ref(t) ]，保证非负（clip(KL, 0)）。
    关键修复：policy logp 和 ref logp 必须用完全相同的截断 context 计算，
    否则条件分布不一致，KL 会出现负值。
    两者都只用 prompt 末尾 ref_ctx_len 个 token 作为条件前缀。
    速度优化：ref model G 条 response 批量一次 CPU forward。
    """
    model.train()
    # ── Step 1：准备截断序列 ─────────────────────────────────
    entries = [] # (orig_idx, trimmed_ids, ctx_len, response_len, adv)
    for i, (gen_text, adv) in enumerate(zip(generated_texts, advantages)):
        if not gen_text.strip():
            continue
        input_ids, _, prompt_len, response_len = build_input_labels(
            tokenizer, prompt, gen_text)
        if response_len == 0:
            continue
        ctx = min(ref_ctx_len, prompt_len)
        # 截断序列：prompt末尾ctx个token + response
        trimmed = input_ids[prompt_len - ctx : prompt_len + response_len]
        entries.append((i, trimmed, ctx, response_len, adv, input_ids, prompt_len))
    if not entries:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0
    # ── Step 2：policy logp（GPU，截断context，no_grad，eval模式）─
    # 必须切到 eval() 再切回，否则 gradient_checkpointing 下
    # no_grad 无法保证结果与 ref 对齐，导致 KL 为负。
    model.eval()
    logp_theta_list = []
    for (i, trimmed, ctx, response_len, adv, full_ids, prompt_len) in entries:
        ids_gpu = trimmed.unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids=ids_gpu).logits[0] # (ctx+rl, vocab)
        lp = torch.nn.functional.log_softmax(
                 logits[ctx-1 : ctx+response_len-1].float(), dim=-1)
        tokens = trimmed[ctx : ctx+response_len]
        logp_theta_list.append(lp[range(len(tokens)), tokens].mean().item())
    model.train()
    # ── Step 3：ref logp（CPU batch forward）─────────────────
    trimmed_list = [e[1] for e in entries]
    max_len = max(t.shape[0] for t in trimmed_list)
    pad_id = tokenizer.pad_token_id or 0
    batch_cpu = torch.full((len(trimmed_list), max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros(len(trimmed_list), max_len, dtype=torch.long)
    for k, t in enumerate(trimmed_list):
        batch_cpu[k, :t.shape[0]] = t
        attn_mask[k, :t.shape[0]] = 1
    with torch.no_grad():
        ref_logits_batch = ref_model(
            input_ids=batch_cpu, attention_mask=attn_mask).logits # (G, max_len, vocab)
    logp_ref_list = []
    for k, (i, trimmed, ctx, response_len, adv, full_ids, prompt_len) in enumerate(entries):
        lp = torch.nn.functional.log_softmax(
                  ref_logits_batch[k, ctx-1 : ctx+response_len-1, :].float(), dim=-1)
        tokens = trimmed[ctx : ctx+response_len]
        logp_ref_list.append(lp[range(len(tokens)), tokens].mean().item())
    # ── Step 4：policy NLL（GPU，完整context，有梯度）+ loss ──
    loss_list = []
    kl_vals = []
    for k, (i, trimmed, ctx, response_len, adv, full_ids, prompt_len) in enumerate(entries):
        # NLL 用完整序列保证训练质量
        ids_gpu = full_ids.unsqueeze(0).to(device)
        labels_gpu = full_ids.clone()
        labels_gpu[:prompt_len] = -100
        labels_gpu = labels_gpu.unsqueeze(0).to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            nll_i = model(input_ids=ids_gpu, labels=labels_gpu).loss
        if torch.isnan(nll_i) or torch.isinf(nll_i):
            continue
        # KL：用截断context算的logp，保证符号正确
        kl_i = max(logp_theta_list[k] - logp_ref_list[k], 0.0)
        kl_vals.append(kl_i)
        adv_t = torch.tensor(adv, device=device, dtype=nll_i.dtype)
        kl_t = torch.tensor(kl_i, device=device, dtype=nll_i.dtype)
        loss_list.append(adv_t * nll_i + kl_coef * kl_t)
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
    print(f"[init] device={device} G={args.G} lr={args.lr} kl_coef={args.kl_coef}")
    print(f"{'='*60}\n")
    tokenizer = AutoTokenizer.from_pretrained(
        args.sft_model_path, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.base_model_path, args.sft_model_path, tokenizer)
    ref_model = load_ref_model(args.base_model_path, args.sft_model_path, tokenizer)
    bart_scorer = load_bart_scorer(device)
    bert_model, bert_tokenizer = load_bert_model()
    dataset = GRPODataset(args.raft_path, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(args.num_epochs):
        total_loss, total_reward, steps = 0.0, 0.0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            prompt = batch['prompt'][0]
            chosen = batch['chosen'][0]
            # DataLoader 对 str collate 后是 list of str（每个字符），取 batch[key][0] 是完整字符串
            path_entities = set(batch['path_entities'][0].split())
            profile_entities = set(batch['profile_entities'][0].split())
            # ── 采样 G 个输出 ────────────────────────────────
            gen_texts = sample_outputs(
                model, tokenizer, prompt,
                args.G, args.max_new_tokens, device)
            # ── 计算奖励 ─────────────────────────────────────
            rewards, r_bart, r_bert, r_format, r_rep, r_path, r_entity, r_faithful = compute_reward(
                gen_texts, [chosen] * args.G, [prompt] * args.G,
                bart_scorer, bert_model, bert_tokenizer, device,
                args.alpha, args.beta, args.gamma, args.delta,
                args.epsilon, args.zeta, args.eta,
                args.rep_penalty, args.len_min, args.len_max, args.format_penalty,
                args.prefix_penalty, args.prefix_free_len,
                path_entities, args.path_reward,
                profile_entities, args.entity_reward,
                args.halluc_penalty)
            # ── 组内归一化 → advantage ───────────────────────
            mean_r = rewards.mean()
            std_r = rewards.std() + 1e-8
            advantages = ((rewards - mean_r) / std_r).tolist()
            # ── loss & 反向传播 ──────────────────────────────
            optimizer.zero_grad()
            loss, mean_kl = grpo_loss(
                model, ref_model, tokenizer, prompt,
                gen_texts, advantages, device,
                kl_coef=args.kl_coef)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_val = loss.item()
            reward_val = mean_r.item()
            # ── 调试信息（在 grpo_loss 之后打印，KL 已有真实值）──
            if steps % args.debug_every == 0:
                best_idx = rewards.argmax().item()
                print(f"\n[debug step={steps}]")
                print(f" chosen : {chosen[:100]!r}")
                print(f" {'─'*56}")
                for i, (t, r, adv) in enumerate(
                        zip(gen_texts, rewards.tolist(), advantages)):
                    tag = " ← BEST" if i == best_idx else ""
                    fmt_ok = "✓" if t.startswith("### The user would") else "✗"
                    print(f" [{i}] reward={r:.4f} adv={adv:+.4f} "
                          f"len={len(t.split())}w fmt={fmt_ok}{tag}")
                    print(f" {t[:100]!r}")
                print(f" {'─'*56}")
                print(f" r_bart : {[f'{v:.4f}' for v in r_bart.tolist()]}")
                print(f" r_bert : {[f'{v:.4f}' for v in r_bert.tolist()]}")
                print(f" r_format : {[f'{v:.4f}' for v in r_format.tolist()]}")
                print(f" r_rep : {[f'{v:.4f}' for v in r_rep.tolist()]}")
                print(f" r_path : {[f'{v:.4f}' for v in r_path.tolist()]} "
                      f"path_entities({len(path_entities)}): {sorted(path_entities)[:10]}")
                print(f" r_entity : {[f'{v:.4f}' for v in r_entity.tolist()]} "
                      f"profile_entities({len(profile_entities)}): {sorted(profile_entities)[:10]}")
                print(f" r_faithful: {[f'{v:.4f}' for v in r_faithful.tolist()]}")
                shared_lens = _common_prefix_len(gen_texts)
                for i, t in enumerate(gen_texts):
                    n = len(t.split())
                    fmt_pen = 0.0 if t.startswith("### The user would") else -args.format_penalty
                    len_pen = (-(math.exp((args.len_min - n) * 0.1) - 1) if n < args.len_min
                               else -(math.exp((n - args.len_max) * 0.1) - 1) if n > args.len_max
                               else 0.0)
                    words = t.lower().split()
                    n_rep = len(words) - len(set(words))
                    excess = shared_lens[i] - args.prefix_free_len
                    pfx_pen = -(math.exp(excess * args.prefix_penalty) - 1) if excess > 0 else 0.0
                    print(f" [{i}] fmt={fmt_pen:.2f} len={len_pen:.2f} "
                          f"rep={n_rep}x({-n_rep*args.rep_penalty:.2f}) "
                          f"pfx={shared_lens[i]}w({pfx_pen:.2f}) words={n}")
                print(f" mean_reward={mean_r.item():.4f} std={std_r.item():.4f} "
                      f"kl={mean_kl:.4f} loss={loss_val:.4f}")
            total_loss += loss_val
            total_reward += reward_val
            steps += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             reward=f"{reward_val:.4f}",
                             kl=f"{mean_kl:.4f}",
                             bart=f"{r_bart.mean().item():.4f}",
                             bert=f"{r_bert.mean().item():.4f}")
            del loss, gen_texts, rewards, r_bart, r_bert, r_format, r_rep, r_path, r_entity, r_faithful
            torch.cuda.empty_cache()
        print(f"\n[epoch {epoch+1}] avg_loss={total_loss/steps:.4f} "
              f"avg_reward={total_reward/steps:.4f}")
        save_path = f"{args.output_dir}/epoch{epoch+1}"
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"[save] → {save_path}")
    print("\nGRPO 训练完成！")
if __name__ == '__main__':
    train(parse_args())
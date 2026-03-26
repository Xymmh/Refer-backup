"""
协同知识蒸馏 SFT 训练脚本 v3

设计原则：
  - prompt 和 chosen 原汁原味输入，绝不截断
  - 仅当 chosen 超过 max_response_tokens 时才截 chosen 尾部（保护loss有效性）
  - 无 padding：batch_size=1，每条样本单独前向
  - prompt / chosen 分别 tokenize 后拼接，边界精确
  - GNN 软token通过 embedding hook 残差注入
  - 梯度累积模拟大batch
"""

import os, json, torch, torch.nn as nn, argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# 1. 协同适配器
# ─────────────────────────────────────────────────────────────
class CollaborativeAdapter(nn.Module):
    def __init__(self, input_dim=128, output_dim=3072, scale=0.1):
        super().__init__()
        self.scale = scale
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, output_dim),
        )
        nn.init.normal_(self.proj[1].weight, std=0.01)
        nn.init.zeros_(self.proj[1].bias)
        nn.init.normal_(self.proj[3].weight, std=0.01)
        nn.init.zeros_(self.proj[3].bias)

    def forward(self, emb):
        return self.proj(emb) * self.scale


# ─────────────────────────────────────────────────────────────
# 2. 数据集
# ─────────────────────────────────────────────────────────────
class CollabDataset(Dataset):
    def __init__(self, raft_path, pyg_path, gnn_emb_path, max_samples=-1):
        gnn = torch.load(gnn_emb_path, map_location='cpu', weights_only=False)
        self.user_emb = gnn['user_emb'].float()
        self.item_emb = gnn['item_emb'].float()
        print(f"[data] GNN user_emb: {self.user_emb.shape}, item_emb: {self.item_emb.shape}")

        pyg = torch.load(pyg_path, map_location='cpu', weights_only=False)
        self.user_id_to_node = pyg.user_id_to_node
        self.item_id_to_node = pyg.item_id_to_node
        self.num_users = len(pyg.user_id_to_node)

        self.samples, skipped = [], 0

        # 读取全部数据
        with open(raft_path) as f:
            all_lines = f.readlines()

        # 从最后一条开始往前取
        if max_samples > 0:
            all_lines = all_lines[-max_samples:]
        all_lines = all_lines[::-1]  # 反转顺序（最后 → 最前）

        for line in all_lines:
            d = json.loads(line)
            uid, iid = d['uid'], d['iid']
            if uid in self.user_id_to_node and iid in self.item_id_to_node:
                self.samples.append({
                    'uid': uid,
                    'iid': iid,
                    'prompt': d['prompt'],
                    'chosen': d['chosen'],
                })
            else:
                skipped += 1

        print(f"[data] Loaded {len(self.samples)} samples, skipped {skipped}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        user_node = self.user_id_to_node[s['uid']]
        item_node = self.item_id_to_node[s['iid']] - self.num_users
        return {
            'user_emb': self.user_emb[user_node],
            'item_emb': self.item_emb[item_node],
            'prompt':   s['prompt'],
            'chosen':   s['chosen'],
        }


# ─────────────────────────────────────────────────────────────
# 3. Tokenize（prompt 完整保留，chosen 超长才截尾部）
# ─────────────────────────────────────────────────────────────
def tokenize_sample(tokenizer, prompt_text, chosen_text,
                    user_token_id, item_token_id,
                    max_response_tokens=256):
    """
    序列格式：
        <USER_EMB> <ITEM_EMB> [prompt_ids] [chosen_ids] <eos>
        |<──── 全部 mask(-100) ────>|<─── 参与 loss ───>|

    截断策略：
        - prompt：完整保留，绝不截断
        - chosen：若 token 数超过 max_response_tokens，截尾部
          （chosen 本身一般 30-80 词，几乎不会触发）
    """
    PREFIX = [user_token_id, item_token_id]

    # 完整 tokenize，无任何长度限制
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
    chosen_ids = tokenizer(chosen_text, add_special_tokens=False)['input_ids']

    # 仅对 chosen 做保护性截断
    truncated = False
    if len(chosen_ids) > max_response_tokens:
        chosen_ids = chosen_ids[:max_response_tokens]
        truncated = True

    eos_id = tokenizer.eos_token_id
    input_ids = PREFIX + prompt_ids + chosen_ids + [eos_id]

    # prompt_boundary：chosen 开始的位置
    prompt_boundary = len(PREFIX) + len(prompt_ids)

    # labels：prefix+prompt 全部 mask，chosen+eos 参与 loss
    labels = [-100] * prompt_boundary + chosen_ids + [eos_id]

    assert len(input_ids) == len(labels)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels,    dtype=torch.long),
        prompt_boundary,
        len(chosen_ids),
        truncated,
    )


# ─────────────────────────────────────────────────────────────
# 4. 训练
# ─────────────────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"[init] device={device}  model={args.model_path}")
    print(f"[init] output={args.output_dir}")
    print(f"[init] max_response_tokens={args.max_response_tokens}  "
          f"accum={args.accum_steps}  epochs={args.num_epochs}")
    print(f"{'='*60}\n")

    # ── tokenizer ──────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<USER_EMB>', '<ITEM_EMB>']})
    user_token_id = tokenizer.convert_tokens_to_ids('<USER_EMB>')
    item_token_id = tokenizer.convert_tokens_to_ids('<ITEM_EMB>')
    print(f"[tokenizer] vocab={len(tokenizer)}  "
          f"user_id={user_token_id}  item_id={item_token_id}")
    assert user_token_id != tokenizer.unk_token_id, "<USER_EMB> 注册失败"
    assert item_token_id != tokenizer.unk_token_id, "<ITEM_EMB> 注册失败"

    # ── 模型 ──────────────────────────────────────────────────
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb, device_map='auto')
    model.resize_token_embeddings(len(tokenizer))
    print(f"[model] hidden_size={model.config.hidden_size}")

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    ))
    model.print_trainable_parameters()

    llm_hidden  = model.config.hidden_size
    embed_layer = model.get_input_embeddings()

    # ── 协同适配器 ───────────────────────────────────────────
    adapter = CollaborativeAdapter(
        input_dim=args.gnn_dim, output_dim=llm_hidden, scale=0.1).to(device)
    print(f"[adapter] gnn_dim={args.gnn_dim} → hidden={llm_hidden}  "
          f"params={sum(p.numel() for p in adapter.parameters()):,}")

    # ── 数据 & 优化器 ────────────────────────────────────────
    dataset = CollabDataset(
        args.raft_path, args.pyg_path, args.gnn_emb_path, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW([
        {'params': adapter.parameters(),
         'lr': args.lr},
        {'params': [p for p in model.parameters() if p.requires_grad],
         'lr': args.lr * 0.1},
    ], weight_decay=0.01)

    print(f"[train] total_samples={len(dataset)}  "
          f"effective_batch={args.accum_steps}\n")

    # ── 训练循环 ─────────────────────────────────────────────
    model.train()
    adapter.train()

    for epoch in range(args.num_epochs):
        total_loss, valid_steps = 0.0, 0
        nan_steps, trunc_steps  = 0, 0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(pbar):
            user_emb = batch['user_emb'].squeeze(0).to(device)
            item_emb = batch['item_emb'].squeeze(0).to(device)
            prompt   = batch['prompt'][0]
            chosen   = batch['chosen'][0]

            # ── 软token ─────────────────────────────────────
            user_soft = adapter(user_emb.unsqueeze(0)).squeeze(0).to(torch.bfloat16)
            item_soft = adapter(item_emb.unsqueeze(0)).squeeze(0).to(torch.bfloat16)

            # ── tokenize ─────────────────────────────────────
            input_ids, labels, prompt_boundary, chosen_len, truncated = \
                tokenize_sample(tokenizer, prompt, chosen,
                                user_token_id, item_token_id,
                                args.max_response_tokens)

            if truncated:
                trunc_steps += 1

            input_ids = input_ids.unsqueeze(0).to(device)
            labels    = labels.unsqueeze(0).to(device)

            # ── 调试（前5步 + 每100步）───────────────────────
            if step < 5 or (step % 100 == 0 and epoch == 0):
                seq_len = input_ids.shape[1]
                n_loss  = (labels[0] != -100).sum().item()
                print(f"\n[debug step={step}]")
                print(f"  total_tokens={seq_len} | prompt_boundary={prompt_boundary} "
                      f"| chosen_tokens={chosen_len} | truncated={truncated}")
                print(f"  loss_tokens={n_loss}  (should == chosen_len+1 for eos)")
                print(f"  prompt tail : {repr(tokenizer.decode(input_ids[0, max(0, prompt_boundary-5):prompt_boundary].tolist()))}")
                print(f"  chosen head : {repr(tokenizer.decode(input_ids[0, prompt_boundary:prompt_boundary+8].tolist()))}")
                first_loss_pos = (labels[0] != -100).nonzero(as_tuple=True)[0]
                if len(first_loss_pos):
                    print(f"  first loss pos={first_loss_pos[0].item()} "
                          f"(must == prompt_boundary={prompt_boundary})")
                print(f"  soft_user : mean={user_soft.mean().item():.5f}  "
                      f"std={user_soft.std().item():.5f}  "
                      f"absmax={user_soft.abs().max().item():.5f}")

            # ── embedding hook ───────────────────────────────
            def make_hook(u_soft, i_soft, u_id, i_id, ids):
                def hook(module, inp, out):
                    out = out.clone()
                    upos = (ids[0] == u_id).nonzero(as_tuple=True)[0]
                    ipos = (ids[0] == i_id).nonzero(as_tuple=True)[0]
                    if len(upos): out[0, upos[0]] = out[0, upos[0]] + u_soft
                    if len(ipos): out[0, ipos[0]] = out[0, ipos[0]] + i_soft
                    return out
                return hook

            handle = embed_layer.register_forward_hook(
                make_hook(user_soft, item_soft,
                          user_token_id, item_token_id, input_ids))
            outputs = model(input_ids=input_ids, labels=labels)
            handle.remove()
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                nan_steps += 1
                if nan_steps <= 5:
                    print(f"\n[warn step={step}] NaN/Inf loss，跳过")
                continue

            # ── 梯度累积 ─────────────────────────────────────
            (loss / args.accum_steps).backward()
            total_loss  += loss.item()
            valid_steps += 1

            is_last = (step + 1 == len(loader))
            if valid_steps % args.accum_steps == 0 or is_last:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(adapter.parameters()) +
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                optimizer.zero_grad()

                avg_loss = total_loss / valid_steps
                pbar.set_postfix(loss=f"{avg_loss:.4f}",
                                 gnorm=f"{grad_norm:.3f}",
                                 upd=valid_steps // args.accum_steps)

                if valid_steps % (args.accum_steps * 50) == 0:
                    print(f"\n[progress step={step}] avg_loss={avg_loss:.4f}  "
                          f"gnorm={grad_norm:.4f}  nan={nan_steps}  "
                          f"trunc={trunc_steps}")

        avg = total_loss / max(valid_steps, 1)
        print(f"\n{'='*60}")
        print(f"[epoch {epoch+1}] avg_loss={avg:.4f}  "
              f"valid={valid_steps}/{len(loader)}  "
              f"nan={nan_steps}  trunc={trunc_steps}")
        print(f"{'='*60}")

        ckpt = os.path.join(args.output_dir, f"epoch{epoch+1}")
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        torch.save(adapter.state_dict(), os.path.join(ckpt, 'adapter.pth'))
        print(f"[save] → {ckpt}\n")


# ─────────────────────────────────────────────────────────────
# 5. 入口
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',          default='meta-llama/Llama-3.2-3B')
    p.add_argument('--raft_path',           default='raft_data/yelp/train.json')
    p.add_argument('--pyg_path',            default='data/yelp/data_trn.pt')
    p.add_argument('--gnn_emb_path',        default='data/yelp/gnn_embeddings.pt')
    p.add_argument('--output_dir',          default='outputs/collab_llama_yelp_gnn_v3')
    p.add_argument('--max_samples',         type=int,   default=40000)
    p.add_argument('--num_epochs',          type=int,   default=2)
    p.add_argument('--lr',                  type=float, default=1e-4)
    p.add_argument('--accum_steps',         type=int,   default=16)
    p.add_argument('--gnn_dim',             type=int,   default=128)
    p.add_argument('--max_response_tokens', type=int,   default=256,
                   help='chosen 超过此 token 数才截尾部，prompt 永远完整保留')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())

# 运行:
#   python train_collab.py --accum_steps 16 --max_samples 40000 --num_epochs 2
"""
协同知识蒸馏训练脚本
- 4bit量化 + LoRA
- 用embedding hook替代inputs_embeds，速度和显存与普通训练相同
- 软token残差加法注入
"""
import os, json, torch, torch.nn as nn, argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# ── 1. 协同适配器 ──────────────────────────────────────────
class CollaborativeAdapter(nn.Module):
    def __init__(self, input_dim=768, output_dim=3072, scale=0.1):
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


# ── 2. 数据集 ──────────────────────────────────────────────
class CollabDataset(Dataset):
    def __init__(self, raft_path, pyg_path, gnn_emb_path, max_samples=-1):
        # 加载 GNN embedding（128维协同信号）
        gnn = torch.load(gnn_emb_path, map_location='cpu', weights_only=False)
        self.user_emb = gnn['user_emb'].float()  # [num_users, 128]
        self.item_emb = gnn['item_emb'].float()  # [num_items, 128]

        # 加载 uid/iid → 节点索引 的映射（从pyg数据里取）
        pyg = torch.load(pyg_path, map_location='cpu', weights_only=False)
        self.user_id_to_node = pyg.user_id_to_node
        self.item_id_to_node = pyg.item_id_to_node
        self.num_users = len(pyg.user_id_to_node)  # item索引偏移量

        self.samples = []
        with open(raft_path) as f:
            for line in f:
                d = json.loads(line)
                uid, iid = d['uid'], d['iid']
                if uid in self.user_id_to_node and iid in self.item_id_to_node:
                    self.samples.append({
                        'uid': uid, 'iid': iid,
                        'prompt': d['prompt'], 'chosen': d['chosen'],
                    })
                if max_samples > 0 and len(self.samples) >= max_samples:
                    break
        print(f"Dataset size: {len(self.samples)}")
        print(f"GNN user_emb: {self.user_emb.shape}, item_emb: {self.item_emb.shape}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        user_node = self.user_id_to_node[s['uid']]
        item_node = self.item_id_to_node[s['iid']] - self.num_users  # 减去user偏移
        return {
            'user_emb': self.user_emb[user_node],  # [128]
            'item_emb': self.item_emb[item_node],  # [128]
            'prompt': s['prompt'], 'chosen': s['chosen'],
        }


# ── 3. 训练 ────────────────────────────────────────────────
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<USER_EMB>', '<ITEM_EMB>']})
    user_token_id = tokenizer.convert_tokens_to_ids('<USER_EMB>')
    item_token_id = tokenizer.convert_tokens_to_ids('<ITEM_EMB>')
    print(f"user_token_id={user_token_id}, item_token_id={item_token_id}")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, quantization_config=bnb, device_map='auto')
    model.resize_token_embeddings(len(tokenizer))

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
    ))
    model.print_trainable_parameters()

    llm_hidden = model.config.hidden_size
    print(f"LLM hidden size: {llm_hidden}")

    adapter = CollaborativeAdapter(
        input_dim=128, output_dim=llm_hidden, scale=0.1).to(device)

    dataset = CollabDataset(args.raft_path, args.pyg_path, args.gnn_emb_path, args.max_samples)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW([
        {'params': adapter.parameters(), 'lr': args.lr},
        {'params': [p for p in model.parameters() if p.requires_grad],
         'lr': args.lr * 0.1},
    ], weight_decay=0.01)

    # 获取embed_tokens层（hook注册在这里）
    embed_layer = model.get_input_embeddings()

    model.train()
    adapter.train()

    for epoch in range(args.num_epochs):
        total_loss, valid_steps, nan_steps = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")

        for step, batch in enumerate(pbar):
            user_emb = batch['user_emb'].to(device)
            item_emb = batch['item_emb'].to(device)
            prompts, chosens = batch['prompt'], batch['chosen']

            user_soft = adapter(user_emb).to(torch.bfloat16)
            item_soft = adapter(item_emb).to(torch.bfloat16)

            if step == 0 and epoch == 0:
                print(f"[diag] soft mean={user_soft.mean().item():.5f} "
                      f"std={user_soft.std().item():.5f} "
                      f"absmax={user_soft.abs().max().item():.5f}")

            texts = [f"<USER_EMB><ITEM_EMB>{p}\n{c}{tokenizer.eos_token}"
                     for p, c in zip(prompts, chosens)]
            enc = tokenizer(texts, return_tensors='pt', padding=True,
                            truncation=True,
                            max_length=args.max_length).to(device)
            input_ids      = enc['input_ids']
            attention_mask = enc['attention_mask']

            # labels：mask掉prompt部分
            labels = input_ids.clone()
            for i, p in enumerate(prompts):
                plen = len(tokenizer(
                    f"<USER_EMB><ITEM_EMB>{p}\n",
                    add_special_tokens=False)['input_ids'])
                labels[i, :min(plen, labels.shape[1])] = -100
            # 如果labels全是-100则跳过（prompt太长被截断）
            if (labels != -100).sum() == 0:
                continue

            # 注册hook：在embed_tokens输出后注入软token
            def make_hook(u_soft, i_soft, u_id, i_id, ids):
                def hook(module, inp, output):
                    out = output.clone()
                    for i in range(ids.shape[0]):
                        upos = (ids[i] == u_id).nonzero(as_tuple=True)[0]
                        ipos = (ids[i] == i_id).nonzero(as_tuple=True)[0]
                        if len(upos): out[i, upos[0]] = out[i, upos[0]] + u_soft[i]
                        if len(ipos): out[i, ipos[0]] = out[i, ipos[0]] + i_soft[i]
                    return out
                return hook

            handle = embed_layer.register_forward_hook(
                make_hook(user_soft, item_soft,
                          user_token_id, item_token_id, input_ids))

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            handle.remove()  # 用完立即移除hook
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                nan_steps += 1
                optimizer.zero_grad()
                if nan_steps <= 3:
                    print(f"\n[warn] NaN at step {step}")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(adapter.parameters()) +
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            total_loss += loss.item()
            valid_steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / max(valid_steps, 1)
        print(f"Epoch {epoch+1} avg_loss={avg:.4f} "
              f"valid={valid_steps}/{len(loader)} nan={nan_steps}")

        ckpt = os.path.join(args.output_dir, f"epoch{epoch+1}")
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        torch.save(adapter.state_dict(), os.path.join(ckpt, 'adapter.pth'))
        print(f"Saved → {ckpt}")


# ── 4. 推理（纯文字，不需要软token）─────────────────────────
def inference(args):
    from peft import PeftModel
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # tokenizer 从checkpoint加载（含扩展的特殊token）
    tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    # 基础模型 + resize embedding
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map='auto')
    model.resize_token_embeddings(len(tokenizer))
    # 加载LoRA并合并
    model = PeftModel.from_pretrained(model, args.output_dir)
    model = model.merge_and_unload()
    model.eval()
    prompt = (
        "Given the business title, business profile, and user profile, "
        "please explain why the user would enjoy this business within 50 words. "
        "Business title: ZooTampa at Lowry Park. "
        "Business profile: Animal lovers and families with children. "
        "User profile: Enjoys outdoor activities and nature.\n"
    )
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    print(tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                           skip_special_tokens=True))


# ── 5. 入口 ────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',        default='train', choices=['train','inference'])
    p.add_argument('--model_path',  default='meta-llama/Llama-3.2-3B')
    p.add_argument('--raft_path',   default='raft_data/yelp/train.json')
    p.add_argument('--pyg_path',    default='data/yelp/data_trn.pt')
    p.add_argument('--gnn_emb_path', default='data/yelp/gnn_embeddings.pt')
    p.add_argument('--output_dir',  default='outputs/collab_llama_yelp')
    p.add_argument('--max_samples', type=int,   default=10000)
    p.add_argument('--batch_size',  type=int,   default=4)
    p.add_argument('--num_epochs',  type=int,   default=3)
    p.add_argument('--lr',          type=float, default=2e-4)
    p.add_argument('--max_length',  type=int,   default=1200)
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args) if args.mode == 'train' else inference(args)

# 训练: python train_collab.py --mode train --max_samples 10000 --batch_size 4 --num_epochs 3
# 推理: python train_collab.py --mode inference --output_dir outputs/collab_llama_yelp/epoch3
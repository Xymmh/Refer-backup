"""
从训练好的 HeteroRGCN 提取 user/item 节点embedding
输出格式与 data_trn.pt 兼容，可直接替换 node_emb 使用
"""
import torch
import sys
sys.path.append('path_retriever')  # 加载 model.py
from model import HeteroRGCN, HeteroLinkPredictionModel
import dgl

def extract_embeddings(
    graph_path  = 'PaGE-Link/datasets/yelp_trn.bin',
    model_path  = 'path_retriever/saved_models/yelp_model_trn.pth',
    save_path   = 'data/yelp/gnn_embeddings.pt',
    emb_dim     = 128,
    hidden_size = 128,
    out_size    = 128,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载图
    print("Loading graph...")
    graphs, _ = dgl.load_graphs(graph_path)
    g = graphs[0].to(device)
    print(f"Graph: {g}")

    # 重建模型结构
    print("Building model...")
    encoder = HeteroRGCN(g, emb_dim, hidden_size, out_size)
    model   = HeteroLinkPredictionModel(
        encoder, src_ntype='user', tgt_ntype='item', link_pred_op='dot')

    # 加载训练好的权重
    print("Loading checkpoint...")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # 前向传播提取embedding
    print("Extracting embeddings...")
    with torch.no_grad():
        h_dict = model.encode(g)

    user_emb = h_dict['user'].cpu()  # [num_users, 128]
    item_emb = h_dict['item'].cpu()  # [num_items, 128]

    print(f"user_emb shape: {user_emb.shape}")
    print(f"item_emb shape: {item_emb.shape}")
    print(f"user_emb mean: {user_emb.mean():.4f}, std: {user_emb.std():.4f}")
    print(f"item_emb mean: {item_emb.mean():.4f}, std: {item_emb.std():.4f}")

    # 保存
    torch.save({
        'user_emb': user_emb,
        'item_emb': item_emb,
    }, save_path)
    print(f"Saved → {save_path}")

    return user_emb, item_emb


if __name__ == '__main__':
    extract_embeddings()

# 运行：python extract_gnn_emb.py
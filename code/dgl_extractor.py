import dgl
import argparse
from tqdm import tqdm
from utils import load_pickle, load_data

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset to DGL format')
    parser.add_argument('--dataset', type=str, choices=['yelp', 'google', 'amazon'], required=True,
                        help='Dataset to convert (yelp, google, or amazon)')
    parser.add_argument('--split', type=str, choices=['trn', 'val', 'tst'], required=True,
                        help='Data split to convert (trn, val, or tst)')
    return parser.parse_args()

def create_dgl_graph(pyg_data, pkl_data):
    num_users = len(pyg_data.user_id_to_node)
    num_items = len(pyg_data.item_id_to_node)
    
    g = dgl.heterograph({
        ('user', 'have_bought', 'item'): ([], []),
        ('item', 'bought_by', 'user'): ([], []),
        ('user', 'likes', 'item'): ([], [])
    }, num_nodes_dict={'user': num_users, 'item': num_items})

    print(f"Number of rows in pkl_data: {len(pkl_data)}")

    # Create likes_edges set
    likes_edges = set()
    for _, row in tqdm(pkl_data.iterrows(), desc="Processing pkl_data", total=len(pkl_data)):
        if row['uid'] in pyg_data.user_id_to_node and row['iid'] in pyg_data.item_id_to_node:
            user_idx = pyg_data.user_id_to_node[row['uid']]
            item_idx = pyg_data.item_id_to_node[row['iid']] - num_users
            likes_edges.add((user_idx, item_idx))
        else:
            print(row['uid'], row['iid'])

    print(f"Number of likes edges: {len(likes_edges)}")

    buys_edges = set()
    
    # Process all edges from pyg_data
    for src, dst in tqdm(pyg_data.edge_index.t().tolist(), desc="Processing edges"):
        if src < num_users:  # src is a user node
            user_idx = src
            item_idx = dst - num_users
            buys_edges.add((user_idx, item_idx))

    # Remove likes edges from buys edges
    buys_edges -= likes_edges

    # Create final edge lists
    buys_src, buys_dst = zip(*buys_edges) if buys_edges else ([], [])
    likes_src, likes_dst = zip(*likes_edges) if likes_edges else ([], [])
    bought_by_src, bought_by_dst = buys_dst, buys_src

    # Add edges to the graph
    g.add_edges(buys_src, buys_dst, etype='have_bought')
    g.add_edges(bought_by_src, bought_by_dst, etype='bought_by')
    g.add_edges(likes_src, likes_dst, etype='likes')

    print(f"Processed likes edges: {len(likes_src)}")
    print(f"Processed buys edges: {len(buys_src)}")
    print(f"Processed bought_by edges: {len(bought_by_src)}")

    # Add node features
    g.nodes['user'].data['feat'] = pyg_data.x[:num_users]
    g.nodes['item'].data['feat'] = pyg_data.x[num_users:]

    return g

def main():
    args = parse_args()
    
    print(f"Processing {args.dataset} {args.split} data...")
    
    pyg_data = load_data(f'data/{args.dataset}/data_{args.split}.pt')
    pkl_data = load_pickle(f'data/{args.dataset}/{args.split}.pkl')
    
    dgl_graph = create_dgl_graph(pyg_data, pkl_data)
    
    # Save only the DGL graph
    dgl.save_graphs(f'./data/{args.dataset}/{args.split}_graph.bin', [dgl_graph])

    # also save a copy to pagelink
    dgl.save_graphs(f'./PaGE-Link/datasets/{args.dataset}_{args.split}.bin', [dgl_graph])
    
    print(f"Processed graph for {args.split} data")
    print(f"Graph saved to {f'./data/{args.dataset}/{args.split}_graph.bin'}")

    print(f"Graph statistics:")
    print(f"  Number of users: {dgl_graph.num_nodes('user')}")
    print(f"  Number of items: {dgl_graph.num_nodes('item')}")
    print(f"  Number of edges:")
    for etype in dgl_graph.etypes:
        print(f"    {etype}: {dgl_graph.num_edges(etype=etype)}")
    print(f"  Node feature dimensions:")
    print(f"    User: {dgl_graph.nodes['user'].data['feat'].shape[1]}")
    print(f"    Item: {dgl_graph.nodes['item'].data['feat'].shape[1]}")

if __name__ == "__main__":
    main()

# python dgl_extractor.py --split trn --dataset yelp
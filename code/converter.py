import torch
import pandas as pd
import json
from torch_geometric.data import Data
import argparse
import os
import tqdm
import pickle
from model import TextModel
from utils import clean_text

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset to PyG format')
    parser.add_argument('--dataset', type=str, choices=['yelp', 'google', 'amazon'], required=True,
                        help='Dataset to convert (yelp, google, or amazon)')
    parser.add_argument('--split', type=str, choices=['trn', 'val', 'tst'], required=True,
                        help='Data split to convert (trn, val, or tst)')
    parser.add_argument('--text_encoder', default='SentenceBert', type=str, choices=['Bert', 'Roberta', 'SentenceBert', 'SimCSE', 'e5', 't5'],
                        help='Text encoder to use (Bert, Roberta, SentenceBert, SimCSE, e5, or t5)')
    return parser.parse_args()

def main():
    args = parse_args()

    df = pd.read_csv(f'data/{args.dataset}/total_{args.split}.csv')

    unique_users = df['user'].unique()
    unique_items = df['item'].unique()

    # Process all splits for titles
    splits = ['trn', 'tst']
    all_pkl_data = {}
    for split in splits:
        with open(f'data/{args.dataset}/{split}.pkl', 'rb') as f:
            all_pkl_data[split] = pickle.load(f)

    user_id_to_node = {id: i for i, id in enumerate(unique_users)}
    item_id_to_node = {id: i + len(user_id_to_node) for i, id in enumerate(unique_items)}

    # Load user profiles
    with open(f'data/{args.dataset}/user_profile.json', 'r') as f:
        user_profiles = [json.loads(line) for line in f]
    
    # Create user_texts dictionary
    user_texts = {}
    for profile in user_profiles:
        user_id = profile['uid']  # Assuming each profile has a 'user_id' field
        if args.dataset == 'amazon':
            text = clean_text(profile['completion'], args.dataset)
        else:
            text = clean_text(profile['user summary'], args.dataset)
        user_texts[user_id] = text

    # Load item profiles
    with open(f'data/{args.dataset}/item_profile.json', 'r') as f:
        item_profiles = [json.loads(line) for line in f]
    
    # Create item_texts dictionary
    item_texts = {}
    for profile in item_profiles:
        item_id = profile['iid']  # Assuming each profile has an 'item_id' field
        if args.dataset == 'amazon':
            text = clean_text(profile['completion'], args.dataset)
        else:
            text = clean_text(profile['business summary'], args.dataset)
        item_texts[item_id] = text

    # Create ordered lists of texts based on user_id_to_node and item_id_to_node
    ordered_user_texts = [user_texts[uid] for uid in user_id_to_node.keys()]
    ordered_item_texts = [item_texts[iid] for iid in item_id_to_node.keys()]

    item_titles = []
    items_with_title = 0
    items_without_title = 0

    for item_id in unique_items:
        title = ""
        for split, pkl_data in all_pkl_data.items():
            matching_rows = pkl_data[pkl_data['iid'] == item_id]
            if not matching_rows.empty:
                title = matching_rows['title'].iloc[0]
                if title:
                    break  # Stop searching if we found a title
        
        if title:
            items_with_title += 1
        else:
            items_without_title += 1
        
        item_titles.append(title)

    print(f"Items with title: {items_with_title}")
    print(f"Items without title: {items_without_title}")

    # Create bidirectional edge_index
    user_nodes = [user_id_to_node[u] for u in df['user']]
    item_nodes = [item_id_to_node[i] for i in df['item']]
    edge_index = torch.tensor([
        user_nodes + item_nodes, 
        item_nodes + user_nodes
    ], dtype=torch.long)

    # item_combined_texts = [f"Title: {title}\nSummary: {text}" for title, text in zip(item_titles, item_texts)]
    
    all_texts = ordered_user_texts + ordered_item_texts

    text_model = TextModel(args.text_encoder)

    text_model = text_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    text_features = []

    batch_size = 128
    for i in tqdm.tqdm(range(0, len(all_texts), batch_size), desc="Processing texts"):
        batch = all_texts[i:i+batch_size]
        with torch.no_grad():
            batch_features = text_model(batch).cpu()
        text_features.append(batch_features)

    text_features = torch.cat(text_features, dim=0)

    data = Data(
        edge_index=edge_index,
        num_nodes=len(user_id_to_node) + len(item_id_to_node),
        raw_texts=all_texts,
        x=text_features,
        user_id_to_node=user_id_to_node,
        item_id_to_node=item_id_to_node,
        item_titles=item_titles
    )

    save_path = f'data/{args.dataset}/data_{args.split}.pt'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)

    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    main()

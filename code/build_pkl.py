import json
import csv
import pickle
import pandas as pd
import os

DATASET = "yelp"
BASE_DIR = "/home/wangqialun/G-Refer"
DATA_DIR = f"{BASE_DIR}/data/{DATASET}"
RAFT_DIR = f"{BASE_DIR}/raft_data/{DATASET}"

def load_item_titles():
    item_titles = {}
    with open(f"{DATA_DIR}/item_profile.json", "r") as f:
        for line in f:
            obj = json.loads(line)
            iid = obj["iid"]
            # title 字段直接取 name，不存整个 JSON
            if "name" in obj:
                title = obj["name"]
            elif "title" in obj:
                title = obj["title"]
            else:
                title = ""
            item_titles[iid] = title
    return item_titles

def build_split(split, item_titles):
    csv_file = f"{DATA_DIR}/total_{split}.csv"
    raft_file = f"{RAFT_DIR}/{'train' if split == 'trn' else 'test'}.json"

    # 加载 raft_data 正样本对
    raft_pairs = set()
    if os.path.exists(raft_file):
        with open(raft_file) as f:
            for line in f:
                d = json.loads(line)
                raft_pairs.add((d['uid'], d['iid']))
        print(f"[{split}] raft pairs loaded: {len(raft_pairs)}")
    else:
        print(f"[{split}] raft file not found, all interactions treated as likes")

    # 只保留在 raft 里的交互作为 likes
    rows = []
    with open(csv_file) as f:
        for row in csv.DictReader(f):
            uid, iid = int(row['user']), int(row['item'])
            if not raft_pairs or (uid, iid) in raft_pairs:
                rows.append({
                    'uid': uid,
                    'iid': iid,
                    'title': item_titles.get(iid, "")
                })

    pkl_df = pd.DataFrame(rows)
    save_path = f"{DATA_DIR}/{split}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(pkl_df, f)
    print(f"[{split}] saved {save_path}, size={len(pkl_df)}")

def main():
    print("Loading item titles...")
    item_titles = load_item_titles()
    print(f"Loaded {len(item_titles)} item titles")

    for split in ["trn", "tst"]:
        build_split(split, item_titles)

if __name__ == "__main__":
    main()

# python build_pkl.py
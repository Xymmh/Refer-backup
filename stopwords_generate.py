import json
import re
from collections import Counter
from typing import List, Set

def clean_words(text: str) -> List[str]:
    """与原代码中 _clean_words 完全一致：小写、只保留字母词（≥3字符）"""
    return [w for w in re.findall(r'[a-z]+', text.lower()) if len(w) >= 3]


def extract_entities_from_prompt(prompt: str) -> Set[str]:
    """提取所有可能出现在路径和 profile 中的实体词（与你代码中的逻辑一致）"""
    entities = set()

    # 1. Item (Profile: ...) 中的内容
    item_profiles = re.findall(r'Item\s*\(Profile:\s*(.*?)\)', prompt, re.DOTALL)
    for prof in item_profiles:
        entities.update(clean_words(prof))

    # 2. Business profile 字段
    start = prompt.find("Business profile:")
    if start != -1:
        start += len("Business profile:")
        end = prompt.find(" User profile:", start)
        if end == -1:
            end = prompt.find("\n###", start)
        segment = prompt[start:end].strip() if end != -1 else prompt[start:].strip()
        entities.update(clean_words(segment))

    # 3. 路径中的 User (Profile: ...)（辅助提取高频结构词）
    user_profiles = re.findall(r'User\s*\(Profile:\s*(.*?)\)', prompt, re.DOTALL)
    for prof in user_profiles:
        entities.update(clean_words(prof))

    return entities


def generate_stopwords(input_json_path: str, min_freq: int = 50, top_n: int = 300):
    """
    扫描整个 train.json，统计所有 prompt 中出现的实体词频率，
    输出一个适合放在 _clean_words 里的 STOPWORDS 集合。
    """
    word_counter = Counter()

    print(f"Reading {input_json_path} ...")
    with open(input_json_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                ents = extract_entities_from_prompt(prompt)
                word_counter.update(ents)

                if line_num % 10000 == 0:
                    print(f"Processed {line_num} samples...")

            except json.JSONDecodeError:
                print(f"Warning: JSON decode error at line {line_num}")
                continue

    # 所有出现过的词按频率降序
    most_common = word_counter.most_common(top_n)

    print(f"\nTotal unique words extracted: {len(word_counter)}")
    print(f"Top 50 most frequent words:")
    for w, cnt in most_common[:50]:
        print(f"  {w:20} : {cnt}")

    # 自动筛选高频结构词（出现次数 >= min_freq）
    auto_stopwords = {w for w, cnt in most_common if cnt >= min_freq}

    # 保留你原来手动添加的核心停用词（防止被过滤掉）
    manual_core = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'has', 'have',
        'had', 'it', 'its', 'this', 'that', 'i', 'you', 'he', 'she', 'we',
        'they', 'my', 'your', 'his', 'her', 'our', 'their', 'not', 'no', 'as',
        'by', 'from', 'up', 'out', 'so', 'if', 'do', 'did', 'will', 'would',
        'can', 'could', 'may', 'might', 'than', 'then', 'also', 'just', 'more',
        'very', 'all', 'any', 'each', 'both', 'about', 'into', 'through',
        'during', 'what', 'which', 'who', 'whom', 'how', 'when', 'where', 'why',
        # 原代码中已有的高频结构词
        'user', 'item', 'users', 'items', 'profile', 'likely', 'enjoy',
        'business', 'businesses', 'offer', 'offers', 'appreciate', 'looking',
        'experience', 'options', 'service', 'food', 'atmosphere', 'dining',
        'those', 'well', 'find', 'good', 'great', 'make', 'get', 'give',
        'want', 'need', 'place', 'spot', 'setting', 'located', 'area',
        'variety', 'wide', 'range', 'quality', 'high', 'low', 'best', 'top',
        # 额外常见结构词（从样本中高频出现）
        'would', 'that', 'with', 'for', 'and', 'they', 'who', 'from',
        'diverse', 'casual', 'friendly', 'attentive', 'unique', 'fresh',
        'flavorful', 'authentic', 'reasonable', 'generous', 'cozy', 'vibrant',
        'lively', 'creative', 'traditional', 'family', 'families', 'fans',
        'lovers', 'individuals', 'experiences', 'atmospheres', 'menus',
        'dishes', 'cuisine', 'cuisines', 'drinks', 'cocktails', 'beers',
        'prices', 'service', 'staff', 'location', 'atmosphere'
    }

    final_stopwords = manual_core | auto_stopwords

    # 输出格式（可直接复制到你的 train_grpo.py）
    print("\n" + "="*80)
    print("以下是推荐的 STOPWORDS（已按字母排序，便于维护）：")
    print("="*80)

    sorted_stop = sorted(final_stopwords)
    print("STOPWORDS = {")
    for i, w in enumerate(sorted_stop):
        if i % 10 == 0 and i > 0:
            print()
        print(f"    '{w}',", end='')
    print("\n}")

    # 保存到文件（可选）
    with open("stopwords_generated.py", "w", encoding="utf-8") as f:
        f.write("STOPWORDS = {\n")
        for w in sorted_stop:
            f.write(f"    '{w}',\n")
        f.write("}\n")
    print(f"\n已保存到 stopwords_generated.py（共 {len(final_stopwords)} 个词）")

    return final_stopwords


if __name__ == "__main__":
    # 使用方法
    # python generate_stopwords.py
    generate_stopwords("train.json", min_freq=50, top_n=250)

# metrics.py - 与原作者数据格式对齐，使用 Qwen Instruct 打分
# gt 和 gen 均不处理 ### 前缀，直接原样评测

import evaluate
import numpy as np
from bart_score import BARTScorer
import argparse
import json
from bleurt import score
import tensorflow as tf
from tqdm import tqdm
import torch
import re
import os
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",    type=str,   default="yelp")
parser.add_argument("--ratio",      type=float, default=0.1)
parser.add_argument("--input_file", type=str,
                    default="/home/wangqialun/G-Refer/outputs/grpo_results_v5_prompt.jsonl")
parser.add_argument("--qwen_model", type=str,   default="Qwen/Qwen3-4B-Instruct-2507")
args = parser.parse_args()

_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_dir, "system_prompt.txt"), "r") as f:
    system_prompt = f.read().strip()

_qwen_model     = None
_qwen_tokenizer = None


def get_qwen_model():
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is None:
        print(f"Loading Qwen model for scoring: {args.qwen_model}")
        _qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_model, trust_remote_code=True)
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            args.qwen_model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _qwen_model.eval()
        print("Qwen scoring model loaded.")
    return _qwen_model, _qwen_tokenizer


def release_qwen_model():
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is not None:
        print("Releasing Qwen model from GPU memory...")
        del _qwen_model
        del _qwen_tokenizer
        _qwen_model     = None
        _qwen_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU memory released.")


def get_qwen_response(prediction, reference):
    model, tokenizer = get_qwen_model()
    prompt = (
        f"Reference: {reference}\n"
        f"Prediction: {prediction}\n\n"
        f"Score the prediction against the reference on a scale from 0 to 100. "
        f"Reply with a single integer only, no explanation."
    )
    messages = [
        {"role": "system", "content": "You are a strict evaluator. Reply with a single integer score from 0 to 100. No other text."},
        {"role": "user",   "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    response  = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    nums = re.findall(r"\d+\.?\d*", response)
    if nums:
        return min(max(float(nums[0]), 0.0), 100.0)
    else:
        print(f"Warning: could not parse score from: '{response}', defaulting to 50.0")
        return 50.0


def get_qwen_score(predictions, references):
    results = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="Computing Qwen scores"):
        results.append(get_qwen_response(pred, ref))
    return np.mean(results), np.std(results)


class MetricScore:
    def __init__(self):
        print(f"evaluating dataset: {args.dataset}")
        print(f"loading from: {args.input_file}")
        self.data     = []   # ground truth（chosen，原样）
        self.ref_data = []   # generated（output_str，原样）

        with open(args.input_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                gt  = sample["source_data"]["chosen"]
                gen = sample["output_str"]
                self.data.append(gt)
                self.ref_data.append(gen)

        print(f"Loaded {len(self.data)} samples.")
        if self.data:
            print(f"Ground truth example: {self.data[0][:120]}")
            print(f"Generated example:    {self.ref_data[0][:120]}")

    def get_score(self):
        scores = {}

        (bert_precision, bert_recall, bert_f1,
         bert_precision_std, bert_recall_std, bert_f1_std) = BERT_score(self.data, self.ref_data)

        qwen_score, qwen_std = get_qwen_score(self.data, self.ref_data)

        release_qwen_model()

        bart_score, bart_score_std     = BART_score(self.data, self.ref_data)
        bleurt_score, bleurt_score_std = BLEURT_score(self.data, self.ref_data)

        tokens_predict = [s.split() for s in self.ref_data]
        usr, _ = unique_sentence_percent(tokens_predict)

        scores.update({
            "qwen_score": qwen_score, "qwen_std": qwen_std,
            "bert_precision": bert_precision, "bert_precision_std": bert_precision_std,
            "bert_recall": bert_recall, "bert_recall_std": bert_recall_std,
            "bert_f1": bert_f1, "bert_f1_std": bert_f1_std,
            "bart_score": bart_score, "bart_score_std": bart_score_std,
            "bleurt_score": bleurt_score, "bleurt_score_std": bleurt_score_std,
            "usr": usr,
        })
        return scores

    def print_score(self):
        s = self.get_score()
        print(f"\ndataset: {args.dataset}  ratio: {args.ratio}")
        print("Explanability Evaluation Metrics:")
        print(f"qwen_score:     {s['qwen_score']:.4f}  (std: {s['qwen_std']:.4f})")
        print(f"bert_precision: {s['bert_precision']:.4f}  (std: {s['bert_precision_std']:.4f})")
        print(f"bert_recall:    {s['bert_recall']:.4f}  (std: {s['bert_recall_std']:.4f})")
        print(f"bert_f1:        {s['bert_f1']:.4f}  (std: {s['bert_f1_std']:.4f})")
        print(f"bart_score:     {s['bart_score']:.4f}  (std: {s['bart_score_std']:.4f})")
        print(f"bleurt_score:   {s['bleurt_score']:.4f}  (std: {s['bleurt_score_std']:.4f})")
        print(f"usr:            {s['usr']:.4f}")


def two_seq_same(sa, sb):
    return len(sa) == len(sb) and all(a == b for a, b in zip(sa, sb))


def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        if not any(two_seq_same(seq, u) for u in unique_seq):
            unique_seq.append(seq)
    return len(unique_seq) / len(sequence_batch), len(unique_seq)


def BERT_score(predictions, references):
    bertscore = evaluate.load("bertscore")
    results   = bertscore.compute(predictions=predictions, references=references, lang="en")
    p, r, f   = results["precision"], results["recall"], results["f1"]
    return np.mean(p), np.mean(r), np.mean(f), np.std(p), np.std(r), np.std(f)


def BART_score(predictions, references):
    bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-large-cnn")
    scores = []
    for i in tqdm(range(0, len(predictions), 4), desc="Computing BART scores"):
        scores.extend(bart_scorer.score(predictions[i:i+4], references[i:i+4], batch_size=4))
    return np.mean(scores), np.std(scores)


def BLEURT_score(predictions, references):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    tf.config.set_visible_devices([], "GPU")
    bleurt_ops = score.create_bleurt_ops()
    scores = []
    for ref, pred in tqdm(zip(references, predictions), total=len(references), desc="Computing BLEURT scores"):
        out = bleurt_ops(references=tf.constant([ref]), candidates=tf.constant([pred]))
        scores.append(out["predictions"][0])
    return np.mean(scores), np.std(scores)


if __name__ == "__main__":
    metric = MetricScore()
    metric.print_score()
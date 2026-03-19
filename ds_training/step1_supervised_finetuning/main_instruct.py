# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass, field
from typing import Dict, List
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier"})
    use_4bit: bool = field(default=True, metadata={"help": "Use 4bit quantization"})
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_compute_dtype: str = field(default="bfloat16")


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Path to the training JSON file"})


@dataclass
class LoraArguments:
    lora_r: int = field(default=8, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: str = field(default="q_proj,v_proj,k_proj,o_proj",
                                metadata={"help": "LoRA target modules (comma separated)"})


MAX_LENGTH = 2048


def build_formatting_func(tokenizer):
    """
    数据格式与原作者保持一致：
      训练输入 = prompt + chosen（chosen 保留 "### " 前缀）+ EOS
      例：
        prompt:  "...### Explanation:"
        chosen:  "### The user would enjoy..."
        全文：   "...### Explanation:\n### The user would enjoy...<eos>"

    Label masking：
      prompt 部分（到 "### Explanation:" 末尾）设为 -100
      chosen 部分（"### The user would..."）正常计入 loss
    """

    def formatting_func(examples: Dict[str, List]):
        prompts = examples["prompt"]
        chosens = examples["chosen"]

        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for prompt, chosen in zip(prompts, chosens):
            # ? 保留 chosen 原始格式（含 "### " 前缀），与原作者一致
            # 原始数据：prompt 末尾 "### Explanation:"，chosen 开头 "### The user..."
            full_text = prompt + "\n" + chosen + tokenizer.eos_token

            # Tokenize 完整文本
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors=None,
                add_special_tokens=True,
            )
            input_ids      = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            # 单独 tokenize prompt，获取长度用于 label masking 边界
            prompt_tokenized = tokenizer(
                prompt,
                truncation=False,
                add_special_tokens=True,
                return_tensors=None,
            )
            prompt_len = len(prompt_tokenized["input_ids"])

            # 构建 labels：
            #   [0, prompt_len)        -> -100（prompt 部分，不学习）
            #   [prompt_len, real_end) -> input_ids[i]（chosen 部分，学习）
            #   padding 位置           -> -100
            labels = []
            for i, (token_id, attn) in enumerate(zip(input_ids, attention_mask)):
                if attn == 0:
                    labels.append(-100)
                elif i < prompt_len:
                    labels.append(-100)
                else:
                    labels.append(token_id)

            # 安全检查：chosen 被完全截断则该样本无效
            if all(l == -100 for l in labels):
                labels = [-100] * len(input_ids)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids":      all_input_ids,
            "attention_mask": all_attention_masks,
            "labels":         all_labels,
        }

    return formatting_func


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # ------------------------------------------------------------------ #
    # Tokenizer
    # ------------------------------------------------------------------ #
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 训练时 right padding

    # ------------------------------------------------------------------ #
    # 量化配置
    # ------------------------------------------------------------------ #
    quant_config = None
    if model_args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, model_args.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

    # ------------------------------------------------------------------ #
    # 模型
    # ------------------------------------------------------------------ #
    print(f"Loading model: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if not quant_config else None,
        trust_remote_code=True,
    )

    # ------------------------------------------------------------------ #
    # LoRA
    # ------------------------------------------------------------------ #
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=lora_args.target_modules.split(","),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------ #
    # 数据集
    # ------------------------------------------------------------------ #
    print(f"Loading dataset from: {data_args.train_file}")
    dataset = load_dataset("json", data_files=data_args.train_file, split="train")
    dataset = dataset.select(range(min(10000, len(dataset))))
    print(f"Dataset size: {len(dataset)}")

    formatting_func = build_formatting_func(tokenizer)

    print("Formatting and tokenizing dataset...")
    dataset = dataset.map(
        formatting_func,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4,
        desc="Formatting and tokenizing",
    )

    # 过滤无效样本（chosen 被完全截断）
    dataset = dataset.filter(lambda x: any(l != -100 for l in x["labels"]))
    print(f"Dataset size after filtering: {len(dataset)}")

    # 验证第一条样本的 label 分布
    first_labels = dataset[0]["labels"]
    masked  = sum(1 for l in first_labels if l == -100)
    learned = sum(1 for l in first_labels if l != -100)
    print(f"First sample — masked: {masked}, learned: {learned}")

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ------------------------------------------------------------------ #
    # Trainer
    # ------------------------------------------------------------------ #
    import trl
    trl_version = tuple(int(x) for x in trl.__version__.split(".")[:2])
    sft_kwargs = dict(model=model, args=training_args, train_dataset=dataset)
    if trl_version < (0, 8):
        sft_kwargs["max_seq_length"] = MAX_LENGTH

    trainer = SFTTrainer(**sft_kwargs)

    print("开始训练...")
    trainer.train()

    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"训练完成，模型保存到: {training_args.output_dir}")


if __name__ == "__main__":
    main()
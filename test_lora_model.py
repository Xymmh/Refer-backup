# chat_lora.py - 交互式测试 base model + LoRA adapter
# 放在 G-Refer/ 目录下运行：python chat_lora.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

# ------------------------------------------------------------------ #
# 配置：按需修改
# ------------------------------------------------------------------ #
BASE_MODEL_PATH = "meta-llama/Llama-3.2-3B"
LORA_PATH       = "outputs/collab_llama_yelp_gnn_v2/epoch2"   # 相对于 G-Refer/ 的路径
MAX_NEW_TOKENS  = 128
USE_4BIT        = False   # 显存不够时改为 True


# ------------------------------------------------------------------ #
# 加载模型
# ------------------------------------------------------------------ #
def load_model(base_path, lora_path, use_4bit):
    print(f"Loading tokenizer from: {lora_path}")
    tokenizer = AutoTokenizer.from_pretrained(lora_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {base_path}")
    if use_4bit:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    print(f"Loading LoRA adapter from: {lora_path}")
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    model.eval()
    print("Model ready. Type your message below.\n")
    return model, tokenizer


# ------------------------------------------------------------------ #
# 推理
# ------------------------------------------------------------------ #
@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        generation_config=GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )

    input_len = inputs.input_ids.shape[1]
    gen_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    return gen_text


# ------------------------------------------------------------------ #
# 交互主循环
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    model, tokenizer = load_model(BASE_MODEL_PATH, LORA_PATH, USE_4BIT)

    print("=" * 70)
    print("交互模式已启动，输入 'quit' 或 'exit' 退出，输入 'clear' 清屏")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("退出。")
            break
        if user_input.lower() == "clear":
            import os
            os.system("clear" if os.name == "posix" else "cls")
            continue

        response = generate(model, tokenizer, user_input, MAX_NEW_TOKENS)
        print(f"\nModel: {response}")
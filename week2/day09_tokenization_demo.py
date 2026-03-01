"""
Day 9 可选实践：Tokenization 与生成

对应 day09_llm_basics.md 第二部分，练习：
- 分词、input_ids、attention_mask 的获取与含义
- encode / decode 还原
- （可选）若提供模型路径，观察 model.generate() 参数对输出的影响
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. 仅用 Tokenizer：不加载模型，适合无 GPU 或只想练分词的场景
# ---------------------------------------------------------------------------

def demo_tokenizer_only(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """只加载 tokenizer，演示 encode/decode、input_ids、attention_mask。"""
    from transformers import AutoTokenizer

    print("加载 tokenizer（不加载模型）...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 单句编码
    text = "你好，请用一句话介绍注意力机制。"
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    print("\n--- 单句 ---")
    print("原文:", text)
    print("input_ids shape:", input_ids.shape, "  (batch_size, seq_len)")
    print("input_ids:", input_ids)
    print("attention_mask:", attention_mask)
    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    print("decode 还原:", decoded)

    # 多句 + padding：长度不一时会自动 padding，attention_mask 标记有效位置
    texts = ["短句", "这是一个稍长一点的句子，用来演示 padding。"]
    enc_batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    print("\n--- 多句（padding）---")
    print("input_ids shape:", enc_batch["input_ids"].shape)
    print("attention_mask:\n", enc_batch["attention_mask"])
    for i, t in enumerate(texts):
        print(f"  句 {i+1} 还原: {tokenizer.decode(enc_batch['input_ids'][i], skip_special_tokens=True)}")


# ---------------------------------------------------------------------------
# 2. 可选：Tokenizer + 模型生成（需本地有小模型时取消注释并设置路径）
# ---------------------------------------------------------------------------

def demo_generation(model_path: str | None = None):
    """
    若提供 model_path（例如 day08 使用的本地 Qwen2.5-1.5B 路径），
    演示同一 prompt 下 do_sample / temperature / max_new_tokens 对生成结果的影响。
    """
    if not model_path:
        print("未设置 model_path，跳过生成演示。")
        return

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("\n加载模型与 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()

    prompt = "一句话解释什么是自回归生成："
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    print("\n--- 生成参数对比 ---")
    print("prompt:", prompt)

    # 贪婪解码
    with torch.no_grad():
        out_greedy = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text_greedy = tokenizer.decode(out_greedy[0][input_ids.shape[1]:], skip_special_tokens=True)
    print("\n1. do_sample=False（贪婪）:", text_greedy[:80], "..." if len(text_greedy) > 80 else "")

    # 采样，不同 temperature
    with torch.no_grad():
        out_sample = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text_sample = tokenizer.decode(out_sample[0][input_ids.shape[1]:], skip_special_tokens=True)
    print("2. do_sample=True, temperature=0.8:", text_sample[:80], "..." if len(text_sample) > 80 else "")


# ---------------------------------------------------------------------------
# 主入口：默认只跑 tokenizer；有本地模型时可传入路径跑生成
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model_name = "/Users/tong/Documents/models/Qwen/Qwen2.5-1.5B-Instruct"
    demo_tokenizer_only(model_name)
    demo_generation(model_name)

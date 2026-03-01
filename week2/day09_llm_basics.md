# Day 9 详细学习指南：大语言模型基础

> 目标：理解 LLM 的训练阶段（PT、SFT、RLHF）和推理相关概念（分词、input_ids、自回归生成）

---

## 学习时间分配建议

| 时段 | 内容 | 时长 |
|------|------|------|
| 0:00–0:55 | 第一部分：LLM 训练流程（多阶段→PT→SFT→RLHF→与 verl 衔接） | 55 分钟 |
| 0:55–1:00 | 休息 | 5 分钟 |
| 1:00–1:55 | 第二部分：Tokenization 与生成（分词→input_ids/attention_mask→自回归生成→代码过一遍→verl 衔接） | 55 分钟 |
| 1:55–2:00 | 自测题与检查清单 | 5 分钟 |

**预计总时长：2 小时**

---

## 一、第一部分：LLM 训练流程（约 1 小时）

### 1.1 必学知识点清单

| 知识点 | 说明 | 与 verl / LLM 的关系 |
|--------|------|----------------------|
| 预训练（PT） | 海量文本、下一个 token 预测（next-token prediction），得到「通用语言能力」 | verl 不负责 PT，通常加载已 PT 的 base 模型 |
| 监督微调（SFT） | 指令-回答对、在 PT 模型上做有监督训练，得到「听从指令」的能力 | verl 的 PPO/GRPO 等常以 SFT 模型为起点 |
| RLHF / 后训练 | 偏好数据 + 奖励模型 + 强化学习（如 PPO），对齐人类偏好 | verl 的核心场景，做的是后训练阶段 |

### 1.2 建议学习顺序

1. **为什么需要多阶段**（10 分钟）  
   - PT 只学「下一个词」，不会按指令回答；SFT 教指令遵循；RLHF 对齐偏好、减少有害输出  
   - 练习：用一句话分别说明 PT、SFT、RLHF 要解决什么问题

2. **预训练 PT**（15 分钟）  
   - 数据：网页、书籍、代码等海量文本  
   - 目标：next-token prediction，即给定前文预测下一个 token，loss 通常为交叉熵  
   - 规模与算力：参数量大、数据量大，需要大规模算力  
   - 练习：用一句话说明 PT 的输入和输出（输入：token 序列；输出：下一个 token 的预测分布）

3. **监督微调 SFT**（15 分钟）  
   - 数据格式：指令-回答对（instruction-response），如「请写一首诗」→「春眠不觉晓…」  
   - Loss：通常只在「回答」部分计算交叉熵，不计算指令部分的 loss  
   - 与 PT 的区别：PT 是无监督/自监督，SFT 是有监督的指令数据  
   - 练习：SFT 解决什么问题？（让模型学会按人类指令回答问题、遵循格式）

4. **RLHF 与后训练**（15 分钟）  
   - 三阶段概览：SFT → 训练 Reward Model（偏好数据）→ 用 RL（如 PPO）优化策略  
   - RLHF 解决什么问题：让模型输出更符合人类偏好（更有用、无害、诚实），而不仅是「模仿」SFT 数据  
   - 练习：能说明 SFT 和 RLHF 分别解决什么问题（见自测题）

5. **与 Day08 / verl 衔接**（5 分钟）  
   - Transformer Decoder 是 LLM 的骨架，PT/SFT/RLHF 是不同训练目标，模型结构相同  
   - verl 做的是 RLHF 阶段的训练（PPO、GRPO 等），不做 PT，常用 SFT 模型作为初始策略  
   - 练习：verl 主要对应三阶段中的哪一阶段？

### 1.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 文章 | 大模型训练流程 / LLM 三阶段训练 | 知乎、博客搜索「大模型训练流程」 | 建立 PT → SFT → RLHF 的整体图景 |
| 论文 | InstructGPT | https://arxiv.org/abs/2203.02155 | 摘要与 Figure 1 三阶段图 |
| 文档 | verl 文档 | verl 官方文档 | 了解 base model / SFT model 在配置中的角色 |

### 1.4 自测与检查

**自测题**  
1. PT 的训练目标是什么？用一句话说明。  
2. SFT 和 RLHF 分别解决什么问题？各用一句话说明。  
3. RLHF 通常包含哪三阶段？（可简写为 SFT → ? → ?）

<details>
<summary>点击展开参考答案</summary>

1. PT 的训练目标是「下一个 token 预测」：给定前文 token 序列，预测下一个 token，用交叉熵等 loss 在海量文本上训练，得到通用语言能力。  
2. SFT 解决「听从指令」：让模型学会按人类指令回答问题、遵循格式。RLHF 解决「对齐人类偏好」：让输出更有用、无害、诚实，而不仅是模仿 SFT 数据。  
3. 三阶段：SFT（监督微调）→ Reward Model 训练（用偏好数据训练奖励模型）→ RL（用 PPO 等强化学习优化策略）。
</details>

**检查清单**  
- [ ] 能说出 PT、SFT、RLHF 各自的目标和典型数据  
- [ ] 能说明 SFT 和 RLHF 分别解决什么问题  

---

## 二、第二部分：Tokenization 与生成（约 1 小时）

### 2.1 必学知识点清单

| 知识点 | 说明 | 与 verl / LLM 的关系 |
|--------|------|----------------------|
| 分词（Tokenization） | 文本 → token 序列 → token ids；词表（vocab）、BPE/WordPiece 等子词算法 | 数据与推理都要先 tokenize，verl 中 prompt 和 response 均为 token 序列 |
| input_ids | 模型输入，形状通常 (batch, seq_len)，每个元素是词表中的 id | 模型只接受数字 id，不能直接吃文本 |
| attention_mask | 标记有效位置（1）与 padding（0），避免 attention 算到 padding | batch 内序列长度不一致时必需 |
| 自回归生成 | 每次预测下一个 token，直到 EOS 或达到 max_new_tokens | 与 Day08 Decoder 因果掩码对应；verl rollout 时也是自回归生成 |

### 2.2 建议学习顺序

1. **为什么需要分词**（10 分钟）  
   - 模型处理的是离散 id 序列，不是字符或单词；词表大小通常几万，子词（BPE 等）平衡词表大小与序列长度  
   - 练习：说出「中文一句话」和「英文一句话」经过 BPE 分词后，token 数量大概谁多谁少（与语言和词表有关，不必精确，理解「子词」即可）

2. **input_ids 与 attention_mask**（15 分钟）  
   - `tokenizer(text, return_tensors="pt")` 返回 `input_ids`、`attention_mask` 等  
   - padding：短句补 0 到同一长度；attention_mask 中 padding 位置为 0  
   - truncation：过长则截断，避免超过模型最大长度  
   - 练习：给定一句话，batch_size=1，序列长度 10，说出 input_ids 的 shape 含义（(1, 10)，即 1 条样本、10 个 token id）

3. **自回归生成**（20 分钟）  
   - 流程：输入 prompt → 预测下一个 token → 拼到序列末尾 → 再预测下一个，直到 EOS 或 max_new_tokens  
   - 常用参数：temperature（控制随机性）、top_p、max_new_tokens、do_sample（是否采样）  
   - 与 Day08 因果掩码：生成时位置 i 只能看到 0..i，不能看到未来，否则会「作弊」  
   - 练习：自回归生成时为什么必须用 causal mask？（因为不能看到未来 token，否则就不是「逐词生成」）

4. **在代码中过一遍**（10 分钟）  
   - 结合 [day08_llm_inference_demo.ipynb](day08_llm_inference_demo.ipynb) 或 HuggingFace 文档  
   - 看 `tokenizer(text, return_tensors="pt")` 的返回；看 `model.generate()` 的输入（input_ids 等）和输出（生成的 token ids）  
   - 练习：不加载模型，仅用 tokenizer 对一句话 encode 再 decode，确认能还原

5. **与 verl 的衔接**（5 分钟）  
   - verl 里 rollout：把 prompt tokenize 后送入策略模型，自回归生成 response；reward 通常基于生成的文本或 token 序列计算  
   - 练习：说出「从用户输入到 reward」在 verl 中的大致数据流（prompt → tokenize → generate → response 文本 → reward）

### 2.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 文档 | HuggingFace Preprocessing | https://huggingface.co/docs/transformers/preprocessing | tokenizer 的 encode/decode、padding、truncation |
| 文档 | HuggingFace Text generation | https://huggingface.co/docs/transformers/main_classes/text_generation | `generate()` 参数说明 |
| 实践 | day08_llm_inference_demo | [day08_llm_inference_demo.ipynb](day08_llm_inference_demo.ipynb) | 已有 tokenizer + generate 示例，可在此基础上观察 input_ids/attention_mask |

### 2.4 自测与检查

**自测题**  
1. 解释 input_ids 和 attention_mask 的含义，以及 attention_mask 在 attention 计算中的作用。  
2. 自回归生成时为什么必须使用因果掩码（causal mask）？

<details>
<summary>点击展开参考答案</summary>

1. input_ids：将文本分词后得到的 token 在词表中的 id 序列，形状 (batch, seq_len)，是模型的直接输入。attention_mask：与 input_ids 同形状，1 表示有效位置、0 表示 padding；在 attention 计算中会把 mask 为 0 的位置置为 -inf，使 softmax 后权重为 0，避免模型「看到」padding。  
2. 自回归生成是「逐词生成」：当前只能根据已生成的 token 预测下一个，不能使用未来 token。若不用因果掩码，模型会看到后续位置，相当于作弊，且推理时本来就没有未来 token，所以必须用 causal mask 保证位置 i 只依赖 0..i。
</details>

**检查清单**  
- [ ] 能解释 input_ids、attention_mask 及在模型中的作用  
- [ ] 能解释自回归生成的基本过程及与因果注意力的关系  

### 2.5 可选实践：Tokenization 与生成

仅用 tokenizer 即可练习分词与编解码（无需加载模型）；若本机有小型 LLM（如 day08 使用的 Qwen2.5-1.5B），可加上 `model.generate()` 观察参数对输出的影响。

**方案 A：仅 Tokenizer（不加载模型）**

```python
from transformers import AutoTokenizer

# 使用与 day08 一致的模型名，只加载词表与分词器，无需下载完整模型
model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 或本地路径
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 编码与解码
text = "你好，请用一句话介绍注意力机制。"
enc = tokenizer(text, return_tensors="pt")
input_ids = enc["input_ids"]
attention_mask = enc["attention_mask"]
print("input_ids shape:", input_ids.shape)
print("input_ids:", input_ids)
print("attention_mask:", attention_mask)
decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
print("decode 还原:", decoded)
```

**方案 B：Tokenizer + 生成（需本地有小模型）**  
在 [day08_llm_inference_demo.ipynb](day08_llm_inference_demo.ipynb) 已加载模型和 tokenizer 的前提下，可增加一栏：对同一 prompt 分别用 `do_sample=False` 和 `do_sample=True`、不同 `temperature` 或 `max_new_tokens` 调用 `model.generate()`，观察输出差异；打印一次 `tokenizer(prompt, return_tensors="pt")` 的 `input_ids`、`attention_mask`，理解 shape 与含义。更多完整示例见 [day09_tokenization_demo.py](day09_tokenization_demo.py)。

---

**与前后天的衔接**
- **Day 8**：Transformer Decoder 是 LLM 的骨架，Day 9 讲「怎么训练」（PT/SFT/RLHF）和「怎么喂数据、怎么生成」。
- **Day 10**：进入强化学习入门；Day 9 的 RLHF 概念会在 Day 12 展开。
- **verl**：做后训练（RLHF 等），数据需 tokenize，rollout 即自回归生成。

*完成 Day 9 后，可以进入 Day 10：强化学习入门。*

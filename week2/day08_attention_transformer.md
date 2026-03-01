# Day 8 详细学习指南：注意力机制与 Transformer

> 预计时长：2 小时 | 目标：理解 self-attention 和 Transformer 的基本结构

---

## 一、第一部分：注意力机制（约 1 小时）

### 1.1 必学知识点清单

| 知识点 | 说明 | 与 verl / LLM 的关系 |
|--------|------|----------------------|
| Query 查询 | 当前要关注的「问题」向量 | 每个 token 生成自己的 Q，用于检索相关信息 |
| Key 键 | 被查询的「索引」向量 | 每个 token 的 K，用于与 Q 计算相似度 |
| Value 值 | 实际要取用的「内容」向量 | 根据注意力权重加权求和得到输出 |
| Scaled Dot-Product Attention | Attention(Q,K,V) = softmax(QK^T/√d_k)V | LLM 的核心计算，verl 训练的就是包含它的模型 |
| 注意力权重 | softmax 后的概率分布，表示「关注谁」 | 决定每个位置对当前输出的贡献 |

### 1.2 建议学习顺序

1. **为什么需要注意力**（10 分钟）  
   - 全连接层：每个位置只能看到固定窗口，无法灵活关注任意位置  
   - RNN：顺序计算，难以并行，长序列易遗忘  
   - 注意力：任意位置两两可交互，可并行，能显式建模「谁重要」  
   - 练习：能说出「注意力机制」相比全连接/RNN 的一个核心优势

2. **Query、Key、Value 直观理解**（15 分钟）  
   - 检索类比：Q=「我要找什么」，K=「每条记录的索引」，V=「每条记录的内容」  
   - 计算流程：Q 与每个 K 算相似度 → softmax 得权重 → 对 V 加权求和  
   - 来源：Q、K、V 都由输入通过线性变换得到（W_Q、W_K、W_V）  
   - 练习：若序列长度 n=5，Q、K、V 的 shape 各是什么？（通常 (batch, n, d)）

3. **Scaled Dot-Product Attention 公式**（20 分钟）  
   - 评分：score = QK^T，表示 Q 与每个 K 的相似度  
   - 缩放：除以 √d_k，防止 d_k 大时 softmax 梯度消失  
   - 权重：α = softmax(score / √d_k)  
   - 输出：Output = αV，即按权重对 V 加权求和  
   - 练习：写出完整的 Attention(Q,K,V) 公式（一行）

4. **Self-Attention 自注意力**（10 分钟）  
   - Q、K、V 都来自同一输入序列（不同线性变换）  
   - 每个位置可以「看到」并关注序列中所有位置  
   - 与 Cross-Attention 区别：后者 Q 来自一个序列，K/V 来自另一个  
   - 练习：Self-Attention 中，若输入是「我 喜欢 猫」，每个词会关注哪些词？

5. **与第一周的衔接**（5 分钟）  
   - 注意力输出仍是向量，可接全连接层（FFN）  
   - 训练方式相同：前向 → loss → backward → 更新  
   - 练习：注意力层和 nn.Linear 的输入输出形式有何不同？

### 1.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 文字 | 《动手学深度学习》注意力机制 | https://zh.d2l.ai/chapter_attention-mechanisms/ | 第 10 章：注意力评分函数、多头注意力 |
| 博客 | The Illustrated Transformer | https://jalammar.github.io/illustrated-transformer/ | 图文并茂，理解 Q/K/V 和计算流程 |
| 视频 | 李沐《动手学深度学习》注意力机制 | B 站搜「李沐 注意力机制」 | 配合 D2L 书食用 |

**D2L 建议阅读顺序**：
- 10.1 注意力提示：理解「为什么需要注意力」
- 10.2 注意力汇聚：注意力权重的直观
- 10.3 注意力评分函数：Scaled Dot-Product 的数学形式

---

## 二、第二部分：Transformer 架构（约 1 小时）

### 2.1 必学知识点清单

| 知识点 | 说明 | 与 verl / LLM 的关系 |
|--------|------|----------------------|
| Encoder 编码器 | 双向注意力，看到整句，用于理解 | BERT 等 |
| Decoder 解码器 | 单向注意力（因果掩码），逐词生成 | GPT、LLM 的核心 |
| Multi-Head Attention | 多组 Q/K/V 并行计算，拼接后线性变换 | 从不同子空间建模关系 |
| FFN 前馈网络 | 两层全连接 + 激活，每位置独立计算 | 每个 Transformer block 都有 |
| LayerNorm | 对特征维度归一化，稳定训练 | 每个子层后都有 |
| Residual 残差连接 | 输出 = 子层输出 + 输入 | 缓解梯度消失，便于堆叠 |

### 2.2 建议学习顺序

1. **Encoder 与 Decoder 的区别**（15 分钟）  
   - Encoder：双向注意力，每个位置可看左右，适合「理解」任务  
   - Decoder：因果（Causal）注意力，位置 i 只能看 1,…,i，适合「生成」  
   - GPT 是纯 Decoder：只有 decoder 层堆叠，自回归生成  
   - 练习：为什么 LLM 用 Decoder 而不是 Encoder？

2. **一个 Transformer Block 的结构**（20 分钟）  
   - 以 Decoder 为例：  
     - 子层 1：Multi-Head Self-Attention + LayerNorm + Residual  
     - 子层 2：FFN（Linear → ReLU/GELU → Linear）+ LayerNorm + Residual  
   - 流程：x → Attention(x) + x → LayerNorm → FFN(...) + ... → LayerNorm → 输出  
   - 练习：画出「一个 Decoder Block」的框图，标出 5 个模块

3. **Multi-Head Attention**（15 分钟）  
   - 把 Q、K、V 拆成 h 个头，每个头独立做 Attention  
   - 输出：Concat(head_1,...,head_h) → 线性变换  
   - 作用：不同头可关注不同模式（如语法、指代、语义）  
   - 练习：8 个头、d_model=64，每个头的 d_k 是多少？（64/8=8）

4. **FFN 与 LayerNorm**（5 分钟）  
   - FFN：通常 d_model → 4×d_model → d_model，中间用 GELU  
   - LayerNorm：对最后一维做归一化，使每层输出分布稳定  
   - 练习：FFN 和 Attention 的输入输出维度有何不同？

5. **整体架构：堆叠 N 个 Block**（5 分钟）  
   - 输入：embedding + 位置编码  
   - N 个 Block 堆叠  
   - 输出：最后一层 hidden → 投影到词表大小 → logits  
   - 练习：能说出「从输入 token 到输出 logits」的完整数据流

### 2.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 博客 | The Illustrated GPT-2 | https://jalammar.github.io/illustrated-gpt2/ | 理解 GPT 的 Decoder 结构 |
| 论文 | Attention Is All You Need | 搜索「Transformer 原论文」 | 可选，了解原始设计 |
| 视频 | 李沐《动手学深度学习》Transformer | B 站搜「李沐 Transformer」 | 配合 D2L 第 11 章 |

**The Illustrated GPT-2 建议重点看**：
- GPT-2 的 Decoder 结构图
- 每个 Block 内部的 Attention + FFN 流程
- 自回归生成时如何「只看前面」

---

## 三、自测题

### 3.1 注意力机制自测

1. **Q、K、V**：在 Self-Attention 中，Q、K、V 从哪里来？它们的关系是什么？
2. **公式**：写出 Scaled Dot-Product Attention 的完整公式（含 softmax 和缩放）。
3. **为什么除以 √d_k**：不除以会有什么问题？

### 3.2 Transformer 自测（重点）

1. **Block 结构**：能画出 Transformer Decoder 的一个 block，并标出 Multi-Head Attention、FFN、LayerNorm、Residual 的位置。
2. **Encoder vs Decoder**：Encoder 和 Decoder 的注意力机制有何本质区别？
3. **Multi-Head**：若 d_model=256、head=8，每个头的 d_k 和 d_v 是多少？

### 3.3 参考答案

<details>
<summary>点击展开 注意力机制自测答案</summary>

1. Q、K、V 都由输入 X 通过线性变换得到：Q=XW_Q、K=XW_K、V=XW_V；三者都来自同一输入，但通过不同权重矩阵投影。  
2. Attention(Q,K,V) = softmax(QK^T / √d_k) V  
3. d_k 大时，QK^T 的值会很大，softmax 后梯度接近 0，导致梯度消失；除以 √d_k 使数值稳定。
</details>

<details>
<summary>点击展开 Transformer 自测答案</summary>

1. 框图：输入 → [Multi-Head Attention + Residual + LayerNorm] → [FFN + Residual + LayerNorm] → 输出  
2. Encoder 双向：每个位置可看所有位置；Decoder 因果：位置 i 只能看 1,…,i，用于自回归生成。  
3. d_k = d_v = d_model / head = 256 / 8 = 32
</details>

---

## 四、学习时间分配建议

| 时段 | 内容 | 时长 |
|------|------|------|
| 0:00–0:55 | 注意力机制（为什么需要→Q/K/V→Scaled Dot-Product→Self-Attention） | 55 分钟 |
| 0:55–1:00 | 休息 | 5 分钟 |
| 1:00–1:55 | Transformer 架构（Encoder/Decoder→Block 结构→Multi-Head→FFN/LayerNorm） | 55 分钟 |
| 1:55–2:00 | 完成自测题（重点：画出 Transformer block） | 5 分钟 |

---

## 五、今日学习检查清单

- [ ] 能解释 Query、Key、Value 的含义及在注意力中的作用
- [ ] 能写出 Scaled Dot-Product Attention 的公式
- [ ] 能解释为什么需要除以 √d_k
- [ ] 能区分 Encoder 与 Decoder 的注意力机制（双向 vs 因果）
- [ ] 能画出 Transformer Decoder 的一个 block，并说明各模块作用
- [ ] 能解释 Multi-Head Attention 的结构和动机
- [ ] 完成全部自测题

---

## 六、与前后天的衔接

- **Day 7**：第一周复习了神经网络、PyTorch 训练循环；注意力层和 FFN 都是「可学习的层」，训练方式相同。
- **Day 9**：大语言模型基础，会讲到 LLM 的训练阶段（PT、SFT、RLHF）；今天学的 Transformer 是 LLM 的骨架。
- **verl**：verl 训练的是基于 Transformer 的 LLM，理解 Attention 和 Block 结构后，能更好理解模型配置和 loss 计算。

---

## 七、可选实践：手写 Attention 公式

在 PyTorch 中实现一个简单的 Scaled Dot-Product Attention（不要求完整训练）：

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, seq_len, d_k)
    mask: 可选，(batch, seq_len, seq_len)，True 表示要遮蔽的位置
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights

# 简单测试
batch, seq, d = 2, 4, 8
Q = torch.randn(batch, seq, d)
K = torch.randn(batch, seq, d)
V = torch.randn(batch, seq, d)
out, weights = scaled_dot_product_attention(Q, K, V)
print(out.shape)   # (2, 4, 8)
print(weights.shape)  # (2, 4, 4)
```

---

*完成 Day 8 后，可以进入 Day 9：大语言模型基础。*

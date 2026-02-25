# verl 框架二次开发 · 三周速成学习计划

> 面向零基础转行新人 · 每天 2 小时 · 共 21 天

---

## 一、你需要掌握的知识体系

基于 verl 框架（RLHF/PPO/GRPO 等大模型后训练）的二次开发，建议按以下优先级掌握：

| 优先级 | 知识领域 | 与 verl 的关系 |
|--------|----------|----------------|
| 高 | Python 基础 | 框架与脚本均为 Python |
| 高 | PyTorch 基础 | 训练与模型加载均基于 PyTorch |
| 高 | 深度学习基础 | 理解神经网络、反向传播、优化器 |
| 高 | Transformer / LLM | verl 面向大语言模型后训练 |
| 高 | 强化学习基础 | PPO、RLHF 是 verl 的核心算法 |
| 中 | 分布式训练概念 | FSDP、Ray、多卡/多机 |
| 中 | verl 框架本身 | 配置、数据流、API 设计 |

---

## 二、三周学习路线总览

```
第 1 周：深度学习基础（能看懂训练循环和 loss）
第 2 周：LLM + 强化学习（理解 RLHF 在做什么）
第 3 周：verl 实战（跑通 PPO，看懂代码结构）
```

---

## 三、第一周：深度学习基础

### Day 1（2h）—— Python 与 NumPy

**目标**：能写简单脚本、会用 NumPy 做矩阵运算。

- [ ] **1h**：Python 基础  
  - 变量、列表、字典、函数、类  
  - 推荐：菜鸟教程 Python3 教程（前 15 章）或 [Python 官方教程](https://docs.python.org/zh-cn/3/tutorial/)
- [ ] **1h**：NumPy 入门  
  - `np.array`、`reshape`、矩阵乘法、广播  
  - 推荐：NumPy 官方 [Quickstart tutorial](https://numpy.org/doc/stable/user/quickstart.html)

**自测**：用 NumPy 实现一个简单的矩阵乘法。

---

### Day 2（2h）—— 线性代数与梯度

**目标**：理解向量、矩阵、梯度、链式法则。

- [ ] **1h**：线性代数基础  
  - 向量、矩阵、矩阵乘法、转置  
  - 推荐：3Blue1Brown 线性代数本质（B 站有中字）前 3–4 集
- [ ] **1h**：微积分与梯度  
  - 导数、偏导数、梯度、链式法则  
  - 推荐：吴恩达机器学习 Week 2（梯度下降部分）

**自测**：手算一个简单函数的梯度（如 f(x,y)=x²+y²）。

---

### Day 3（2h）—— 神经网络入门

**目标**：理解前向传播、反向传播、loss、优化器。

- [ ] **1h**：感知机与多层神经网络  
  - 神经元、激活函数、全连接层  
  - 推荐：李宏毅机器学习 2023「神经网络」一集
- [ ] **1h**：反向传播与梯度下降  
  - loss、梯度、参数更新  
  - 推荐：3Blue1Brown 神经网络系列（前 3 集）

**自测**：能解释「为什么需要反向传播」。

---

### Day 4（2h）—— PyTorch 基础（上）

**目标**：会用 PyTorch 定义张量、做运算、自动求导。

- [ ] **1h**：PyTorch 张量与运算  
  - `torch.tensor`、`device`、`dtype`、`reshape`、矩阵运算  
  - 推荐：PyTorch 官方 [60 分钟入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) 前 2 节
- [ ] **1h**：自动求导 `autograd`  
  - `requires_grad`、`backward()`、`grad`  
  - 推荐：同上教程的 autograd 部分

**自测**：用 PyTorch 实现一个简单函数并求梯度。

---

### Day 5（2h）—— PyTorch 基础（下）

**目标**：会用 `nn.Module` 和 `optimizer` 写一个完整训练循环。

- [ ] **1h**：`nn.Module` 与 `nn.Linear`  
  - 定义网络、`forward`、`parameters()`  
  - 推荐：PyTorch 官方教程「神经网络」部分
- [ ] **1h**：训练循环  
  - `optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()`  
  - 推荐：手写一个 MNIST 或线性回归的完整训练脚本

**自测**：能独立写出一个「定义模型 + 训练循环」的完整脚本。

---

### Day 6（2h）—— 深度学习概念补充

**目标**：理解 batch、epoch、过拟合、正则化等概念。

- [ ] **1h**：训练相关概念  
  - batch size、epoch、learning rate、learning rate schedule  
  - 推荐：吴恩达深度学习专项 Course 1 Week 2
- [ ] **1h**：过拟合与正则化  
  - 过拟合、dropout、L2 正则  
  - 推荐：李宏毅「过拟合」相关一集

**自测**：能解释 batch size 和 learning rate 对训练的影响。

---

### Day 7（2h）—— 第一周复习与小结

**目标**：巩固第一周内容，为第二周打基础。

- [ ] **1h**：复习  
  - 重看笔记，整理「神经网络 + PyTorch 训练」的思维导图
- [ ] **1h**：实践  
  - 用 PyTorch 完成一个简单分类任务（如 MNIST 或 Iris）

---

## 四、第二周：LLM 与强化学习

### Day 8（2h）—— 注意力机制与 Transformer

**目标**：理解 self-attention 和 Transformer 的基本结构。

- [ ] **1h**：注意力机制  
  - Query、Key、Value、Scaled Dot-Product Attention  
  - 推荐：李沐《动手学深度学习》注意力机制章节，或 The Illustrated Transformer
- [ ] **1h**：Transformer 架构  
  - Encoder、Decoder、Multi-Head Attention、FFN、LayerNorm  
  - 推荐：[The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)

**自测**：能画出 Transformer 的一个 block 并说明各模块作用。

---

### Day 9（2h）—— 大语言模型（LLM）基础

**目标**：理解 LLM 的训练阶段和基本概念。

- [ ] **1h**：LLM 训练流程  
  - 预训练（PT）、监督微调（SFT）、RLHF/后训练  
  - 推荐：知乎/博客「大模型训练流程」类文章
- [ ] **1h**：Tokenization 与生成  
  - 分词、`input_ids`、`attention_mask`、自回归生成  
  - 推荐：HuggingFace Transformers 文档「Text generation」部分

**自测**：能说明 SFT 和 RLHF 分别解决什么问题。

---

### Day 10（2h）—— 强化学习入门

**目标**：理解 MDP、策略、奖励、价值函数。

- [ ] **1h**：RL 基本概念  
  - 状态、动作、策略、奖励、回报  
  - 推荐：李宏毅「强化学习」系列前 2 集，或 Sutton RL 书第 1–3 章
- [ ] **1h**：策略梯度  
  - Policy Gradient、REINFORCE  
  - 推荐：李宏毅「策略梯度」一集

**自测**：能解释「策略梯度」和「监督学习」的区别。

---

### Day 11（2h）—— PPO 算法

**目标**：理解 PPO 的核心思想和实现要点。

- [ ] **1h**：PPO 原理  
  - 重要性采样、 clipped objective、KL 惩罚  
  - 推荐：李宏毅「Proximal Policy Optimization」或 Spinning Up PPO 文档
- [ ] **1h**：PPO 在 LLM 中的角色  
  - Actor-Critic、advantage、GAE  
  - 推荐：OpenAI InstructGPT 论文（RLHF 部分）

**自测**：能说明 PPO 为什么要做 clip，以及和 REINFORCE 的区别。

---

### Day 12（2h）—— RLHF 全流程

**目标**：理解 RLHF 三阶段和 reward model。

- [ ] **1h**：RLHF 三阶段  
  - SFT → Reward Model → RL（PPO）  
  - 推荐：Anthropic RLHF 博客或 InstructGPT 论文
- [ ] **1h**：Reward Model 与偏好数据  
  - 偏好对 (chosen, rejected)、reward 计算  
  - 推荐：verl 文档 [Implement Reward Function](https://verl.readthedocs.io/en/v0.5.x/preparation/reward_function.html)

**自测**：能画出 RLHF 三阶段的流程图并说明每步输入输出。

---

### Day 13（2h）—— GRPO / DAPO 等算法概览

**目标**：知道 verl 支持哪些算法，各自特点。

- [ ] **1h**：GRPO  
  - Group Relative Policy Optimization，组内相对优化  
  - 推荐：verl 文档 [GRPO](https://verl.readthedocs.io/en/v0.5.x/algo/grpo.html)
- [ ] **1h**：DAPO 及其他  
  - DAPO、OPO、SPPO 等简要概念  
  - 推荐：verl 文档 Algorithms 部分浏览

**自测**：能说出 PPO、GRPO、DAPO 的主要区别。

---

### Day 14（2h）—— 第二周复习与小结

**目标**：串联 LLM + RL + RLHF。

- [ ] **1h**：复习  
  - 整理「LLM 训练阶段 → RLHF → PPO」的完整链路
- [ ] **1h**：阅读  
  - 精读 InstructGPT 或 RLHF 相关论文摘要和图表

---

## 五、第三周：verl 实战

### Day 15（2h）—— verl 安装与 Quickstart

**目标**：环境搭好，跑通官方示例。

- [ ] **1h**：安装  
  - 按 [verl 安装文档](https://verl.readthedocs.io/en/v0.5.x/start/install.html) 配置环境  
  - 注意：CUDA 版本、PyTorch 版本、vLLM 等依赖
- [ ] **1h**：Quickstart  
  - 按 [PPO on GSM8K](https://verl.readthedocs.io/en/v0.5.x/start/quickstart.html) 跑通  
  - 理解：数据格式、配置项、训练命令

**自测**：能成功启动一次 PPO 训练（可用小模型/小数据）。

---

### Day 16（2h）—— verl 配置与数据准备

**目标**：理解配置结构和数据格式。

- [ ] **1h**：配置说明  
  - 阅读 [ppo_trainer.yaml 说明](https://verl.readthedocs.io/en/v0.5.x/examples/config.html#ppo-trainer-yaml-for-rl-fsdp-backend)  
  - 理解：train_batch_size、mini_batch_size、micro_batch_size 等
- [ ] **1h**：数据准备  
  - [Prepare Data for Post-Training](https://verl.readthedocs.io/en/v0.5.x/preparation/prepare_data.html)  
  - [Implement Reward Function](https://verl.readthedocs.io/en/v0.5.x/preparation/reward_function.html)

**自测**：能修改配置并准备一个自己的小数据集。

---

### Day 17（2h）—— verl 代码结构

**目标**：能定位 PPO 训练的主流程和关键模块。

- [ ] **1h**：HybridFlow 与代码组织  
  - 阅读 [Codebase walkthrough (PPO)](https://verl.readthedocs.io/en/v0.5.x/hybrid_flow.html#codebase-walkthrough-ppo)  
  - 理解：single_controller、rollout、train 等概念
- [ ] **1h**：浏览源码  
  - `verl/` 目录结构、`ppo_trainer`、`fsdp_workers` 等  
  - 在 IDE 中搜索 `def train`、`def rollout` 等关键函数

**自测**：能说出「一次 PPO 迭代」的大致数据流。

---

### Day 18（2h）—— verl 扩展点

**目标**：知道如何加新算法、新模型、新 reward。

- [ ] **1h**：扩展 RL 算法  
  - [Extend to other RL(HF) algorithms](https://verl.readthedocs.io/en/v0.5.x/advance/dpo_extension.html)  
  - 理解：如何接入新的 loss 或算法
- [ ] **1h**：添加新模型  
  - [Add models with FSDP backend](https://verl.readthedocs.io/en/v0.5.x/advance/fsdp_extension.html)  
  - 理解：模型注册、config 映射

**自测**：能列出「如果要加一个新 reward 函数」需要改哪些文件。

---

### Day 19（2h）—— 分布式与性能

**目标**：理解多卡、多机训练和基本调优。

- [ ] **1h**：分布式训练  
  - [Multinode Training](https://verl.readthedocs.io/en/v0.5.x/start/multinode.html)  
  - FSDP、Ray、设备映射等概念
- [ ] **1h**：性能调优  
  - [Performance Tuning Guide](https://verl.readthedocs.io/en/v0.5.x/perf/perf_tuning.html)  
  - batch size、micro batch、设备放置等

**自测**：能解释 train_batch_size、mini_batch_size、micro_batch_size 的含义。

---

### Day 20（2h）—— 实战：小改动

**目标**：完成一个实际的小改动（如改 reward、改 prompt）。

- [ ] **2h**：动手实践  
  - 选一个：修改 reward 函数 / 修改 prompt 模板 / 换一个小模型  
  - 跑通并观察 loss / reward 变化

**自测**：能独立完成一次「改配置或改代码 → 跑通训练」的流程。

---

### Day 21（2h）—— 第三周总结与后续规划

**目标**：整理知识，规划后续学习。

- [ ] **1h**：总结  
  - 写一份「verl 学习笔记」：概念、配置、代码结构、扩展方式
- [ ] **1h**：规划  
  - 列出工作中可能遇到的场景（如新算法、新任务、新模型）  
  - 制定后续 1–2 个月的学习重点

---

## 六、推荐学习资源汇总

| 类型 | 资源 | 用途 |
|------|------|------|
| 视频 | 李宏毅机器学习 2023 | 神经网络、RL、PPO |
| 视频 | 3Blue1Brown 线性代数/神经网络 | 直观理解 |
| 视频 | 吴恩达机器学习/深度学习 | 系统入门 |
| 文档 | PyTorch 官方教程 | PyTorch 实战 |
| 文档 | verl 官方文档 | 框架使用与扩展 |
| 论文 | InstructGPT | RLHF 全流程 |
| 博客 | The Illustrated Transformer / GPT-2 | Transformer 与 LLM |

---

## 七、学习建议

1. **动手优先**：每学一个概念，尽量写一小段代码验证。
2. **先跑通再深挖**：第三周先跑通官方示例，再逐步理解实现细节。
3. **记笔记**：用思维导图或 Markdown 整理「概念 → 在 verl 中的对应」。
4. **遇到问题**：优先查 verl 文档、FAQ、GitHub issues。
5. **保持节奏**：每天 2 小时坚持 21 天，比偶尔突击更有效。

---

## 八、三周后的预期水平

- 能解释：深度学习训练流程、Transformer、PPO、RLHF。
- 能操作：安装 verl、跑通 PPO、修改配置、准备数据。
- 能定位：verl 中训练主流程、reward、模型加载等关键代码。
- 能扩展：在指导下完成简单的 reward 修改或算法接入。

更深入的分布式调优、新算法实现等，可在后续工作中边做边学。

---

*祝学习顺利！有问题可参考 verl 官方文档与社区。*

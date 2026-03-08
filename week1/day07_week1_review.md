# Day 7 详细学习指南：第一周复习与小结

> 目标：巩固第一周内容，为第二周打基础

---

## 学习时间分配建议

| 时段 | 内容 | 时长 |
|------|------|------|
| 0:00–0:30 | 第一部分：过一遍思维导图，完成核心概念自检 | 30 分钟 |
| 0:30–0:35 | 休息 | 5 分钟 |
| 0:35–1:25 | 第二部分：实践（Iris 或 MNIST 分类） | 50 分钟 |
| 1:25–2:00 | 自测与检查、整理笔记 | 35 分钟 |

**预计总时长：2 小时**

---

## 一、第一周知识回顾

### 1.1 思维导图：神经网络 + PyTorch 训练

```
第一周：深度学习基础
│
├── 基础工具（Day 1-2）
│   ├── Python：list、dict、函数、类
│   ├── NumPy：array、reshape、矩阵乘法 @
│   ├── 线性代数：向量、矩阵、线性变换
│   └── 微积分：导数、偏导、梯度、链式法则
│
├── 神经网络（Day 3）
│   ├── 神经元：加权求和 + 激活函数
│   ├── 激活函数：ReLU、Sigmoid（引入非线性）
│   ├── 全连接层：y = Wx + b
│   ├── 前向传播：输入 → 隐藏层 → 输出
│   ├── Loss：MSE、CrossEntropy
│   ├── 反向传播：链式法则，从输出往输入算梯度
│   └── 训练循环四步：zero_grad → 前向 → backward → step
│
├── PyTorch（Day 4-5）
│   ├── 张量：torch.tensor、shape、device、dtype
│   ├── 自动求导：requires_grad、backward()、.grad
│   ├── nn.Module：继承、forward、parameters
│   ├── nn.Linear、nn.ReLU、nn.Sequential
│   ├── 损失函数：MSELoss、CrossEntropyLoss
│   └── 优化器：SGD、Adam
│
└── 训练概念（Day 6）
    ├── batch、epoch、iteration
    ├── learning rate、lr schedule
    ├── 过拟合、欠拟合
    └── Dropout、L2 正则（weight_decay）
```

### 1.2 核心概念自检（约 30 分钟）

按顺序过一遍，能解释或写出即通过：

| 序号 | 概念 | 自检 |
|------|------|------|
| 1 | 梯度下降公式 | θ ← θ − lr × ∇θ |
| 2 | 为什么需要反向传播 | 高效算每层梯度，避免对每个参数单独求导 |
| 3 | 训练循环四步 | zero_grad → 前向 → loss.backward → step |
| 4 | requires_grad 的作用 | 标记需要求梯度的张量 |
| 5 | 为什么需要 zero_grad | 避免梯度累积 |
| 6 | nn.Linear(10, 5) 的含义 | 输入 10 维，输出 5 维的全连接层 |
| 7 | model.train() 与 model.eval() | 训练时 Dropout 生效，推理时关闭 |
| 8 | batch size 对训练的影响 | 大→梯度稳但慢、占显存；小→快但噪声大 |
| 9 | learning rate 过大/过小 | 过大发散，过小收敛慢 |
| 10 | Dropout 的作用 | 训练时随机丢弃神经元，防过拟合 |
| 11 | weight_decay 是什么 | L2 正则，惩罚大权重 |

### 1.3 自测与检查

**自测题**
从上表中任选 3–5 条，能解释或写出即通过（例如：训练循环四步、为什么需要 zero_grad、batch size 对训练的影响）。

**检查清单**
- [ ] 能解释梯度下降公式、反向传播、训练循环四步
- [ ] 能解释 requires_grad、zero_grad、model.train()/eval()
- [ ] 能解释 batch size、learning rate、Dropout、weight_decay 的作用

---

## 二、实践任务：用 PyTorch 完成分类任务（约 1 小时）

### 2.1 任务选择

| 任务 | 难度 | 数据量 | 建议 |
|------|------|--------|------|
| Iris 鸢尾花分类 | ⭐ 简单 | 150 样本，4 特征，3 类 | 快速完成，巩固流程 |
| MNIST 手写数字 | ⭐⭐ 中等 | 6 万训练，28×28 图像，10 类 | 经典入门，推荐 |

**建议**：若时间紧选 Iris；若想多练选 MNIST。

---

### 2.2 方案 A：Iris 鸢尾花分类（约 40 分钟）

**数据**：`sklearn.datasets.load_iris()`，150 样本，4 特征，3 类。

**模型**：`nn.Sequential(Linear(4, 16), ReLU(), Dropout(0.2), Linear(16, 3))`

**目标**：训练集准确率 > 90%，理解完整流程。

**步骤**：
1. 加载数据，划分 train/val（如 80%/20%）
2. 转为 Tensor，可选标准化
3. 定义模型、损失（CrossEntropyLoss）、优化器（Adam）
4. 训练循环：多 epoch，每 epoch 遍历所有数据（或小 batch）
5. 验证集评估准确率

**参考代码框架**（尽量先自己写，再对照）：

```python
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 加载数据
X, y = load_iris(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)  # CrossEntropy 需要 Long
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)

# 2. 模型
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(16, 3)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 3. 训练
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()

# 4. 验证
model.eval()
with torch.no_grad():
    pred = model(X_val)
    acc = (pred.argmax(1) == y_val).float().mean().item()
print(f"Val Accuracy: {acc:.2%}")
```

---

### 2.3 方案 B：MNIST 手写数字（约 1 小时）

**数据**：`torchvision.datasets.MNIST`，28×28 灰度图，10 类。

**模型**：`Flatten → Linear(784, 256) → ReLU → Dropout(0.2) → Linear(256, 10)`

**目标**：验证集准确率 > 95%。

**步骤**：
1. 用 `torchvision.datasets.MNIST` + `DataLoader` 加载
2. 模型：展平 28×28=784，两层全连接
3. 训练循环：多 epoch，按 batch 遍历
4. 每若干 epoch 在验证集上算准确率

**参考代码框架**：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
val_set = datasets.MNIST("./data", train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=256)

# 2. 模型
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. 训练
for epoch in range(10):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    # 4. 验证
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"Epoch {epoch+1}, Val Acc: {correct/total:.2%}")
```

### 2.4 自测与检查

**检查清单**
完成实践后，确认以下能力：

- [ ] 能加载数据并转为 Tensor
- [ ] 能定义 `nn.Sequential` 或继承 `nn.Module` 的网络
- [ ] 能正确使用 `CrossEntropyLoss`（输入 logits，target 为类别索引）
- [ ] 能写出完整的「多 epoch + 多 batch」训练循环
- [ ] 能在验证集上计算准确率
- [ ] 能使用 `model.eval()` 和 `torch.no_grad()` 做推理
- [ ] （MNIST）能使用 `DataLoader` 按 batch 加载数据

---

**与前后天的衔接**
- **本周小结**：完成 Day 7 后，你应能理解神经网络前向/反向、梯度下降、过拟合与正则化，使用 PyTorch 张量、autograd、nn.Module、优化器，并独立完成一个「定义模型 + 训练循环 + 验证」的分类任务。
- **Day 8 起**：进入 LLM 与强化学习（注意力机制与 Transformer → 大语言模型基础 → 强化学习 → PPO → RLHF），第一周打下的基础是理解 Transformer 和 RLHF 的前提。

*完成 Day 7 后，可以进入 Day 8：注意力机制与 Transformer。*

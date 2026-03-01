# Day 5 详细学习指南：PyTorch 基础（下）——nn.Module 与训练循环

> 目标：会用 `nn.Module` 和 `optimizer` 写一个完整训练循环

---

## 学习时间分配建议

| 时段 | 内容 | 时长 |
|------|------|------|
| 0:00–0:55 | 第一部分：nn.Module 与 nn.Linear（Linear→继承定义→Sequential→parameters→train/eval） | 55 分钟 |
| 0:55–1:00 | 休息 | 5 分钟 |
| 1:00–1:55 | 第二部分：训练循环（损失函数→优化器→四步→线性回归完整示例） | 55 分钟 |
| 1:55–2:00 | 自测与检查（独立写一遍线性回归脚本） | 5 分钟 |

**预计总时长：2 小时**

---

## 一、第一部分：nn.Module 与 nn.Linear（约 1 小时）

### 1.1 必学知识点清单

| 知识点 | 说明 | 与 verl / 深度学习的关系 |
|--------|------|---------------------------|
| `nn.Module` | 所有神经网络层的基类，封装参数和计算 | verl 中的模型都继承 nn.Module |
| `nn.Linear` | 全连接层，y = xW^T + b | 最基础的层，对应 Day 3 的线性变换 |
| `forward()` | 定义前向传播，输入→输出 | 调用 `model(x)` 时自动执行 forward |
| `parameters()` | 返回模型所有可学习参数 | 优化器需要传入 model.parameters() |
| `nn.Sequential` | 按顺序组合多个层 | 快速搭建简单网络 |

### 1.2 建议学习顺序

1. **nn.Linear 全连接层**（15 分钟）  
   - `nn.Linear(in_features, out_features)`：输入维度、输出维度  
   - 内部：y = xW^T + b，W 和 b 自动初始化  
   - 示例：`layer = nn.Linear(10, 5)`，输入 (batch, 10) → 输出 (batch, 5)  
   - 练习：创建一个 3→4 的 Linear 层，传入 (2, 3) 的输入，看输出形状

2. **继承 nn.Module 定义网络**（20 分钟）  
   - `class MyNet(nn.Module):`  
   - `__init__`：用 `super().__init__()`，定义 `self.layer1 = nn.Linear(...)`  
   - `forward(self, x)`：写前向逻辑，`return` 最终输出  
   - 注意：只写 `forward`，不要重写 `__call__`（父类已处理）  
   - 练习：定义一个 2 层网络：输入 4 维 → 隐藏 8 维（ReLU）→ 输出 2 维

3. **nn.Sequential**（10 分钟）  
   - `nn.Sequential(layer1, layer2, ...)` 按顺序执行  
   - 适合没有分支的简单网络  
   - 练习：用 Sequential 实现上面的 2 层网络

4. **model.parameters() 与 state_dict**（10 分钟）  
   - `model.parameters()`：生成器，遍历所有可学习参数（W、b）  
   - `model.state_dict()`：字典，参数名→张量，用于保存/加载  
   - 优化器需要：`optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`  
   - 练习：打印一个简单模型的 parameters 和 state_dict 的 keys

5. **model.train() 与 model.eval()**（5 分钟）  
   - `train()`：训练模式，Dropout 等会生效  
   - `eval()`：评估模式，Dropout 关闭  
   - 推理时记得 `model.eval()`  
   - 练习：切换模式并理解区别（Day 6 会讲 Dropout）

### 1.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 官方教程 | PyTorch 构建模型 | https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html | 精读 |
| 官方教程 | torch.nn 是什么 | https://pytorch.org/tutorials/beginner/nn_tutorial.html | 可选，更细 |
| 中文 | PyTorch 中文文档 | https://docs.pytorch.ac.cn/tutorials/beginner/basics/buildmodel_tutorial.html | 同上，中文版 |

### 1.4 自测与检查

**自测题**  
1. 定义一个类 `TwoLayerNet`：输入 5 维 → Linear(5, 10) + ReLU → Linear(10, 2)，实现 `forward`。  
2. 用 `nn.Sequential` 实现同样的网络。  
3. 写出把模型参数传给 Adam 优化器的代码。

<details>
<summary>点击展开参考答案</summary>

```python
import torch.nn as nn

# 1. 继承 nn.Module
class TwoLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 2. Sequential
model = nn.Sequential(nn.Linear(5, 10), nn.ReLU(), nn.Linear(10, 2))

# 3. 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
</details>

**检查清单**  
- [ ] 能继承 `nn.Module` 定义网络并实现 `forward`  
- [ ] 能使用 `nn.Linear`、`nn.ReLU`、`nn.Sequential`  

---

## 二、第二部分：训练循环（约 1 小时）

### 2.1 必学知识点清单

| 知识点 | 说明 | 与 verl / 深度学习的关系 |
|--------|------|---------------------------|
| 损失函数 | `nn.MSELoss`、`nn.CrossEntropyLoss` 等 | 衡量预测与真实的差距 |
| 优化器 | `torch.optim.SGD`、`torch.optim.Adam` | 用梯度更新参数 |
| `zero_grad()` | 清零上一步的梯度 | 避免梯度累积 |
| `backward()` | 反向传播算梯度 | Day 4 已学 |
| `step()` | 用梯度更新参数 | 完成一次参数更新 |

### 2.2 建议学习顺序

1. **损失函数**（10 分钟）  
   - `nn.MSELoss()`：回归，均方误差  
   - `nn.CrossEntropyLoss()`：分类，输入 logits（未 softmax），target 为类别索引  
   - 用法：`criterion = nn.MSELoss()`，`loss = criterion(pred, target)`  
   - 练习：手算一个简单 MSE，用 PyTorch 验证

2. **优化器**（10 分钟）  
   - `optimizer = torch.optim.SGD(model.parameters(), lr=0.01)`  
   - `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`（更常用）  
   - 优化器持有参数的引用，`step()` 时按梯度更新  
   - 练习：创建模型和 Adam 优化器

3. **训练循环四步**（20 分钟）  
   - ① `optimizer.zero_grad()`：清零梯度（重要！否则会累积）  
   - ② `output = model(input)`：前向  
   - ③ `loss = criterion(output, target)`，`loss.backward()`：算 loss 并反向传播  
   - ④ `optimizer.step()`：更新参数  
   - 练习：在一个 for 循环里写齐这四步

4. **完整示例：线性回归**（15 分钟）  
   - 数据：y = 2x + 1 + 噪声，构造 100 个点  
   - 模型：`nn.Linear(1, 1)`  
   - 训练：多个 epoch，每 epoch 遍历数据，四步更新  
   - 目标：学出 w≈2, b≈1  
   - 练习：完整跑通，打印训练前后的 w、b

5. **可选：MNIST 简化版**（5 分钟）  
   - 若时间充裕：用 `torchvision.datasets.MNIST` 加载数据  
   - 模型：Flatten → Linear(784, 128) → ReLU → Linear(128, 10)  
   - 损失：CrossEntropyLoss，优化器：Adam  
   - 可留到复习时做

### 2.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 官方教程 | PyTorch 训练模型 | https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html | 精读 |
| 官方教程 | 完整训练流程 | https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html | 可选 |
| 文字 | 《动手学深度学习》线性回归 | https://zh.d2l.ai/chapter_linear-networks/linear-regression-concise.html | 配合代码 |

**完整代码示例：线性回归**

```python
import torch
import torch.nn as nn

# 1. 构造数据：y = 2x + 1 + 噪声
torch.manual_seed(42)
x = torch.randn(100, 1) * 3
y = 2 * x + 1 + torch.randn(100, 1) * 0.5

# 2. 定义模型
model = nn.Linear(1, 1)
print("训练前:", model.weight.item(), model.bias.item())

# 3. 损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4. 训练循环
for epoch in range(100):
    optimizer.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("训练后:", model.weight.item(), model.bias.item())
# 应接近 2 和 1
```

### 2.4 自测与检查

**自测题**  
1. 训练循环四步：请按正确顺序写出（用中文或代码均可）。  
2. 为什么需要 zero_grad？若忘记调用，会发生什么？  
3. 不参考示例，独立写一个「线性回归」的完整训练脚本（可简化数据）。

<details>
<summary>点击展开参考答案</summary>

1. 四步：zero_grad → 前向(算 pred) → loss + backward → step  
2. 梯度会累积，导致更新方向错误、训练不稳定  
3. 参考上面「完整代码示例」
</details>

**检查清单**  
- [ ] 能写出训练循环四步：zero_grad → 前向 → loss.backward → step  
- [ ] 能独立写出一个「定义模型 + 训练循环」的完整脚本（线性回归即可）  
- [ ] 理解为什么需要 `zero_grad()`  
- [ ] 完成全部自测题  

---

**与前后天的衔接**
- **Day 3**：训练循环四步的理论，今天用 PyTorch 实现。
- **Day 4**：`backward()` 算梯度，今天配合 `optimizer.step()` 完成更新。
- **Day 6**：讲 batch、epoch、过拟合等。
- **verl**：训练大模型本质也是同一套「前向→loss→反向→更新」循环。

*完成 Day 5 后，可以进入 Day 6：深度学习概念补充（batch、epoch、过拟合、正则化）。*

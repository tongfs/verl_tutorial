# Day 4 详细学习指南：PyTorch 基础（上）——张量与自动求导

> 目标：会用 PyTorch 定义张量、做运算、自动求导

---

## 学习时间分配建议

| 时段 | 内容 | 时长 |
|------|------|------|
| 0:00–0:55 | 第一部分：PyTorch 张量与运算（创建→属性→device→reshape→矩阵运算） | 55 分钟 |
| 0:55–1:00 | 休息 | 5 分钟 |
| 1:00–1:55 | 第二部分：自动求导（requires_grad→backward→链式验证→梯度累积→no_grad） | 55 分钟 |
| 1:55–2:00 | 自测与检查（重点：用 PyTorch 实现简单函数并求梯度） | 5 分钟 |

**预计总时长：2 小时**

---

## 一、第一部分：PyTorch 张量与运算（约 1 小时）

### 1.1 必学知识点清单

| 知识点 | 说明 | 与 verl / 深度学习的关系 |
|--------|------|---------------------------|
| `torch.tensor` | 创建张量，类似 NumPy 的 array | 模型输入、权重、中间结果都是张量 |
| `device` | CPU 或 GPU（cuda），决定计算在哪执行 | verl 训练在 GPU 上 |
| `dtype` | 数据类型：float32、int64 等 | 模型通常用 float32 |
| `reshape` / `view` | 改变张量形状 | 维度变换，如 (batch, seq) → (batch*seq,) |
| 矩阵运算 | `@`、`*`、`+`、`-` | 神经网络核心运算 |
| 与 NumPy 互转 | `tensor.numpy()`、`torch.from_numpy()` | 数据预处理常用 |

### 1.2 建议学习顺序

1. **创建张量**（15 分钟）  
   - `torch.tensor([1, 2, 3])`、`torch.zeros(2, 3)`、`torch.ones(2, 3)`  
   - `torch.arange(0, 10, 2)`、`torch.randn(2, 3)`（随机）  
   - 指定 `dtype`：`torch.tensor([1.0, 2.0], dtype=torch.float32)`  
   - 练习：创建一个形状 (3, 4) 的随机张量

2. **张量属性**（10 分钟）  
   - `.shape` 或 `.size()`：形状  
   - `.dtype`：数据类型  
   - `.device`：所在设备  
   - 练习：打印上面创建张量的 shape、dtype、device

3. **device：CPU 与 GPU**（10 分钟）  
   - `x.device` 查看；`x.to('cuda')` 或 `x.cuda()` 移到 GPU  
   - `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`  
   - 运算时参与运算的张量需在同一 device  
   - 练习：若有 GPU，把张量移到 cuda 并做一次加法

4. **reshape 与 view**（10 分钟）  
   - `x.reshape(2, 6)`、`x.view(2, 6)`（view 要求内存连续）  
   - `x.flatten()` 或 `x.view(-1)` 展平  
   - `-1` 表示自动推断：`x.view(2, -1)`  
   - 练习：把 (4, 5) 变成 (10, 2)

5. **矩阵与逐元素运算**（15 分钟）  
   - 逐元素：`+`、`-`、`*`、`/`、`**`  
   - 矩阵乘法：`A @ B` 或 `torch.mm(A, B)`  
   - 广播：与 NumPy 规则相同  
   - 练习：两个 2×3 张量做逐元素乘；两个 2×3 和 3×2 做矩阵乘

### 1.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 官方教程 | PyTorch 60 分钟入门 | https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html | 前 2 节：Tensors、Datasets & DataLoaders（可略过 DataLoaders） |
| 官方文档 | torch.Tensor | https://pytorch.org/docs/stable/tensors.html | 查阅 API |
| 中文 | PyTorch 中文文档 | https://pytorch.apachecn.org/ | 可选 |

**PyTorch 60min 入门建议阅读**：
- **Tensors**：完整阅读，边看边在 Jupyter/Python 里敲
- **Datasets & DataLoaders**：可略过或快速浏览，Day 5 会用到

### 1.4 自测与检查

**自测题**  
1. 创建一个形状为 (2, 3, 4) 的张量，用 `reshape` 变成 (6, 4)。  
2. 写出把张量 `x` 从 CPU 移到 GPU 的代码（若可用）。  
3. 计算 `A @ B`：A 形状 (2, 3)，B 形状 (3, 4)，结果形状是多少？

<details>
<summary>点击展开参考答案</summary>

```python
import torch
# 1
x = torch.randn(2, 3, 4)
y = x.reshape(6, 4)
# 2
x = x.to('cuda')  # 或 x.cuda()
# 3
A = torch.randn(2, 3)
B = torch.randn(3, 4)
C = A @ B  # 形状 (2, 4)
```
</details>

**检查清单**  
- [ ] 能创建张量并指定 dtype、device  
- [ ] 能使用 reshape/view 改变张量形状  
- [ ] 能区分逐元素运算与矩阵乘法 `@`  

---

## 二、第二部分：自动求导 autograd（约 1 小时）

### 2.1 必学知识点清单

| 知识点 | 说明 | 与 verl / 深度学习的关系 |
|--------|------|---------------------------|
| `requires_grad=True` | 标记需要求梯度的张量 | 模型参数、有时输入也需要 |
| `backward()` | 从调用者开始反向传播，计算梯度 | 对应 Day 3 的反向传播 |
| `.grad` | 张量的梯度，backward 后才有 | 优化器用梯度更新参数 |
| 计算图 | 前向时记录操作，反向时按图求导 | PyTorch 自动构建，无需手写 |

### 2.2 建议学习顺序

1. **requires_grad**（10 分钟）  
   - `x = torch.tensor([1.0, 2.0], requires_grad=True)`  
   - 参与运算的张量，若有一个 requires_grad=True，结果也会追踪  
   - `x.requires_grad` 查看  
   - 练习：创建 requires_grad=True 的张量，做 y=x*2，看 y.requires_grad

2. **简单 backward**（15 分钟）  
   - `y = x ** 2`，`y.backward()`，则 `x.grad` 为 dy/dx  
   - 注意：`backward()` 默认对标量调用，非标量需传 `gradient` 或先 `sum()`  
   - 练习：`x = torch.tensor(3.0, requires_grad=True)`，`y = x**2`，backward 后打印 x.grad（应为 6）

3. **链式法则验证**（15 分钟）  
   - `z = x*y`，`L = z**2`，backward 后看 x.grad、y.grad  
   - 与 Day 2 链式法则对照：∂L/∂x = ∂L/∂z * ∂z/∂x  
   - 练习：手算验证 PyTorch 给出的梯度是否正确

4. **梯度累积与清零**（10 分钟）  
   - 多次 backward 时，梯度会**累加**到 .grad 上  
   - 训练时每次迭代前需 `optimizer.zero_grad()` 清零（Day 5 详讲）  
   - 练习：对同一计算图 backward 两次，观察 grad 是否翻倍

5. **no_grad 与 detach**（10 分钟）  
   - `with torch.no_grad():` 块内不追踪梯度，省内存、提速  
   - 推理时通常用 `with torch.no_grad():`  
   - `x.detach()` 得到不需要梯度的新张量  
   - 练习：在 no_grad 块内做运算，确认结果无 grad

### 2.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 官方教程 | PyTorch 60min 入门 - Autograd | 同上页面，Autograd 部分 | 精读 |
| 官方文档 | torch.autograd | https://pytorch.org/docs/stable/autograd.html | 查阅 API |
| 文字 | 《动手学深度学习》自动求导 | https://zh.d2l.ai/chapter_preliminaries/autograd.html | 配合代码 |

### 2.4 自测与检查

**自测题**  
1. 用 PyTorch 求梯度：设 `f(x) = x² + 2x`，在 x=3 处求 df/dx。（答案：2x+2 = 8）  
2. 多元函数：设 `f(x, y) = x²y`，x=2, y=3。求 ∂f/∂x 和 ∂f/∂y。（答案：∂f/∂x=12，∂f/∂y=4）  
3. 写一段不超过 10 行的代码，对 `z = x*y + x**2` 在 x=1, y=2 处求 ∂z/∂x 和 ∂z/∂y。

<details>
<summary>点击展开参考答案</summary>

```python
import torch
# 1. f(x)=x²+2x, x=3
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x
y.backward()
print(x.grad)  # tensor(8.)

# 2. f(x,y)=x²y, x=2, y=3
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
L = x**2 * y
L.backward()
print(x.grad, y.grad)  # tensor(12.) tensor(4.)

# 3. z = x*y + x**2, x=1, y=2
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x*y + x**2
z.backward()
print(x.grad, y.grad)  # tensor(4.) tensor(1.)
```
</details>

**检查清单**  
- [ ] 能对 `requires_grad=True` 的张量做运算并调用 `backward()`  
- [ ] 能解释 `.grad` 的含义及梯度累积问题  
- [ ] 能用 PyTorch 对简单函数（如 x²、x²y）求梯度  
- [ ] 完成全部自测题  

---

**与前后天的衔接**
- **Day 2**：手算梯度，今天用 PyTorch 自动算。
- **Day 3**：反向传播数学对应今天的 `backward()`。
- **Day 5**：写训练循环时会用到 `loss.backward()` 和 `optimizer.step()`。

*完成 Day 4 后，可以进入 Day 5：PyTorch 基础（下）——nn.Module 与训练循环。*

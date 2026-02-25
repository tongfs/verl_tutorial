# Day 1 详细学习指南：Python 与 NumPy

> 预计时长：2 小时 | 目标：能写简单脚本、会用 NumPy 做矩阵运算

---

## 一、第一部分：Python 基础（约 1 小时）

### 1.1 必学知识点清单

| 知识点 | 说明 | 与 verl 的关系 |
|--------|------|----------------|
| 变量与数据类型 | `int`, `float`, `str`, `bool` | 配置、日志、调试 |
| 列表 `list` | `[]`, 索引、切片、`append` | 数据批次、序列 |
| 字典 `dict` | `{}`, `key: value`, `get()` | 配置、模型参数 |
| 函数 `def` | 定义、参数、返回值 | 封装逻辑、reward 函数 |
| 类 `class` | `__init__`, `self`, 方法 | 理解 PyTorch 的 `nn.Module` |
| 条件与循环 | `if/else`, `for`, `while` | 训练循环、数据处理 |

### 1.2 建议学习顺序

1. **变量与基本类型**（10 分钟）  
   - 赋值、打印、类型检查  
   - 练习：定义几个变量并打印类型

2. **列表**（15 分钟）  
   - 创建、索引、切片、`len()`、`append()`、`for x in list`  
   - 练习：用列表存储 5 个数，遍历并求和

3. **字典**（15 分钟）  
   - 创建、访问、`keys()`、`values()`、`get(key, default)`  
   - 练习：用字典表示一个「模型配置」（如 `{"lr": 0.001, "batch_size": 32}`）

4. **函数**（10 分钟）  
   - `def 函数名(参数):`、`return`  
   - 练习：写一个函数，输入两个数，返回较大者

5. **类**（10 分钟）  
   - `class`、`__init__`、`self`  
   - 练习：定义一个简单的 `Config` 类，有 `lr` 和 `batch_size` 两个属性

### 1.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 在线教程 | 菜鸟教程 Python3 | https://www.runoob.com/python3/python3-tutorial.html | 按章节阅读，重点：1–15 章 |
| 官方文档 | Python 官方教程（中文） | https://docs.python.org/zh-cn/3/tutorial/ | 作为补充查阅 |
| 视频 | 莫烦 Python 基础 | B 站搜索「莫烦 Python」 | 若文字看不懂可看前几集 |

**菜鸟教程建议阅读章节**（按顺序）：
- 第 1 章：Python3 简介
- 第 2 章：Python3 环境搭建
- 第 3 章：Python3 基础语法
- 第 4 章：Python3 基本数据类型
- 第 5 章：Python3 条件控制
- 第 6 章：Python3 循环
- 第 7 章：Python3 函数
- 第 8 章：Python3 数据结构（列表、字典等）
- 第 12 章：Python3 面向对象（类的基础）

---

## 二、第二部分：NumPy 入门（约 1 小时）

### 2.1 必学知识点清单

| 知识点 | 说明 | 与 verl 的关系 |
|--------|------|----------------|
| `np.array()` | 创建数组 | 数据转成张量的基础 |
| `shape` | 数组维度 | 理解 batch、seq_len、hidden_dim |
| `reshape` | 改变形状 | 张量形状变换 |
| 矩阵乘法 | `@` 或 `np.dot` | 神经网络核心运算 |
| 广播 `broadcasting` | 不同形状的运算规则 | 理解 PyTorch 中的广播 |

### 2.2 建议学习顺序

1. **创建数组**（15 分钟）  
   - `np.array([1,2,3])`、`np.zeros()`、`np.ones()`、`np.arange()`  
   - 练习：创建一个 3×4 的零矩阵

2. **索引与切片**（10 分钟）  
   - `a[0]`、`a[1:3]`、`a[:, 1]`（二维）  
   - 练习：从一个 4×5 的数组中取出第 2 行、第 3 列

3. **shape 与 reshape**（10 分钟）  
   - `a.shape`、`a.reshape(2, 6)`、`a.flatten()`  
   - 练习：把 `np.arange(12)` 变成 3×4 的矩阵

4. **矩阵运算**（15 分钟）  
   - 逐元素：`+`、`*`  
   - 矩阵乘法：`A @ B` 或 `np.dot(A, B)`  
   - 练习：两个 2×3 和 3×2 的矩阵相乘，得到 2×2

5. **广播**（10 分钟）  
   - 形状 `(3,4)` 与 `(4,)` 相加  
   - 练习：用广播实现「每行减去该行均值」

### 2.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 官方教程 | NumPy Quickstart | https://numpy.org/doc/stable/user/quickstart.html | 精读，边看边在 Jupyter 里敲 |
| 官方教程 | NumPy 中文文档 | https://www.numpy.org.cn/user/quickstart.html | 若英文吃力可用中文版 |
| 视频 | 李沐 NumPy 入门 | B 站「动手学深度学习」前几集 | 可选，配合动手 |

**NumPy Quickstart 建议阅读部分**：
- The Basics（基础）
- Array creation（数组创建）
- Indexing and Slicing（索引与切片）
- Shape manipulation（形状操作）
- Universal functions（通用函数，了解即可）
- Linear algebra（线性代数，重点看矩阵乘法）

---

## 三、自测题

### 3.1 Python 自测

1. 用字典表示：`{"model": "llama", "hidden_size": 4096}`，并打印 `hidden_size`。
2. 写一个函数 `def sum_list(lst):`，返回列表中所有元素的和。
3. 定义一个类 `DataBatch`，有 `input_ids` 和 `attention_mask` 两个属性，在 `__init__` 中初始化。

### 3.2 NumPy 自测

1. 用 NumPy 实现矩阵乘法：  
   `A = [[1,2],[3,4]]`，`B = [[5,6],[7,8]]`，计算 `A @ B`。
2. 创建一个形状为 `(2, 3, 4)` 的数组，用 `reshape` 变成 `(6, 4)`。
3. 给定 `x = np.array([[1,2,3],[4,5,6]])`，计算每行的和（结果为 `[6, 15]`）。

### 3.3 参考答案（先自己做再对照）

<details>
<summary>点击展开 Python 自测答案</summary>

```python
# 1. 字典
config = {"model": "llama", "hidden_size": 4096}
print(config["hidden_size"])  # 4096

# 2. 求和函数
def sum_list(lst):
    return sum(lst)

# 3. DataBatch 类
class DataBatch:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
```
</details>

<details>
<summary>点击展开 NumPy 自测答案</summary>

```python
import numpy as np

# 1. 矩阵乘法
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = A @ B  # 或 np.dot(A, B)
print(C)   # [[19 22] [43 50]]

# 2. reshape
a = np.arange(24).reshape(2, 3, 4)
b = a.reshape(6, 4)

# 3. 每行求和
x = np.array([[1,2,3],[4,5,6]])
row_sum = x.sum(axis=1)  # [6, 15]
```
</details>

---

## 四、学习时间分配建议

| 时段 | 内容 | 时长 |
|------|------|------|
| 0:00–0:50 | Python 基础（按 1.2 顺序学习） | 50 分钟 |
| 0:50–1:00 | 休息 | 10 分钟 |
| 1:00–1:50 | NumPy 入门（按 2.2 顺序学习） | 50 分钟 |
| 1:50–2:00 | 完成自测题 | 10 分钟 |

---

## 五、今日学习检查清单

- [ ] 能独立写一个包含列表、字典、函数的 Python 脚本
- [ ] 能解释 `list` 和 `dict` 的区别及典型用法
- [ ] 能用 `np.array` 创建数组并做 `reshape`
- [ ] 能用手写或查文档的方式完成 `A @ B` 矩阵乘法
- [ ] 完成全部自测题（可先做再对答案）

---

*完成 Day 1 后，可以进入 Day 2：线性代数与梯度。*

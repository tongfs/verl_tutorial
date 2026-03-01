# Day 6 详细学习指南：深度学习概念补充

> 目标：理解 batch、epoch、过拟合、正则化等概念

---

## 学习时间分配建议

| 时段 | 内容 | 时长 |
|------|------|------|
| 0:00–0:55 | 第一部分：训练概念（batch→epoch→lr→lr_schedule→多 batch 循环） | 55 分钟 |
| 0:55–1:00 | 休息 | 5 分钟 |
| 1:00–1:55 | 第二部分：过拟合与正则化（过拟合→Dropout→L2→其他手段→train/val/test） | 55 分钟 |
| 1:55–2:00 | 自测与检查（重点：batch size 和 lr 的影响） | 5 分钟 |

**预计总时长：2 小时**

---

## 一、第一部分：训练相关概念（约 1 小时）

### 1.1 必学知识点清单

| 知识点 | 说明 | 与 verl / 深度学习的关系 |
|--------|------|---------------------------|
| Batch 批次 | 每次前向/反向用的样本数 | verl 配置中有 train_batch_size、mini_batch_size |
| Epoch 轮次 | 遍历完整训练集一遍 | 训练多少 epoch 是超参数 |
| Learning Rate 学习率 | 梯度下降的步长 | 过大发散、过小收敛慢 |
| Learning Rate Schedule | 训练过程中调整学习率 | 常用：warmup、衰减、cosine |
| 迭代 Iteration | 一次参数更新（一个 batch） | 1 epoch = 数据量/batch_size 次迭代 |

### 1.2 建议学习顺序

1. **Batch 与 Batch Size**（15 分钟）  
   - 全量梯度：用全部数据算梯度 → 准确但慢、占内存  
   - 随机梯度（SGD）：每次用 1 个样本 → 快但噪声大  
   - 小批量（Mini-batch）：每次用 batch_size 个样本，折中  
   - 常见 batch size：32、64、128、256；大模型可能上千  
   - 练习：1000 个样本，batch_size=32，1 个 epoch 有多少次迭代？（1000/32≈31）

2. **Epoch**（10 分钟）  
   - 1 epoch = 把所有训练数据过一遍  
   - 通常训练多个 epoch，直到 loss 收敛或验证集不再提升  
   - 总迭代数 = epoch × (样本数 / batch_size)  
   - 练习：1000 样本、batch_size=64、训练 10 epoch，共多少次参数更新？

3. **Learning Rate 学习率**（15 分钟）  
   - 更新公式：θ ← θ − lr × ∇θ  
   - lr 过大：loss 震荡、甚至发散  
   - lr 过小：收敛慢、可能卡在局部最优  
   - 常用范围：SGD 0.01~0.1，Adam 1e-4~1e-3  
   - 练习：画一张「lr 过大 / 适中 / 过小」时 loss 曲线的示意图（脑中或纸上）

4. **Learning Rate Schedule**（15 分钟）  
   - Warmup：训练初期 lr 从 0 线性升到目标值，稳定训练  
   - 衰减：随 epoch 增加逐渐降低 lr，后期精细调参  
   - Cosine：lr 按余弦曲线从高到低，常用于大模型  
   - PyTorch：`torch.optim.lr_scheduler.StepLR`、`CosineAnnealingLR` 等  
   - 练习：用 `StepLR` 每 5 个 epoch 将 lr 乘 0.5

### 1.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 视频 | 吴恩达深度学习专项 Course 1 | Coursera / B 站搜「吴恩达 深度学习」 | Week 2：Mini-batch、学习率 |
| 文字 | 《动手学深度学习》小批量梯度下降 | https://zh.d2l.ai/chapter_linear-networks/linear-regression-scratch.html | 配合代码 |
| 文档 | PyTorch lr_scheduler | https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate | 查阅 API |

**吴恩达深度学习 Course 1 Week 2 建议观看**：
- Mini-batch 梯度下降
- 理解指数加权平均（可选）
- 学习率衰减（Learning rate decay）

### 1.4 自测与检查

**自测题**  
1. batch_size 从 32 改为 256，对训练速度、收敛稳定性、显存占用各有何影响？  
2. lr 过大和过小分别会导致什么现象？  
3. 5000 样本、batch_size=128、训练 20 epoch，共多少次参数更新？（取整）

<details>
<summary>点击展开参考答案</summary>

1. batch_size 增大：每次迭代算更多样本，梯度更稳定，但单次迭代更慢、显存占用更大  
2. lr 过大：loss 震荡或发散；lr 过小：收敛慢、易卡局部最优  
3. 5000/128≈39 次/epoch，20 epoch → 约 780 次更新
</details>

**检查清单**  
- [ ] 能解释 batch、epoch、iteration 的区别与关系  
- [ ] 能解释 batch size 和 learning rate 对训练的影响  
- [ ] 知道常见的 lr schedule（warmup、衰减、cosine）  

---

## 二、第二部分：过拟合与正则化（约 1 小时）

### 2.1 必学知识点清单

| 知识点 | 说明 | 与 verl / 深度学习的关系 |
|--------|------|---------------------------|
| 过拟合 | 训练集表现好、验证集差，记了噪声而非规律 | 大模型也需防止过拟合 |
| 欠拟合 | 模型太简单，训练集都拟合不好 | 增加容量或训练更久 |
| Dropout | 训练时随机「丢弃」部分神经元，减少共适应 | LLM 中常用，如 0.1 |
| L2 正则（权重衰减） | 在 loss 中加 λ‖W‖²，惩罚大权重 | Adam 的 weight_decay 参数 |
| 验证集 | 不参与训练，用于选模型、早停 | 训练时监控 val loss |

### 2.2 建议学习顺序

1. **过拟合与欠拟合**（15 分钟）  
   - 过拟合：训练 loss 低、验证 loss 高，模型「背题」  
   - 欠拟合：训练和验证 loss 都高，模型能力不足  
   - 直观：模型复杂度 vs 数据量，要匹配  
   - 练习：能画出「欠拟合 / 刚好 / 过拟合」时 train/val loss 随 epoch 的曲线趋势

2. **Dropout**（20 分钟）  
   - 训练时：每个神经元以概率 p 置 0，否则输出除以 (1−p) 保持期望  
   - 推理时：不 dropout，所有神经元参与  
   - PyTorch：`nn.Dropout(p=0.5)`，`model.eval()` 时自动关闭  
   - 作用：打破神经元间的共适应，类似「集成」多子网络  
   - 练习：在 Day 5 的 2 层网络中加 `nn.Dropout(0.2)`，观察 train/eval 行为

3. **L2 正则（权重衰减 Weight Decay）**（15 分钟）  
   - 损失加正则项：L_total = L_task + λ Σ w²  
   - 等价于：每次更新时对权重做衰减，抑制过大参数  
   - PyTorch：`optimizer = Adam(..., weight_decay=0.01)`  
   - 常用：1e-2 ~ 1e-5，大模型常用较小值  
   - 练习：在优化器中加 weight_decay=0.01，对比训练曲线

4. **其他正则手段**（5 分钟）  
   - 早停（Early Stopping）：val loss 不再下降时停止  
   - 数据增强：扩充数据多样性  
   - 简化模型：减少层数、参数  
   - 练习：能说出 2 种防止过拟合的方法

5. **训练/验证/测试**（5 分钟）  
   - 训练集：更新参数  
   - 验证集：调超参、早停、选 checkpoint  
   - 测试集：最终评估，不参与任何选择  
   - 练习：能解释为什么需要单独的验证集

### 2.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 视频 | 李宏毅机器学习 过拟合 | B 站搜「李宏毅 过拟合」 | 直观理解 |
| 文字 | 《动手学深度学习》权重衰减 | https://zh.d2l.ai/chapter_multilayer-perceptrons/weight-decay.html | 配合代码 |
| 文字 | 《动手学深度学习》Dropout | https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html | 同上 |

**李宏毅建议观看**：
- 2023 课程中「过拟合」「正则化」或「深度学习」相关集
- 重点：过拟合现象、Dropout 直观、L2 正则

### 2.4 自测与检查

**自测题**  
1. 若训练准确率 99%、验证准确率 70%，可能是什么问题？如何缓解？  
2. `nn.Dropout(0.3)` 的含义是什么？推理时要不要手动关闭？  
3. 写出在 PyTorch 中给 Adam 优化器加 L2 正则的代码。

<details>
<summary>点击展开参考答案</summary>

1. 过拟合；可加大 Dropout、L2、早停、数据增强或简化模型  
2. 训练时每个神经元 30% 概率置 0；推理时 `model.eval()` 会自动关闭 Dropout  
3. `torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)`
</details>

**检查清单**  
- [ ] 能区分过拟合与欠拟合，并画出 loss 曲线趋势  
- [ ] 能解释 Dropout 的作用及 train/eval 模式下的区别  
- [ ] 能在优化器中设置 weight_decay（L2 正则）  
- [ ] 完成全部自测题  

### 2.5 可选实践：带 Dropout 和 L2 的训练

在 Day 5 的线性回归或 2 层网络上：1）加 `nn.Dropout(0.2)` 在隐藏层后；2）优化器加 `weight_decay=0.01`；3）观察 loss 曲线变化（可与不加时对比）。

---

**与前后天的衔接**
- **Day 5**：单 batch 或全量训练循环，今天理解多 batch、多 epoch 及 lr 的作用。
- **Day 7**：第一周复习会用到 batch、epoch、Dropout 等。
- **verl**：配置中有 train_batch_size、learning_rate 等，理解今天内容后能更好调参。

*完成 Day 6 后，可以进入 Day 7：第一周复习与小结。*

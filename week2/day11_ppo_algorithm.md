# Day 11 详细学习指南：PPO 算法

> 目标：理解 PPO 的核心思想（重要性采样、clip、KL）以及 Actor-Critic、Advantage、GAE，为 Day 12 RLHF 全流程打基础

---

## 学习时间分配建议

| 时段 | 内容 | 时长 |
|------|------|------|
| 0:00–0:55 | 第一部分：PPO 原理（重要性采样→clip→KL） | 55 分钟 |
| 0:55–1:00 | 休息 | 5 分钟 |
| 1:00–1:55 | 第二部分：PPO 在 LLM 中的角色（Actor-Critic→Advantage→GAE） | 55 分钟 |
| 1:55–2:00 | 自测题与检查清单 | 5 分钟 |

**预计总时长：2 小时**

---

## 一、第一部分：PPO 原理（约 1 小时）

### 1.1 必学知识点清单

| 知识点 | 说明 | 与 verl / LLM 的关系 |
|--------|------|----------------------|
| 重要性采样（Importance Sampling） | 用旧策略采样的数据估计新策略的期望；引入比率 $\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}$ | PPO 可对同一批 rollout 数据做多轮更新，提高样本利用率；verl 中 policy 和 ref policy 的 log prob 比即此 |
| PPO-Clip 目标 | 对策略比率做裁剪，限制新策略相对旧策略的变化幅度，避免更新过大导致性能崩溃 | verl 的 PPO 实现使用 clip；clip_ratio 通常 0.1–0.2 |
| KL 惩罚 / KL 散度 | 衡量新旧策略分布的差异；PPO-Penalty 在目标中加 KL 项，PPO-Clip 用 early stopping 基于 KL | RLHF 中防止策略偏离 SFT 太远，保持生成质量；verl 中 ref model 即用于计算 KL |
| On-policy 与 Off-policy | On-policy：只能用当前策略采样的数据；Off-policy：可用旧策略数据 | PPO 本质 on-policy，但通过 clip 允许对同一批数据做少量多轮更新，是「近似 on-policy」 |

---

### 1.2 建议学习顺序

1. **REINFORCE 的问题：为什么需要 PPO**（15 分钟）
   - REINFORCE：采样一条轨迹 → 用回报 $G$ 加权更新 → 数据只用一次；且若某次 $G$ 很大，梯度会很大，策略可能「一步跨太远」导致崩溃
   - 核心矛盾：想多利用数据（多轮更新）vs 策略更新不能太大（否则分布变化，旧数据失效）
   - PPO 的思路：在「多轮更新同一批数据」的同时，用 clip 或 KL 限制每次更新步长
   - 练习：用一句话说明「REINFORCE 为什么不能对同一批数据做多轮更新」？（因为数据是按旧策略采样的，多轮更新后策略变了，数据分布与当前策略不匹配，估计会偏）

2. **重要性采样**（25 分钟）
   - 目标：估计 $\mathbb{E}_{a \sim \pi_\theta}[f(a)]$，但只有 $\pi_{\theta_{\text{old}}}$ 采样的数据
   - 技巧：$\mathbb{E}_{a \sim \pi_\theta}[f(a)] = \mathbb{E}_{a \sim \pi_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)} f(a)\right]$
   - 比率 $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$：新策略下该动作概率 / 旧策略下该动作概率
   - 问题：若 $\theta$ 与 $\theta_{\text{old}}$ 差太多，比率可能极大或极小，方差爆炸
   - 练习：若 $\pi_\theta$ 对某动作概率从 0.1 增到 0.9，比率 $r$ 是多少？（$r = 0.9/0.1 = 9$，很大，梯度会不稳定）

3. **PPO-Clip 目标**（30 分钟）
   - 策略梯度目标可写成：$L^{\text{PG}} = \mathbb{E}[r_t(\theta) \cdot \hat{A}_t]$，其中 $\hat{A}_t$ 是 advantage 估计
   - PPO-Clip：$L^{\text{CLIP}} = \mathbb{E}\left[\min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right)\right]$
   - $\epsilon$ 通常 0.1–0.2；clip 把 $r_t$ 限制在 $[1-\epsilon, 1+\epsilon]$
   - 直观：
     - 若 $\hat{A}_t > 0$（该动作好）：希望增大 $\pi_\theta(a\mid s)$，但 $r_t$ 被 clip 后不会无限增大，更新有上限
     - 若 $\hat{A}_t < 0$（该动作差）：希望减小 $\pi_\theta(a\mid s)$，同样 clip 限制下降幅度
   - 效果：新策略不会离旧策略太远，可以安全地对同一批数据做多轮 SGD
   - 练习：当 $\hat{A}_t > 0$ 且 $r_t = 2$、$\epsilon = 0.2$ 时，clip 后的 $r_t$ 是多少？对目标的影响是什么？（clip 后 $r_t = 1.2$，目标不再随 $r_t$ 增大而增大，起到「截断」作用）

4. **KL 散度与 Early Stopping**（15 分钟）
   - KL 散度 $D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta)$：衡量新旧策略分布差异
   - PPO-Clip 中：不做 KL 约束，但常用 **early stopping**：若当前 batch 上平均 KL 超过阈值（如 0.01），就停止本轮 policy 的梯度更新
   - PPO-Penalty：在目标中加 $-\beta \cdot D_{\text{KL}}$，用 KL 惩罚限制偏离
   - 在 LLM/RLHF 中：ref model（参考模型，通常是 SFT 模型）用于计算 KL，防止策略生成偏离 SFT 太远、导致乱码或退化
   - 练习：RLHF 中为什么要限制策略与 ref model 的 KL？（防止模型为追求高 reward 而输出无意义或有害内容，保持与 SFT 的「接近」）

---

### 1.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 视频 | 李宏毅「Proximal Policy Optimization」 | https://www.youtube.com/watch?v=OAKAZhFmYoI | PPO-Clip 直观、clip 公式 |
| 文档 | Spinning Up — PPO | https://spinningup.openai.com/en/latest/algorithms/ppo.html | Key Equations、clip 直观 |
| 论文 | Schulman et al. «Proximal Policy Optimization Algorithms» | https://arxiv.org/abs/1707.06347 | 完整推导，时间紧可看摘要和算法框 |

---

### 1.4 自测与检查

**自测题**

1. PPO 为什么要做 clip？用一句话说明。
2. 重要性采样中的比率 $r_t(\theta)$ 是什么？若比率过大或过小会有什么问题？
3. PPO-Clip 和 REINFORCE 在「能否对同一批数据多轮更新」上有什么区别？

<details>
<summary>点击展开参考答案</summary>

1. Clip 限制策略更新幅度：当 advantage 为正时，防止新策略过度增大该动作概率；当 advantage 为负时，防止过度减小。这样新策略不会离旧策略太远，可以安全地对同一批 rollout 数据做多轮梯度更新，同时避免性能崩溃。
2. $r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$，即新策略与旧策略在该动作上的概率比。若比率过大或过小，梯度估计方差会很大，甚至导致训练不稳定。
3. REINFORCE 是严格 on-policy，数据用一次就废弃；若对同一批数据多轮更新，策略变化后数据分布与当前策略不匹配，估计会偏。PPO 通过 clip 限制每次更新步长，使新策略不会偏离旧策略太多，因此可以对同一批数据做多轮（如 3–4 轮）SGD 更新，提高样本利用率。
</details>

**检查清单**

- [ ] 能说出 REINFORCE 的问题（数据只用一次、更新可能过大）
- [ ] 能解释重要性采样比率 $r_t$ 的含义
- [ ] 能说明 PPO-Clip 中 clip 的作用（限制更新幅度、允许多轮更新）
- [ ] 能说明 RLHF 中 KL/ref model 的作用（防止偏离 SFT 太远）

---

## 二、第二部分：PPO 在 LLM 中的角色（约 1 小时）

### 2.1 必学知识点清单

| 知识点 | 说明 | 与 verl / LLM 的关系 |
|--------|------|----------------------|
| Actor-Critic | Actor：策略 $\pi_\theta(a\mid s)$；Critic：价值函数 $V_\phi(s)$，用于估计状态价值 | 在 LLM 中：Actor = 语言模型；Critic 基于 LM 的 hidden state 接线性层，估计 advantage |
| Advantage $A_t$ | $A_t = Q(s_t,a_t) - V(s_t)$，用 advantage 替代回报 $G$ 作为梯度权重，可降低方差 | PPO 中用 $\hat{A}_t$ 替代 $G$；方差更小，训练更稳定 |
| GAE（Generalized Advantage Estimation） | $\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$，其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$；$\lambda \approx 0.95$ 平衡偏差与方差 | verl 的 PPO 使用 GAE 估计 advantage；InstructGPT 等 RLHF 实现均采用 |
| RLHF 中的角色映射 | Actor = 要优化的 LM；Ref = SFT 模型（冻结）；Critic = 价值网络；reward 来自 RM | verl 中 policy、ref_policy、critic、reward_model 对应这些角色 |

---

### 2.2 建议学习顺序

1. **为什么用 Advantage 替代回报 $G$**（15 分钟）
   - REINFORCE 用 $G$ 加权：$\nabla J \propto \mathbb{E}[G \cdot \nabla \log \pi(a\mid s)]$，但 $G$ 方差大（整条轨迹只用一个数）
   - Advantage $A_t = Q(s_t,a_t) - V(s_t)$：减去 baseline $V(s_t)$ 可降低方差，且不改变梯度无偏性
   - 直观：$A_t > 0$ 表示该动作比「平均」好，应增加概率；$A_t < 0$ 表示比平均差，应减少
   - 练习：为什么减去 $V(s_t)$ 不会改变策略梯度的无偏性？（因为 $\mathbb{E}[V(s_t) \nabla \log \pi(a\mid s)] = 0$，即 baseline 与梯度正交）

2. **Actor-Critic 结构**（15 分钟）
   - Actor：输出动作概率分布 $\pi_\theta(a\mid s)$，即策略；在 LLM 中为语言模型的 logits
   - Critic：输出 $V_\phi(s)$，即从状态 $s$ 出发的期望回报；在 LLM 中常共享 LM 的 backbone，最后一层 hidden state 接线性层
   - 训练：Actor 用 PPO 目标更新；Critic 用 MSE 拟合 $V(s_t) \approx G_t$ 或 TD target
   - 练习：在 LLM 的 PPO 中，Actor 和 Critic 分别对应什么？（Actor = 语言模型本身；Critic = 价值网络，输入 prompt+response 的 hidden state，输出标量）

3. **GAE 公式与直观**（20 分钟）
   - TD 残差：$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，表示「实际一步回报与预期」的差
   - GAE：$\hat{A}_t^{\text{GAE}} = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$
   - $\lambda = 0$：只用一步 TD，低方差高偏差；$\lambda = 1$：等价于 $G_t - V(s_t)$，高方差低偏差；$\lambda \approx 0.95$ 折中
   - 在 LLM 中：常对整段 response 只给一个 reward（在序列末尾），中间步 $r_t = 0$，只有最后一步有 $r$
   - 练习：若整段生成只在结束时给 reward $R$，且 $\gamma=1$，则 $G_t$ 对每个 $t$ 都等于多少？$\delta_t$ 在中间步和最后一步分别如何？（$G_t = R$ 对所有 $t$；最后一步 $\delta = R - V(s_{T-1})$，中间步 $\delta = \gamma V(s_{t+1}) - V(s_t)$）

4. **RLHF / InstructGPT 中的 PPO 配置**（15 分钟）
   - Actor：要优化的语言模型（policy）
   - Ref：冻结的 SFT 模型，用于计算 KL 和 importance sampling 的旧策略
   - Critic：价值网络，估计 advantage
   - Reward：由 Reward Model 对 `(prompt, response)` 打分
   - 总目标：最大化 reward，约束 KL(ref)，用 PPO-Clip 限制策略更新
   - 练习：RLHF 中 ref model 有哪两个作用？（① 计算 KL 散度，限制策略偏离 SFT；② 作为 importance sampling 的 $\pi_{\text{old}}$，计算比率 $r_t$）

---

### 2.3 参考资料

| 类型 | 资源 | 链接/获取方式 | 建议用法 |
|------|------|---------------|----------|
| 论文 | InstructGPT（RLHF 部分） | https://arxiv.org/abs/2203.02155 | 3.4 节 PPO 配置、advantage、KL |
| 文档 | Spinning Up — GAE | https://spinningup.openai.com/en/latest/algorithms/ppo.html | GAE 公式与 $\lambda$ 的直观 |
| 博客 | Lil'Log — GAE | 搜索「Generalized Advantage Estimation」 | GAE 推导与偏差-方差权衡 |

---

### 2.4 自测与检查

**自测题**

1. Advantage $A_t$ 的定义是什么？为什么用 $A_t$ 替代 $G$ 可以降低方差？
2. GAE 中 $\lambda$ 的作用是什么？$\lambda=0$ 和 $\lambda=1$ 分别对应什么？
3. 在 RLHF 的 PPO 中，Actor、Ref、Critic、Reward 分别对应什么？

<details>
<summary>点击展开参考答案</summary>

1. $A_t = Q(s_t,a_t) - V(s_t)$，表示该动作相对于「平均」有多好。减去 baseline $V(s_t)$ 不改变梯度无偏性，但 $A_t$ 的方差通常比 $G_t$ 小，因为 $V(s_t)$ 吸收了部分随机性。
2. $\lambda$ 控制 GAE 中多步 TD 的权重：$\lambda=0$ 只用一步 $\delta_t$，低方差高偏差；$\lambda=1$ 等价于 $G_t - V(s_t)$，高方差低偏差。$\lambda \approx 0.95$ 是常用折中。
3. Actor = 要优化的语言模型（策略）；Ref = 冻结的 SFT 模型，用于 KL 和 importance sampling；Critic = 价值网络，估计 advantage；Reward = Reward Model 对 (prompt, response) 的打分。
</details>

**检查清单**

- [ ] 能解释 Advantage 的含义及为何能降低方差
- [ ] 能说明 Actor-Critic 在 LLM 中的对应关系
- [ ] 能写出 GAE 公式并说明 $\lambda$ 的作用
- [ ] 能说出 RLHF 中 Actor、Ref、Critic、Reward 的角色

---

### 2.5 可选实践：理解 PPO 的完整目标

不要求写代码，可在纸笔或注释中梳理：

- PPO 的 policy loss 通常为：$L^{\text{CLIP}} + c_1 L^{\text{VF}} - c_2 \cdot \text{KL}$
- $L^{\text{CLIP}}$：clip 后的策略梯度目标
- $L^{\text{VF}}$：Critic 的 value loss，如 $(V(s_t) - G_t)^2$
- $\text{KL}$：与 ref 的 KL 散度（可选，或通过 early stopping 实现）
- 与 Day 10 结合：$L^{\text{CLIP}}$ 中的 $\hat{A}_t$ 来自 GAE，替代了 REINFORCE 中的 $G$。

---

**与前后天的衔接**

- **Day 10**：策略梯度、REINFORCE；Day 11 在此基础上引入重要性采样、clip、KL、Actor-Critic、Advantage。
- **Day 12**：RLHF 全流程；Day 11 的 PPO 是 RLHF 第三阶段的核心实现。
- **verl**：PPO 是 verl 的核心算法；理解 clip、ref model、advantage 后，第三周看 verl 源码会更容易定位。

*完成 Day 11 后，可以进入 Day 12：RLHF 全流程。*

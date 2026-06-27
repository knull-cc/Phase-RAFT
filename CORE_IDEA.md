# Phase-aligned IdeaBlock Retrieval 核心 Idea

## 一句话版本

**长序列预测不只应该看最近的连续窗口，还应该从历史中检索“处在相同周期相位、局部形态相似”的片段，并直接借用这些历史片段之后真实发生过的未来趋势来辅助预测。**

## 核心问题

多数长序列预测模型把输入窗口看成一段连续历史，然后用参数化模型学习从过去到未来的映射。这类方法能捕捉局部趋势，但对强周期序列有一个天然浪费：很多真实可复用的信息并不在最近几个时间点，而在历史上相同相位附近的周期片段里。

例如电力、交通、天气、Solar、PEMS 等数据通常存在日周期、周周期或采样频率诱导的固定周期。当前时刻如果处在某个周期相位，那么历史上相同相位附近的局部模式，往往比任意连续窗口中的远距离点更有预测价值。

因此，本项目的核心假设是：

> 如果两个历史窗口在相同周期相位附近呈现相似的局部形态，那么它们后续的未来残差趋势也可能相似。

## 基本想法

PIBR 将预测任务拆成两部分：

1. 用简单的 lookback prediction head 从当前输入窗口预测一个基础未来趋势。
2. 从训练集中检索与当前窗口相位局部块相似的历史样本，聚合这些历史样本真实发生过的未来残差趋势，再与基础预测融合。

这里最关键的不是普通 nearest-neighbor retrieval，而是 **phase-aligned retrieval**：检索的 key 不是完整输入窗口，也不是随机 patch，而是围绕最后观测点相位构造的相位对齐局部块。

## IdeaBlock 定义

给定周期长度 `P`，当前输入窗口的最后一个观测点有一个中心相位：

```text
p = (index_abs + seq_len - 1) mod P
```

给定相位半径 `r`，PIBR 选择相位邻域：

```text
[p-r, ..., p, ..., p+r]
```

然后向前回看若干个周期 `C`，抽取这些相位位置上的观测值，组成一个 **Phase-aligned IdeaBlock**：

```text
IdeaBlock(p, r, C)
  = observations at phases [p-r, p+r]
    over previous C cycles
```

默认设置中：

```text
r = 1
C = 4
```

也就是每个 key 尝试抽取最近 4 个周期内、中心相位及左右相邻相位的局部观测，共 `4 * 3 = 12` 个相位槽位。缺失槽位通过 mask 标记，观测块和 mask 一起构成检索 key。

## Key-Value 记忆库

训练前，模型遍历训练集样本，构建一个非参数记忆库：

```text
Key   = normalized Phase-aligned IdeaBlock
Value = future - last_observed
```

这里 `Value` 不是未来原值，而是相对最后观测值的未来残差趋势。这样做的直觉是：不同样本的绝对数值水平可能不同，但“接下来怎么走”的形态更容易在 offset-normalized 空间中复用。

## 预测流程

预测时，当前输入窗口也被转换成同样形式的 query key：

```text
query = IdeaBlock(current window)
```

然后与训练集所有 keys 计算余弦相似度：

```text
sim(query, key_i) = query @ key_i
```

取 top-k 个最相似历史样本，用 temperature softmax 得到权重：

```text
weights = softmax(top_sim / temperature)
```

再对这些历史样本对应的真实未来残差做加权求和：

```text
retrieved_trend = sum_i weights_i * value_i
```

最后，模型把两个未来趋势拼接后融合：

```text
lookback_trend  = Linear(current_window - last_observed)
retrieved_trend = Retrieval(query)
future_trend    = Linear([lookback_trend, retrieved_trend])
prediction      = future_trend + last_observed
```

## 为什么这个 Idea 有意义

PIBR 的核心价值在于把周期结构显式变成了检索索引，而不是完全交给神经网络自己学习。

它利用了三类信息：

1. **相位信息**：同一周期相位往往对应相似的外部状态，例如一天中的同一小时、一周中的同一天。
2. **局部形态信息**：只看相位还不够，PIBR 还比较相位邻域内的实际观测走势。
3. **历史真实未来**：检索到相似历史片段后，不是只用其 embedding，而是直接使用其真实后续残差作为预测依据。

这使得 PIBR 更接近一种“基于历史类比的预测”：当前处境像历史上的哪些处境，历史上这些处境后来怎么发展，现在就把这些后续发展作为强先验。

## 与普通检索预测的区别

普通检索增强预测可能直接用完整窗口、随机 patch 或 learned representation 做相似度匹配。PIBR 的差异在于：

- 检索单位是按周期相位重组后的 IdeaBlock，而不是原始连续窗口。
- Key 中显式包含相位邻域和跨周期槽位。
- Value 是未来残差趋势，而不是标签类别、embedding 或单步目标。
- 训练时会屏蔽时间重叠样本，降低直接检索自身窗口造成的数据泄漏。

因此，PIBR 的 novelty 不在于“用了 retrieval”本身，而在于 **用周期相位定义可检索的局部历史状态，并把检索结果作为未来残差先验注入预测头**。

## 可写成论文贡献点的版本

1. **Phase-aligned IdeaBlock representation**  
   提出一种面向周期时间序列的相位对齐局部块表示，将当前预测点附近的周期相位及其跨周期历史观测组织成可比较的检索 key。

2. **Future residual Key-Value memory**  
   构建训练集级别的 Key-Value 记忆库，其中 key 表示相位局部状态，value 存储该状态之后真实发生的 offset-normalized future residual，从而把历史未来趋势作为非参数预测先验。

3. **Retrieval-fused forecasting head**  
   将线性 lookback 预测趋势与检索得到的历史未来趋势融合，使模型同时利用当前窗口的参数化外推能力和历史相似状态的非参数类比能力。

4. **Overlap-aware training retrieval**  
   训练阶段屏蔽与当前样本时间区间过近的 memory entries，减少重叠窗口导致的泄漏式检索。

## 适合放在摘要里的版本

本文提出 Phase-aligned IdeaBlock Retrieval（PIBR），一种面向长序列预测的相位对齐检索增强方法。PIBR 认为，在强周期时间序列中，与当前预测点处于相同周期相位且局部形态相似的历史片段，其后续真实趋势可为当前预测提供直接先验。具体地，PIBR 将输入窗口转换为围绕最后观测相位的跨周期局部块，并以该块作为 key 构建训练集 Key-Value 记忆库，其中 value 为对应样本的未来残差趋势。预测时，模型检索相似历史 IdeaBlocks，聚合其未来残差，并与当前窗口的线性预测趋势融合。该设计显式利用周期相位结构，将长序列预测从纯参数化外推扩展为参数化预测与历史类比检索的结合。

## 适合放在 Introduction 里的动机版本

长序列预测中的许多 benchmark 数据具有强周期性，例如电力负载、交通流量、天气与太阳能数据。对于这类序列，预测未来并不只依赖最近的连续变化，还依赖当前时间点处于周期中的什么位置，以及历史上相同相位附近曾出现过怎样的局部状态。现有参数化模型通常把输入窗口作为连续序列建模，希望模型自行学习周期对齐关系；但这种方式没有直接利用一个简单事实：如果当前相位局部模式与历史上某些周期片段相似，那么这些历史片段之后真实发生的趋势本身就是有价值的预测证据。

基于这一观察，PIBR 将周期相位作为检索坐标，构造 Phase-aligned IdeaBlock 来描述当前预测点附近的跨周期局部状态，并从训练集中检索相似状态对应的未来残差。相比仅从当前窗口外推，PIBR 显式引入了历史相似状态的真实未来作为非参数先验；相比普通检索方法，PIBR 的 key 由周期相位结构定义，因而更贴合周期时间序列的预测机制。

## 毒舌版判断

这个 idea 真正能打的地方不是模型复杂，而是它把一个常识变成了可执行机制：周期序列里，“同一相位附近以前怎么走”经常比堆更深的网络更直接。它的风险也很明确：如果周期假设不准、相位错位严重，或者训练集中没有足够相似的历史状态，检索到的 future residual 就会从先验变成噪声。换句话说，PIBR 是一个很有解释力的 retrieval bias，但不是一个能自动拯救所有非周期数据的万能预测器。

## 最简公式化描述

给定输入窗口 `X_t`、最后观测值 `x_t`、周期长度 `P`：

```text
q_t = normalize(IdeaBlock(X_t, phase(t), r, C))
M = {(k_i, v_i)} where
  k_i = normalize(IdeaBlock(X_i, phase(i), r, C))
  v_i = Y_i - x_i
```

检索：

```text
N_t = TopK_i(q_t · k_i)
w_i = softmax((q_t · k_i) / tau)
R_t = sum_{i in N_t} w_i v_i
```

融合预测：

```text
B_t = Linear(X_t - x_t)
Y_hat_t = Linear(concat(B_t, R_t)) + x_t
```

其中 `B_t` 是基础 lookback trend，`R_t` 是检索得到的 historical future residual trend。

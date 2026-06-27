# Phase-RAFT 项目代码结构报告

## 1. 项目定位

本项目实现的是一个长序列时间序列预测方法：**Phase-aligned IdeaBlock Retrieval（PIBR）**。项目当前不是一个通用模型库，而是围绕单一研究想法组织的实验代码：先从训练集构建相位对齐的 Key-Value 记忆库，再在预测时检索历史上相似的相位局部块，将检索得到的未来残差趋势与线性预测头融合。

从代码结构看，项目继承了常见 long-term forecasting benchmark 的组织方式，但已经将可选模型收敛到 `PIBR` 一个模型。核心贡献集中在 `models/PIBR.py` 和 `layers/Retrieval.py`，其他目录主要承担数据加载、实验训练、指标计算、脚本批量运行和兼容性工具。

## 2. 顶层目录结构

```text
Phase-RAFT/
├── run.py                         # 命令行入口，解析参数并启动训练/测试
├── run_main.sh                    # 默认批量运行脚本，目前只启用 ETTh1
├── README.md                      # 方法概述与基础使用说明
├── requirements.txt               # Python 依赖
├── models/
│   └── PIBR.py                    # PIBR 预测模型与检索融合逻辑
├── layers/
│   └── Retrieval.py               # Phase-aligned IdeaBlock 检索器
├── exp/
│   ├── exp_basic.py               # 实验基类、设备选择、模型字典
│   └── exp_long_term_forecasting.py # 长期预测训练、验证、测试流程
├── data_provider/
│   ├── data_factory.py            # 数据集类选择与 DataLoader 构造
│   └── data_loader.py             # ETT、Custom、Solar、PEMS 数据集实现
├── scripts/
│   ├── etth1.sh ... pems08.sh     # 各数据集实验脚本
└── utils/
    ├── metrics.py                 # MAE/MSE/RMSE/MAPE/MSPE
    ├── tools.py                   # 学习率调整、可视化、EarlyStopping 等
    ├── timefeatures.py            # 时间特征编码
    ├── augmentation.py            # 时间序列增强方法
    ├── dtw.py / dtw_metric.py     # DTW 相关实现
    └── losses.py / m4_summary.py / masking.py / ADFtest.py
```

代码量大致分布为：入口 `run.py` 约 255 行，实验流程约 336 行，数据加载约 508 行，核心模型和检索层约 295 行，工具函数约 1.5k 行。实际研究逻辑主要集中在不到 300 行的模型与检索代码中。

## 3. 运行主链路

项目主链路如下：

```text
run.py
  -> Exp_Long_Term_Forecast(args)
    -> Exp_Basic._build_model()
      -> data_provider(..., flag=train/val/test)
      -> PIBR.Model(args)
      -> model.prepare_dataset(train_data, valid_data, test_data)
        -> PhaseAlignedIdeaBlockRetrieval.prepare_dataset(train_data)
    -> exp.train(setting)
      -> train loop
      -> vali(...)
      -> 保存验证集最优 checkpoint
    -> exp.test(setting)
      -> 生成 metrics.npy / pred.npy / true.npy
      -> 写入 result_long_term_forecast.txt
```

`run.py` 负责做三件关键事情：

1. 解析实验、数据、模型、优化器、GPU、增强等参数。
2. 根据数据集名、文件名或频率推断周期长度 `period_len`。
3. 固定 PIBR 的检索超参数：`idea_block_radius=1`、`idea_block_cycles=4`、`topm=20`、`temperature=0.1`。

当前 `task_name` 虽然保留了参数形式，但实际只支持 `long_term_forecast`。`-Phase` / `--phase` 是兼容性开关，开启后仍然强制使用 `PIBR`。

## 4. 核心模型结构

`models/PIBR.py` 中的 `Model` 是完整预测器，结构很简洁：

```text
输入 x: [batch, seq_len, channels]
  -> offset normalization: x - last_observed
  -> linear_x: lookback trend prediction
  -> retriever.retrieve(...): historical future residual retrieval
  -> concat([lookback_trend, retrieved_trend], time dimension)
  -> linear_pred: fused future trend
  -> add last_observed offset
输出: [batch, pred_len, channels]
```

模型使用两个线性层：

- `linear_x`: 将归一化后的历史窗口从 `seq_len` 投影到 `pred_len`，形成基础 lookback 预测。
- `linear_pred`: 将基础预测趋势与检索趋势拼接后，从 `2 * pred_len` 压回 `pred_len`。

检索器 `PhaseAlignedIdeaBlockRetrieval` 在模型初始化时创建，但记忆库不是参数化层，而是在 `prepare_dataset(train_data)` 阶段从训练集样本预先构造出来。训练期间检索值参与前向计算，但训练参数主要来自两个线性层。

## 5. Phase-aligned IdeaBlock 检索机制

`layers/Retrieval.py` 是项目最关键的文件。它实现了一个非参数 Key-Value 检索器。

### 5.1 记忆库构造

`prepare_dataset(train_data)` 遍历训练集每个样本：

- Key：调用 `make_keys()`，从输入窗口中抽取围绕最后观测点相位的历史局部块。
- Value：取真实未来窗口的残差，即 `future - last_observed`。
- Index：记录样本在原始序列中的绝对起点，用于训练时屏蔽自重叠样本。

最终得到：

```text
keys:   [n_train, key_dim]
values: [n_train, pred_len, channels]
train_abs_indices: [n_train]
```

### 5.2 Key 构造

`make_keys(x, index_abs)` 的逻辑是：

1. 计算最后观测点的绝对相位：
   `center_phase = (index_abs + seq_len - 1) % period_len`
2. 以 `phase_radius` 为半径，取相邻相位窗口。
3. 向前回看 `num_cycles` 个周期，在输入窗口内抽取对应相位位置。
4. 对缺失位置加 mask，并将观测块和 mask 拼接。
5. 对 key 做均值中心化和 L2 normalize。

默认配置下，`phase_radius=1`、`num_cycles=4`，所以每个 key 会尝试使用 `4 * 3 = 12` 个相位槽位；实际 key 还包含同长度的 mask 信息。

### 5.3 检索与融合

`retrieve()` 会计算 query key 与训练 keys 的余弦相似度：

```text
sim = query @ keys.T
top_sim, top_idx = topk(sim)
weights = softmax(top_sim / temperature)
retrieved_trend = weighted_sum(values[top_idx])
```

训练模式下会调用 `_mask_self_overlap()`，屏蔽与当前样本时间区间过近的训练样本，避免模型直接检索到自身或高度重叠窗口。屏蔽范围是：

```text
abs(train_index - query_index) < seq_len + pred_len
```

如果一整行全被屏蔽，代码会回退到未屏蔽相似度，避免 `topk` 面对全 `-inf`。

## 6. 数据层设计

`data_provider/data_factory.py` 根据 `args.data` 选择数据集类：

- `ETTh1`, `ETTh2` -> `Dataset_ETT_hour`
- `ETTm1`, `ETTm2` -> `Dataset_ETT_minute`
- `custom` -> `Dataset_Custom`
- `PEMS` -> `Dataset_PEMS`
- `Solar` -> `Dataset_Solar`

每个数据集 `__getitem__()` 都返回统一五元组：

```python
index, seq_x, seq_y, seq_x_mark, seq_y_mark
```

其中 `index` 对 PIBR 很重要，因为检索器需要通过它还原样本的绝对时间位置，进一步计算相位和训练自重叠屏蔽。

数据切分规则如下：

- ETT hourly/minute：固定按 benchmark 时间段切分。
- Custom/Solar：`70% train / 10% val / 20% test`。
- PEMS：`60% train / 20% val / 20% test`。

所有主要数据集默认使用 `StandardScaler`，只在训练区间拟合 scaler，然后应用到全量数据。

## 7. 实验流程

`exp/exp_long_term_forecasting.py` 承担训练、验证和测试：

- `_build_model()` 会先构造 train/val/test 数据集，再调用 `model.prepare_dataset()` 构建检索记忆库。
- `train()` 使用 Adam 和 MSELoss。
- 虽然参数中保留 `patience`，但 EarlyStopping 被注释掉；训练会完整跑完 `train_epochs`。
- 每个 epoch 后同时计算验证集和测试集 loss。
- 最终保存验证集 loss 最低的模型状态到 `checkpoints/<setting>/checkpoint.pth`。
- `test()` 输出 `pred.npy`、`true.npy`、`metrics.npy`，并将指标追加写入 `result_long_term_forecast.txt`。

需要注意，验证集 loss 计算时把 `outputs` 和 `batch_y` 都 detach 到 CPU 后再传入 criterion，因此验证不影响梯度；这在功能上没问题，但会带来额外 CPU 拷贝。

## 8. 脚本与实验配置

`scripts/` 下包含常见长序列预测数据集脚本：

- ETT: `etth1.sh`, `etth2.sh`, `ettm1.sh`, `ettm2.sh`
- Traffic/Electricity/Weather/Solar
- PEMS03/04/07/08

脚本共同特征：

- 固定 `model_name=PIBR`。
- 大多数长序列数据集跑 `pred_len in 96 192 336 720`。
- PEMS 跑 `pred_len in 12 24 48 96`。
- 随机种子当前固定为 `2024`。
- 通过 `extra_args="$@"` 支持从命令行追加参数。

顶层 `run_main.sh` 当前只启用 `scripts/etth1.sh`，其他数据集被注释。这个设计适合快速 smoke run，但如果用于论文实验，需要显式打开全部脚本或写一个统一调度脚本。

## 9. 工具模块

`utils/` 中有不少继承式或通用 benchmark 工具：

- `metrics.py`: 常规预测指标。
- `tools.py`: 学习率调整、可视化、EarlyStopping、dotdict、异常检测 adjustment。
- `timefeatures.py`: 根据 pandas offset 生成周期时间特征。
- `augmentation.py`: jitter、scaling、permutation、warp、WDBA、guided warp 等增强。
- `dtw.py`, `dtw_metric.py`: DTW 计算。
- `losses.py`, `m4_summary.py`: M4/短序列预测相关损失和汇总。
- `masking.py`: attention mask，当前 PIBR 主链路基本不用。

这些工具说明项目来源或框架风格更接近通用时间序列实验仓库；PIBR 主线只用到了其中一部分。

## 10. 代码结构优点

1. **研究核心位置清楚**：PIBR 的模型融合与检索机制分别在 `models/PIBR.py` 和 `layers/Retrieval.py`，没有散落在训练循环里。
2. **数据接口统一**：所有数据集都返回 `index, seq_x, seq_y, seq_x_mark, seq_y_mark`，使 PIBR 能在不同数据集上复用相位检索逻辑。
3. **检索记忆库显式构建**：`prepare_dataset()` 把 key/value 构建过程放在训练前，便于调试、统计记忆库规模和检查相位配置。
4. **训练自重叠屏蔽合理**：训练检索时考虑了样本窗口重叠问题，避免最直接的数据泄漏。
5. **脚本覆盖主流 benchmark**：ETT、PEMS、Traffic、Electricity、Weather、Solar 都有对应脚本，实验入口比较完整。

## 11. 主要风险与改进点

1. **模型字典只保留 PIBR，但训练框架仍保留大量多模型分支**  
   `Exp_Long_Term_Forecast` 中仍有非检索模型的 decoder input、AMP 分支和接口兼容逻辑。当前不影响运行，但会让读者误以为项目支持 Transformer 类 encoder-decoder 模型。建议要么清理为 PIBR 专用框架，要么补全多模型接口。

2. **`patience` 参数实际未生效**  
   `EarlyStopping` 被注释，训练总是跑满 `train_epochs`。脚本里仍传入 `--patience 5`，这会造成实验配置和实际行为不一致。建议 README 或脚本明确说明，或者恢复 early stopping。

3. **检索记忆库按样本逐个构建，可能成为大数据集瓶颈**  
   `prepare_dataset()` 逐样本调用 `train_data[i]` 和 `make_keys()`，对 Traffic、PEMS07 等高维或大样本数据可能启动较慢。后续可考虑批量 key 构建、缓存 keys/values，或将 memory 保存到磁盘。

4. **周期长度推断是硬编码规则**  
   `period_len` 在 `run.py` 中根据数据集名、文件名或频率推断。这个策略对标准数据集够用，但对 custom 数据集可能错误。建议开放显式 `--period_len` 参数，并让自动推断只作为默认值。

5. **分类分支不可用**  
   `PIBR.Model.classification()` 调用了 `self.projection`，但模型没有定义该层。虽然当前 `run.py` 只支持 long-term forecast，但保留不可用分支会增加维护风险。

6. **验证阶段每个 epoch 都计算 test loss**  
   训练过程中持续查看测试集 loss 在论文实验中容易引发 protocol 争议。建议将 epoch 内 test loss 改为可选 debug 行为，正式实验只用 validation 选择 checkpoint，最后测试一次。

7. **缺少自动化测试或最小 smoke test**  
   当前没有测试目录。对于 PIBR 这种依赖 index、border、period、mask 的方法，建议至少加入 `make_keys()` shape/phase/mask 单测，以及 `_mask_self_overlap()` 的边界测试。

## 12. 建议的后续整理方向

短期建议：

- 在 README 中补充完整运行链路、输出文件位置和 `period_len` 推断规则。
- 恢复或移除 EarlyStopping，避免参数误导。
- 增加 `--period_len` 显式参数。
- 删除或标注当前不支持的 classification/imputation/anomaly 分支。
- 为 `PhaseAlignedIdeaBlockRetrieval` 加 3-5 个单元测试。

中期建议：

- 支持 memory cache，避免每次实验都重新构造训练集检索库。
- 给 `prepare_dataset()` 增加批处理构建逻辑。
- 把 `scripts/` 的重复参数抽成统一实验配置表，减少手工修改错误。
- 在结果目录中保存完整 args JSON，方便复现实验。

长期建议：

- 将 PIBR 方法代码与 benchmark 兼容代码进一步解耦。
- 增加可解释性输出，例如 top-k 检索样本索引、相似度、相位块描述和 retrieved trend 可视化。
- 如果目标是投稿 artifact，建议补充一键复现实验脚本、环境锁定文件和结果汇总脚本。

## 13. 总结判断

这个项目的代码结构已经足够支撑一个清晰的研究原型：核心方法集中、实验入口完整、数据接口统一，PIBR 的相位对齐检索逻辑也比较直接可读。它目前最大的结构问题不是“代码混乱”，而是“通用 benchmark 外壳和单一 PIBR 方法之间还没有完全收拢”：一些历史兼容分支、未生效参数和未使用工具会降低报告/开源时的可信度。

如果目标是内部实验，当前结构可以继续迭代；如果目标是论文开源或 artifact evaluation，建议优先处理 early stopping、测试集使用协议、显式周期参数、检索缓存和单元测试这几个问题。

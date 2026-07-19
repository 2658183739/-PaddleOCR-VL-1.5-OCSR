# V3 消融实验协议

## 1. 目的

消融不是“多跑几个版本取最高分”，而是回答每个组件是否产生独立收益，以及组件之间是否存在交互。

第一阶段只研究两个数据因素：

- `W`：是否加入论文分组后的 strict MolRecBench train。
- `A`：是否加入一档受控视觉退化。

采用完整 `2×2` 因子设计：

| Run | W | A | 数据文件 |
| --- | ---: | ---: | --- |
| 00 | 0 | 0 | `train_v3_a_control.jsonl` |
| 10 | 1 | 0 | `train_v3_d_wild_only.jsonl` |
| 01 | 0 | 1 | `train_v3_e_aug_only.jsonl` |
| 11 | 1 | 1 | `train_v3_b_recommended.jsonl` |

每个组合运行 seed `20260717` 和 `20260718`。seed 是 block，不是另一项被随意改变的超参。

## 2. 固定条件

四组必须固定：

- `model_name_or_path=V3/models/v2_1_export`
- `max_steps=250`
- effective batch = 32
- learning rate = `2e-5`
- optimizer、scheduler、warmup、LoRA rank
- development labels、prompt、像素范围和 decoding
- Paddle/PaddleFormers/驱动版本

除数据文件与 seed 外，不允许改变其它变量。任何 OOM 导致的 batch 或像素调整必须对四组全部重跑。

## 3. 运行顺序

本轮实际执行顺序为：

```text
seed 20260717: 11 -> 00 -> 10 -> 01
seed 20260718: 11 -> 10 -> 00 -> 01
```

每个 run 记录 GPU、驱动、开始时间、结束时间、samples/s、峰值显存和失败重试。

这个顺序只交换了 00/10 的中间位置，并不是完全随机或位置平衡：11 在两个 seed 中都最先运行，01 都最后运行。因此机器热状态、缓存或时间漂移仍可能与 11/01 混杂。本轮结果应定位为工程探索证据，不能把小差异写成严格因果结论。

下一轮建议至少使用 4 个独立 seed，并先生成平衡 Latin-square 顺序，再随机把顺序分配给 seed。例如：

```text
00 -> 10 -> 01 -> 11
10 -> 01 -> 11 -> 00
01 -> 11 -> 00 -> 10
11 -> 00 -> 10 -> 01
```

这样每个条件在四个运行位置各出现一次。若预算只允许两个 seed，至少让第二个 seed 使用第一个顺序的逆序，并把结论降级为 exploratory。

## 4. 响应变量

主响应只来自 development：

```text
dev_score = 0.5 * legacy_core_exact + 0.5 * legacy_region_exact
```

同时设置硬闸门：

- valid SMILES 相比 00 下降不得超过 0.5pp；
- printed subgroup 下降不得超过 0.5pp；
- weak-layout/real subgroup 的净新增正确数不少于回归数。

locked wild/private 不参与消融选模。

## 5. 分析

每个 seed 内计算：

```text
W 主效应 = ((score10 - score00) + (score11 - score01)) / 2
A 主效应 = ((score01 - score00) + (score11 - score10)) / 2
W×A 交互 = score11 - score10 - score01 + score00
```

再报告两 seed 的均值和范围。设计上可以拟合：

```text
dev_score ~ W + A + W:A + seed_block
```

但本轮只有两个训练 seed，残差自由度和方差估计不足，不报告 ANOVA p 值，也不做显著性声明。主结果只报告四条件均值、seed range、主效应、交互效应和 validity 门槛；小于约 0.5-1.0pp 且不超过 seed 波动的差异写成“无明确差异”。样本级 paired bootstrap 用于同一 checkpoint/模型在配对样本上的预测差异，run-level seed 差异用于判断训练稳定性，两者不能互相替代。

## 6. 剂量响应与 warm-start

如果 `A` 主效应为正，再运行 C：两档 augmentation。它回答“更强 augmentation 是否继续受益”，不参与第一轮 W/A 主效应估计。

`probe_base15_recommended_a100.yaml` 只把基座换成 PaddleOCR-VL-1.5，衡量固定 250-step 预算下的 warm-start 效率。不能据此宣称原始 1.5 经过充分调参后必然更差。

## 7. 后训练消融

后训练采用机制分解，不与 SFT 数据消融混跑：

| 阶段 | 候选池 | 选择器 | 目的 |
| --- | --- | --- | --- |
| P0 | greedy single | 无 | 主模型基线 |
| P1 | fixed multi-candidate | heuristic | 测候选召回 |
| P2 | 与 P1 完全相同 | reward head | 测选择器收益 |
| P3 | targeted crop 扩展 | 与 P2 相同 | 测 weak-layout crop 收益 |
| P4 | P3 | P2 + hard replay model | 测低 LR hard replay |

P1 先报告 oracle 与 P0 的差异。P2 必须在同一候选池上比较；否则候选变化和 selector 变化无法区分。reward head 的 margin、route 和 crop 开关只能在 development 或 cross-validation 中选择。

## 8. 停止规则

- 00/10/01/11 未跑完两个 seed，不宣称某配比最优。
- locked test 一旦解锁，不能再返回搜索超参。
- hard replay 导致 core 或 validity 过闸失败，立即回退。
- 小于 bootstrap/seed 波动的提升写成“无明确差异”，不是“涨分”。

# 2026-06-28 弱布局候选选择强化实验记录

## 当前结论

稳健最好结果仍采用 v4 单 seed candidate-choice reward head + main reward strict CV route：

- v4 候选池 oracle：`0.412946`
- v4 choice 单独：`0.322173`
- v4 full-label route：`0.373512`
- v4 strict CV route：`0.369792`

本轮新增 v5 与 ensemble 没有刷新 strict CV 最好。

## v5 smoke

新增变体：`layout_upper_structure_only`。

结果：

- v5 候选池 oracle：`0.412946`，与 v4 持平，没有新增可命中候选。
- 弱布局 oracle 也持平：
  - `document_embed=0.200000`
  - `journal_fig=0.533333`
  - `multi_grid=0.315789`
- v5 choice 单独：`0.348214`，比 v4 choice 单独高。
- v5 full-label route：`0.373512`，与 v4 持平。
- v5 strict CV route：`0.367560`，低于 v4 `0.369792`。

判断：`layout_upper_structure_only` 不采用。它提升了选择器单独分数，但没有提升候选召回，且 strict CV 泛化略差。

## v4/v5 选择器组合

### 多预测路由

把 `main + v4_choice + v5_choice` 放入同一个 group router：

- full-label route：`0.374256`
- strict CV route：`0.366071`

判断：不采用。可见标签路由略高，但 strict CV 明显低于 v4，泛化风险更高。

### reward head 平均 ensemble

对 v4/v5 两个 reward head 做平均打分后再和 main reward 路由：

- ensemble 单独：`0.334821`
- full-label route：`0.373512`
- strict CV route：`0.369048`

判断：不采用。接近 v4 strict CV，但仍低于 `0.369792`。

## 是否更好了

相对最早 selected baseline `0.267113`，当前稳健最好 `0.369792` 是明确提升，约 `+0.102679` 绝对点。

但相对上一轮 v4 最好，本轮没有更好：

- v4 strict CV：`0.369792`
- v5 strict CV：`0.367560`
- v4/v5 多预测 strict CV：`0.366071`
- v4/v5 head ensemble strict CV：`0.369048`

所以当前应保留 v4 strict CV 作为最好产物。

## 下一步

不要继续盲加弱布局裁剪。当前候选池 oracle 卡在 `0.412946`，新增 `layout_upper_structure_only` 没有召回增益。

后续更值得做两条线：

1. 专门做选择器校准：按 `document_embed / journal_fig / multi_grid` 训练 domain-aware reward head 或小 router，约束只在弱布局类覆盖，减少全局误伤。
2. 转向非弱布局短板：`hard / decimer / edu_exam` 的 oracle 仍低，这些不是当前 crop 变体能解决的，需要新的候选来源或针对性后训练。

# 三评测集当前进展与下一步优化计划

日期：2026-06-27

## 结论

现在确实比原始 V2-1 强了。最稳健的候选选择版本是单候选 reward head 直接按最高 reward 选（margin=0），1344 combined 从 `0.267113` 到 `0.350446`。本地按元数据分组调 margin 后，1344 combined 到 `0.354167`。

但还没有达到“强一倍”。原因不是 selector 还没训够，而是候选池 oracle 不够：canonical 主集 oracle 只有 `0.494133`，770 region oracle 只有 `0.527273`，都低于各自翻倍目标。weak_domain_v2 最接近翻倍，但 oracle `0.246101` 也低于翻倍目标 `0.259965`，必须新增候选源。

## 分数表

| 评测集 | N | V2-1 baseline | 稳健 reward m0 | 本地分组 best | candidate oracle | 相对提升(best) | 距离翻倍 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1344 combined | 1344 | 0.267113 | 0.350446 | 0.354167 | 0.387649 | 32.6% | 0.180060 |
| canonical_smiles_main_v1 | 767 | 0.370274 | 0.458931 | 0.461538 | 0.494133 | 24.6% | 0.279009 |
| weak_domain_v2 | 577 | 0.129983 | 0.206239 | 0.211438 | 0.246101 | 62.7% | 0.048527 |
| region_panel_770 | 770 | 0.384416 | 0.437662 | 0.440260 | 0.527273 | 14.5% | 0.328571 |

说明：`稳健 reward m0` 不依赖每个元数据组单独选 margin；`本地分组 best` 依赖 `source/difficulty/task_type` 等字段，适合作为本地上限参考，但隐藏集泛化风险更高。

## 本轮 pair reward smoke

这轮按 candidate-choice / DPO 风格做了一个小型 pairwise reward head smoke：训练数据来自 1344 combined 的 preference pairs，训练对象只是 MLP reward head，不更新 PaddleOCR-VL/VLM 主模型。新增了更强 guard 权重，减少把 selected 正确样本改错。

| 实验 | 全局 best | 分组 best | 结论 |
| --- | ---: | ---: | --- |
| pair guard reward on 1344 | 0.287202 | 0.299107 | 比旧 pair reward 略好，但低于当前 reward m0 |
| same checkpoint eval 770 | 0.415584 | 0.423377 | 有正收益，但低于 train1344->eval770 单候选 reward m0 |

对应产物已保存到：

- `V2-1/reports/pair_guard_reward_smoke_20260627/reward_head.pt`
- `V2-1/reports/pair_guard_reward_smoke_20260627/margin_sweep_summary.json`
- `V2-1/reports/pair_guard_reward_smoke_20260627/report_pair_reward_m1p00.json`
- `V2-1/reports/pair_guard_reward_smoke_20260627/report_pair_reward_group_margin_source+difficulty+task_type.json`

## 当前推荐使用

如果只追求当前本地 1344 combined：

- 稳健版：`V2-1/reports/reward_head_margin0_sweep_20260627/pred_reward_head_m0p00.jsonl`，exact `0.350446`。
- 本地上限版：`V2-1/reports/reward_head_margin0_sweep_20260627/pred_reward_head_group_margin_source_difficulty.jsonl`，exact `0.354167`。

如果要三评测都稳，当前主线仍是单候选 reward head margin=0；pair guard smoke 作为候选方向保留，但不替换主方案。

## 下一步

1. 远端跑 weak layout crop candidate 生成，优先提高 weak_domain_v2 的 oracle。
2. 对 `document_embed/journal_fig/multi_grid/decimer_handdrawn` 做专项 crop/外部 OCSR 候选补充，因为这些组现在基本是候选池缺真值。
3. 再考虑 LoRA/hard replay smoke：目标不是直接改最终模型，而是先验证 candidate oracle 是否上升。
4. 如果新增候选源有效，再重新训练 pair reward / 单候选 reward head，并固定三评测集回归。

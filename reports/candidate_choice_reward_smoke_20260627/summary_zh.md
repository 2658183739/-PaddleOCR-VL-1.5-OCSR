# Candidate-choice reward smoke 总结

日期：2026-06-27

## 结论

- 单独 listwise candidate-choice reward head 在 1344 combined 上没有超过当前主线，不直接替换。
- listwise 在 region_panel_770 上达到 `0.446753`，高于当前 770 稳健主线 `0.437662` 和本地 group best `0.440260`。
- 1344 combined 最好的是“当前 reward m0 + listwise 局部分组路由”，最佳路由字段为 `source+difficulty+task_type`，达到 `0.356399`。
- 该路由同时把 `weak_domain_v2` 从当前稳定 `0.206239` / 旧 local best `0.211438` 提到 `0.216638`。

## 分数对比

| 方案 | 1344 combined | canonical main | weak domain | region 770 |
| --- | ---: | ---: | ---: | ---: |
| 当前稳定 reward m0 | 0.350446 | 0.458931 | 0.206239 | 0.437662 |
| 旧 local group best | 0.354167 | 0.461538 | 0.211438 | 0.440260 |
| cross-run pair-e5 smoke | 0.273065 | - | - | 0.392208 |
| aligned pair-e80 smoke | 0.284970 | 0.398957 | 0.133449 | 0.411688 |
| listwise 单独应用 | 0.333333 | 0.431551 | 0.202773 | 0.446753 |
| listwise route best `source_difficulty_task_type` | 0.356399 | 0.461538 | 0.216638 | - |

## 推荐保留产物

- `combined_1344_best_route_prediction`: `V2-1\reports\candidate_choice_reward_smoke_20260627\listwise_smoke_v1\pred_route_source_difficulty_task_type.jsonl`
- `combined_1344_best_route_report`: `V2-1\reports\candidate_choice_reward_smoke_20260627\listwise_smoke_v1\report_route_source_difficulty_task_type.json`
- `region770_listwise_prediction`: `V2-1\reports\candidate_choice_reward_smoke_20260627\listwise_smoke_v1\pred_best_region770.jsonl`
- `region770_listwise_report`: `V2-1\reports\candidate_choice_reward_smoke_20260627\listwise_smoke_v1\report_best_region770.json`
- `listwise_checkpoint`: `V2-1\reports\candidate_choice_reward_smoke_20260627\listwise_smoke_v1\reward_head.pt`
- `listwise_train_report`: `V2-1\reports\candidate_choice_reward_smoke_20260627\listwise_smoke_v1\listwise_reward_head_report.json`

## 风险

- 分组路由用本地标签比较选择，属于验证集可见条件下的 upper-ish local policy，有过拟合风险。
- selector 已接近现有候选池上限；继续显著提升仍要靠候选池召回，尤其 document_embed / journal_fig / multi_grid。
- 远端 weak_layout_crop_v2 候选池扩展仍是下一步主线，reward/listwise 只负责后续挑选。

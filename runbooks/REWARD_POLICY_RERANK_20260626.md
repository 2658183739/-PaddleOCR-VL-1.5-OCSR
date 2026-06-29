# Reward Policy Rerank 2026-06-26

## 结论

这次没有直接做大模型 DPO 长训，而是做了一个离线 reward-policy candidate reranker。原因是当前可用 preference pairs 主要来自候选 oracle，适合先优化候选选择策略；直接 DPO 权重训练成本更高，且仍受候选是否包含正确答案限制。

最佳可用候选：

- 远端目录：`/root/autodl-fs/outputs_v2/reward_policy_rerank_20260626_split75`
- 本地报告：`V2-1/reports/reward_policy_rerank_20260626_split75`
- 策略：pairwise logistic reward policy
- fallback：`chem_light`
- `policy_margin=1.5`

## 结果

对 `ocsr_realworld_mixed_eval_v1p1` 770 条面板：

| method | canonical exact | mean Tanimoto | uob | uspto | real_world | edu_chemc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected baseline | 0.3844 | 0.6492 | 0.740 | 0.430 | 0.2120 | 0.1046 |
| `chem_light` rerank | 0.3831 | 0.6478 | 0.755 | 0.400 | 0.2074 | 0.1242 |
| reward policy + `chem_light`, margin 1.5 | 0.3935 | 0.6541 | 0.765 | 0.425 | 0.2120 | 0.1242 |

75/25 oracle-positive split check:

- dev subset size: 95
- dev `chem_light` exact: 0.7263
- dev reward-policy hybrid exact: 0.7579
- dev gain over `chem_light`: +0.0316
- dev changes vs `chem_light`: 4 good, 1 bad

## Reproduce

Remote run:

```bash
cd /root/autodl-tmp/data/platform_migration_bundle_20260531
OUT_DIR=/root/autodl-fs/outputs_v2/reward_policy_rerank_v1 \
TRAIN_FRACTION=0.75 \
POLICY_MARGIN=1.5 \
bash V2-1/run_4090_reward_policy_rerank_v1.sh
```

Manual command for the best existing artifact:

```bash
python V2-1/scripts/reward_policy_reranker.py \
  --prediction-jsonl /root/autodl-fs/outputs_v2/full_eval_region_panel_v1_fast_notta/ocsr_realworld_mixed_eval_v1p1/merged/pred.jsonl \
  --labels-jsonl V2-1/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl \
  --load-policy-json /root/autodl-fs/outputs_v2/reward_policy_rerank_20260626_split75/policy_raw.json \
  --output-jsonl /root/autodl-fs/outputs_v2/reward_policy_rerank_20260626_split75/pred_hybrid_m1p5.jsonl \
  --report-json /root/autodl-fs/outputs_v2/reward_policy_rerank_20260626_split75/report_hybrid_m1p5.json \
  --fallback-mode chem_light \
  --policy-margin 1.5
```

## Notes

This is a labeled-panel optimization. Treat it as a candidate postprocess and validation signal, not as a final hidden-test guarantee.

The important open gap is still candidate recall. On this panel, candidate oracle is about 0.54, while the best hybrid exact is about 0.39. DPO or RL on model weights becomes more attractive after collecting more domain-diverse preference pairs, especially real_world and USpto cases where the correct answer is present but not selected.

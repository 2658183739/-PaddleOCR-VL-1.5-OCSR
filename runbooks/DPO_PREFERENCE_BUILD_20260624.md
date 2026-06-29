# DPO 偏好对构建说明

这份说明只讲数据，不讲训练。

## 输入

推荐直接吃现有推理结果：

- `pred.jsonl`，最好带 `--save-candidates`
- 对应 benchmark 的 `labels.jsonl`

## 输出

默认输出两个文件：

- `dpo_pairs.jsonl`
- `dpo_report.json`

建议放在：

```text
V2-1/data/dpo_materialized/
```

## 直接命令

```bash
python scripts/build_ocsr_dpo_preferences.py \
  --project-root . \
  --benchmark-jsonl V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl \
  --prediction-jsonl V2-1/eval_runs_export_full/weak_domain_v2_continue_sft_beam4/pred.jsonl \
  --output-jsonl V2-1/data/dpo_materialized/dpo_pairs.jsonl \
  --report-json V2-1/data/dpo_materialized/dpo_report.json
```

## 规则

- `chosen` 固定用 gold `canonical SMILES`
- `rejected` 取同图同 prompt 下最像样但仍错误的候选
- 没有干净 rejected 的样本直接丢掉
- 如果 gold 不能 canonicalize，也丢掉

## 先看什么

先看 `dpo_report.json`：

- 总共能产多少对
- 哪个来源占得最多
- rejected 是有效错答案还是无效错答案

如果某个来源太偏，就先限额，不要直接开训。

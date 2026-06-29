# real_world 提分路线和 DPO 准入条件

这条路线分两段：先把 real_world 的候选生成救起来，再考虑 DPO。

## 当前事实

- 原始 V2-1 + `chem_light` 在 UOB80 上能到 0.8000 canonical exact。
- mixed60 里 UOB 能到 0.8500，USpto 只有 0.4500，real_world/chinese_exam 是 0。
- realworld20 高分辨率 no-TTA 还是 0/20，candidate oracle 也是 0/20。

所以现在的问题不是“怎么从候选里选对”，而是“候选里没有对的”。

## 已新增文件

- `V2-1/scripts/build_realworld_preprocess_probe.py`
- `V2-1/run_4090_realworld_preprocess_probe_v1.sh`
- `V2-1/scripts/build_preference_training_sets.py`

## 正在跑的探针

远端输出：

```bash
/root/autodl-fs/outputs_v2/realworld_preprocess_probe_v1
```

它会跑 4 个预处理版本：

- `crop`
- `crop_gray_auto`
- `crop_gray_sharp`
- `crop_bw_thicken`

每个版本都跑：

1. V2-1 原模型推理，保存 candidates。
2. 标准评测。
3. `chem_light` rerank。
4. rerank 后评测。
5. 汇总 selected、rerank、oracle。

## DPO 怎么接

先收集 preference pairs：

```bash
python V2-1/scripts/build_preference_training_sets.py \
  --project-root /root/autodl-tmp/data/platform_migration_bundle_20260531 \
  --preference-jsonl /root/autodl-fs/outputs_v2/v2_1_original_compare/eval/uob_medium_80/preference_pairs.jsonl \
  --preference-jsonl /root/autodl-fs/outputs_v2/v2_1_original_compare/eval/mixed_uob_uspto_realworld_60/preference_pairs.jsonl \
  --output-dir /root/autodl-fs/outputs_v2/preference_training_sets/v1
```

输出：

- `preference_pairs_merged.jsonl`
- `preference_positive_sft_messages.jsonl`
- `preference_dpo_chosen_rejected.jsonl`
- `preference_dataset_report.json`

## 什么时候可以 DPO

满足这些条件再开：

- preference pairs 至少几百条。
- 里面有 real_world 或 USpto 的正例。
- 某个面板的 candidate oracle 明显高于 selected exact。
- PaddleFormers 对当前 PaddleOCR-VL 训练入口确认支持 DPO，或者另建可靠 DPO 训练入口。

现在不满足。现在硬做 DPO，只会把 UOB 上少量排序错误学一遍，解决不了 real_world。

## 如果预处理探针有效

如果某个预处理版本让 realworld20 的 oracle 从 0 变成大于 0：

1. 用这个预处理版本跑 mixed60。
2. 如果 mixed60 不伤 UOB，再跑 UOB80。
3. 把这个预处理接到候选生成流程里。
4. 再积累 preference pairs。

## 如果预处理探针无效

直接转数据路线：

1. 人工抽查 realworld20 图像和标签。
2. 做 100-300 条同风格训练数据。
3. 从原始 V2-1 `outputs/export` 短训，不从 fast90 接。
4. 先看 realworld20 oracle 是否从 0 变动。

停损条件：短训后 realworld20 oracle 还是 0，就不要扩大训练。

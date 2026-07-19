# V3 评测与选模协议

## 1. 面板角色

| 面板 | 角色 | 是否选模 |
| --- | --- | --- |
| `dev_legacy_core_strict` | 历史连续 development | 是 |
| `dev_legacy_region_strict` | real/page/crop development | 是 |
| `wild_strict_v3` | 论文分组 locked final test | 否，只在最终运行 |
| `wild_strict_scaffold_novel_v3` | locked 泛化诊断子集 | 否 |
| `wild_symbolic_v3` | symbolic 独立 track | 否 |
| `private_photo_v3` | 自采 locked final test | 否 |

历史 core/region 已用于 V2-1 调参，因此在 V3 中不再称为最终测试。

## 2. 泄漏控制

- train/development/locked test 按 canonical molecule 隔离。
- MolRecBench 先按 `paper_group` 留出整篇论文，再在每篇最多选 5 图。
- 自采照片按 `structure_id` 分组，四种退化必须进入同一 split。
- 图片名、ID、canonical SMILES、paper group 和 SHA256 均保留审计。
- scaffold overlap 仅作诊断；`wild_strict_scaffold_novel_v3` 单独报告。

## 3. 指标

主指标：RDKit canonical exact。

辅助指标：raw exact、valid SMILES、fingerprint Tanimoto、stereo exact、source/difficulty/task-type exact。Tanimoto 不能替代 exact；非法预测的 validity 下降不能被相似度掩盖。

## 4. 统计单位

- 普通单图 development：至少按 `structure_id` 聚类重复分子。
- MolRecBench：按 `paper_group` cluster bootstrap。
- 自采多退化：按 `structure_id` cluster bootstrap。
- 独立的一图一结构配对才使用 exact McNemar。

```bash
python V3/scripts/compare_eval_runs.py \
  --baseline-details baseline/details.jsonl \
  --candidate-details candidate/details.jsonl \
  --cluster-field paper_group \
  --output-json candidate/paired_clustered.json
```

脚本使用 10,000 次 paired cluster bootstrap，输出差值均值、95% CI 和 `P(delta>0)`。cluster 模式会明确跳过 image-level McNemar。

## 5. Checkpoint 选择

1. 只看 development。
2. 先过 valid/printed 回归闸门。
3. 再比较 dev core 与 dev region 宏平均。
4. 分数接近时选更早 checkpoint。
5. 完成后冻结模型、配置、prompt、decoding 和 hash。
6. 最后一次性解锁 wild/private。

禁止在 locked test 上搜索 prompt、beam、TTA、margin、crop、checkpoint 或 router。

## 6. Locked test 解释

`wild_strict_v3` 有 301 张图、301 个唯一 canonical 分子，但统计上的主要独立来源是 62 篇论文。报告必须同时写 `N_images=301`、`N_molecules=301` 和 `N_papers=62`，不能只写图片数。

private photo 同理：如果 80 个结构各拍 4 张，图像数为 320，但结构层面的独立 N 只有 80。

locked test 使用后若继续调参，这次分数只能标为 exploratory；新的 confirmatory 结论需要新测试集。

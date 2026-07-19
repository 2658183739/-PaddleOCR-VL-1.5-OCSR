# V3 实际训练与最终评测结果

> 本文件由 `scripts/build_final_report.py` 从 JSON 证据自动生成。
> Locked test 不参与任何训练、checkpoint、prompt 或生成参数选择。

## 1. 数据与划分

| 项目 | 数量 | 角色 |
| --- | ---: | --- |
| V2-1 输入记录 | 23047 | 过滤前训练输入 |
| strict control | 22762 | final 胜出训练配比 |
| strict wild train | 800 | 仅进入含 wild 的 probe |
| wild strict locked | 301 | 62 篇整论文留出 |
| scaffold novel | 134 | locked wild 子集 |
| symbolic | 460 | 独立转写 track |

训练与 locked test 的 canonical molecule 和 `paper_group` 重叠均为 0；legacy core/region 是历史 development。

## 2. 2x2 两 seed 数据消融

| 条件 | 两 seed 宏平均 exact | seed 范围 | 最低 valid rate |
| --- | ---: | ---: | ---: |
| 00 | 0.3411 | 0.0093 | 0.7131 |
| 10 | 0.3328 | 0.0126 | 0.7357 |
| 01 | 0.3344 | 0.0013 | 0.7317 |
| 11 | 0.3391 | 0.0053 | 0.7371 |

- wild 主效应：-0.0018
- augmentation 主效应：-0.0002
- 交互效应：0.0129
- 最终训练数据：`./V3/data/sft_materialized/train_v3_a_control.jsonl`
- 解释边界：本轮只有两个 seed，运行顺序未完全随机化或位置平衡；以下结论是工程探索，不报告 ANOVA p 值或统计显著性。

### 2.1 逐样本 paired bootstrap

| baseline | candidate | panel | units | exact delta | 95% CI |
| --- | --- | --- | ---: | ---: | --- |
| data_11_s1 | aug_dose2_s1 | legacy_core_dev | 743 | -0.001346 | [-0.012786, 0.010111] |
| data_11_s1 | aug_dose2_s1 | legacy_region_dev | 744 | -0.002688 | [-0.014785, 0.009409] |
| data_00_s1 | data_01_s1 | legacy_core_dev | 743 | 0.000673 | [-0.010767, 0.012113] |
| data_00_s1 | data_01_s1 | legacy_region_dev | 744 | 0.000672 | [-0.011425, 0.012769] |
| data_00_s2 | data_01_s2 | legacy_core_dev | 743 | -0.009421 | [-0.022880, 0.003382] |
| data_00_s2 | data_01_s2 | legacy_region_dev | 744 | -0.012097 | [-0.025538, 0.001344] |
| data_00_s1 | data_10_s1 | legacy_core_dev | 743 | 0.004711 | [-0.004711, 0.014805] |
| data_00_s1 | data_10_s1 | legacy_region_dev | 744 | 0.002016 | [-0.008065, 0.013441] |
| data_00_s2 | data_10_s2 | legacy_core_dev | 743 | -0.018170 | [-0.030956, -0.006057] |
| data_00_s2 | data_10_s2 | legacy_region_dev | 744 | -0.019489 | [-0.032258, -0.007392] |
| data_00_s1 | data_11_s1 | legacy_core_dev | 743 | 0.008748 | [-0.003365, 0.021534] |
| data_00_s1 | data_11_s1 | legacy_region_dev | 744 | 0.006048 | [-0.006720, 0.019489] |
| warmstart_control_s1 | data_11_s1 | legacy_core_dev | 743 | 0.337820 | [0.304172, 0.371467] |
| warmstart_control_s1 | data_11_s1 | legacy_region_dev | 744 | 0.346774 | [0.313172, 0.380376] |
| data_00_s2 | data_11_s2 | legacy_core_dev | 743 | -0.006057 | [-0.018170, 0.006057] |
| data_00_s2 | data_11_s2 | legacy_region_dev | 744 | -0.011425 | [-0.023522, 0.000672] |

主因子比较中，除 wild-only/seed2 两个面板的 CI 完全低于 0 外，其余单 seed CI 均跨 0；这不支持 wild 或 augmentation 的稳定正向收益。warm-start 对照则在两个面板均有明确正向 CI。

## 3. 辅助消融

- 增强剂量 2：macro exact=0.3397，相对 11/seed1=-0.0020。
- V2-1 continuation 相对原始 1.5 warm-start：0.3417。

## 4. 训练成本

| run | train loss | runtime (min) | samples/s | steps/s |
| --- | ---: | ---: | ---: | ---: |

## 5. Final checkpoint 与 hard replay

| checkpoint | step | development macro exact | min valid |
| --- | ---: | ---: | ---: |
| checkpoint-200 | 200 | 0.3470 | 0.7437 |
| checkpoint-400 | 400 | 0.3391 | 0.7410 |
| checkpoint-600 | 600 | 0.3305 | 0.7503 |
| checkpoint-800 | 800 | 0.3444 | 0.7503 |
| checkpoint-1000 | 1000 | 0.3378 | 0.7371 |
| checkpoint-1200 | 1200 | 0.3537 | 0.7371 |
| checkpoint-1400 | 1400 | 0.3597 | 0.7463 |

- 选中 checkpoint：`checkpoint-1400`，development macro exact=0.3597，min valid=0.7463。
- hard replay 最终决策：`final`。
- hard replay macro delta：-0.0073；采用门槛：至少 0.0050。
- hard replay development macro exact：0.3524；final baseline：0.3597。
- 最终生成策略：`beam4_return4`。
- Decoder 对照：`beam4_return4` macro exact=0.4207，`greedy`=0.3597，delta=0.0610。
- Rerank 对照：`beam4_chem_light` macro exact=0.3955，`beam4_return4`=0.4207，delta=-0.0252；采用门槛：至少 0.0050。

### 5.1 后训练 paired bootstrap

| comparison | panel | units | exact delta | 95% CI | valid delta |
| --- | --- | ---: | ---: | --- | ---: |
| generation_policy | legacy_core_dev | 743 | 0.060565 | [0.041723, 0.080081] | 0.139973 |
| generation_policy | legacy_region_dev | 744 | 0.060484 | [0.040995, 0.080645] | 0.110215 |

## 6. 一次性 locked test

| 面板 | N | 主 exact | valid/nonempty |
| --- | ---: | ---: | ---: |
| wild strict | 301 | 0.2292 | 0.8472 |
| scaffold novel | 134 | 0.1343 | 0.7537 |
| symbolic（独立 track） | 460 | 0.0000 | 1.0000 |

## 7. 解释边界与证据入口

- wild strict/scaffold novel 才进入 canonical SMILES 主结论。
- symbolic 是文字转写诊断，不使用 RDKit canonicalization，也不混入主分数。
- 项目所有者确认 frozen legacy/wild/symbolic labels 已完成离线人工审核；公开证据为 `qc/manual_review_attestation.json` 及其绑定的四个 labels SHA256。
- private photo 若为 N/A，表示没有真实自采 locked test，算法退化不能替代实拍。
- 模型、配置、评测标签、prompt 与生成策略 hash 见 locked run 下的 `locked_test_manifest.sha256`。
- 可恢复 final/hard-replay checkpoint 位于 `evidence/training_artifacts/resume/`。
- 实际环境快照位于 `evidence/runtime/`。

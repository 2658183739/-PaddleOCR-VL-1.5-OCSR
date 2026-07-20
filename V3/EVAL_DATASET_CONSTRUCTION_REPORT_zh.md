# PaddleOCR-VL OCSR V3 评测集数据构建报告

版本：V3 final，2026-07-19  
适用范围：V3 训练、development、locked final test、symbolic 独立轨道及公开提交材料。

## 摘要

本报告说明 V3 评测数据如何从 V2-1 的历史面板演进为可审计的开发集、论文级锁定终测集和独立 symbolic 轨道。V2-1 已经完成了 OCSR 任务适配，但其 `canonical_smiles_main_v1`（767 条）和 `region_panel_770`（770 条）曾反复用于历史调参，因此 V3 不再把它们称为未触碰测试集，而是把清洗后的子集作为 legacy development，用于 checkpoint、数据因素和解码策略选择。V3 的最终结果只来自策略冻结后的一次性 locked test。

V3 的主任务始终是单图输入、单分子输出和 `canonical_smiles` 目标。最终开发面板为 `753 + 754 = 1,507` 条，locked wild 为 `301` 张图、`301` 个唯一 canonical 分子、来自 `62` 篇留出论文，scaffold-novel 诊断子集为 `134` 条；MolRecBench 的 R-group/缩写文字转写另设 `460` 条 symbolic 轨道，不混入 canonical exact。当前未加入 private-photo locked test。

## 1. 与 V2-1 的关系

V2-1 的贡献是先把基础模型从通用视觉语言模型适配成能够输出 OCSR 结果的模型。历史同口径面板显示，`canonical_smiles_main_v1` 的 SFT baseline exact 为 `0.370274`（37.03%），稳定后处理结果为 `0.458931`（45.89%），best 为 46.15%；`region_panel_770` 的 baseline 为 38.44%，稳定结果为 43.77%，best 为 44.03%。另一份固定预算 warm-start 对照中，原始 PaddleOCR-VL-1.5 在两个 development 面板 exact 均为 0，而 V2-1 continuation 的宏平均 exact 为 34.17%。这些数字回答的是“是否已经学会 OCSR 输出格式”，不能与 V3 locked wild 的外推分数纵向混算。

在同一历史诊断口径下，原始 PaddleOCR-VL-1.5 的 canonical exact、valid SMILES、token micro-F1 和 mean Tanimoto 分别为 0.00%、30.78%、6.59% 和 0.0027；V2-1 LoRA export 分别为 33.77%、75.84%、70.18% 和 0.6849。V3 因此采用从 V2-1 export 继续做低学习率 LoRA SFT 的路线，把有限预算用于数据混合、checkpoint、后训练和解码策略的可解释对照，而不是重复学习输出格式。

## 2. 评测角色与统计单位

评测角色在构建前冻结。development 只用于选择 checkpoint、比较训练数据因素和检查 decoder；locked final test 只在模型、prompt、生成策略和文件 hash 全部冻结后运行一次；symbolic 轨道只做文字转写诊断，不参与 canonical exact 选模。普通 OCSR 使用 canonical molecule 或 `structure_id` 作为独立单位；MolRecBench 先按 `paper_group` 留出整篇论文，再在每篇论文最多取 5 张图。这样可以避免同一分子、同一论文或同一图像视角被错误当作独立重复。

## 3. 评测目录清单

下表是工作区 `V3/data/eval/` 当前实际目录。带有“历史/辅助”标记的目录不用于 V3 locked final score，但保留用于追溯 V2-1、来源和构建过程。

| 目录 | 样本量 | 内容 | V3 角色 |
| --- | ---: | --- | --- |
| `canonical_smiles_main_v1/` | 767 | `annotations/labels.jsonl`、`labels.csv`；`images/decimer`、`uob`、`uspto`、`real_world`，来源数为 150/200/200/217 | V2-1 历史主面板；清洗后派生 `dev_legacy_core_strict` |
| `ocsr_realworld_mixed_eval_v1p1/` | 770 | `annotations/`、`images/`、`mixed_eval_summary.json`、来源溯源、候选池、技术报告和统计文件 | V2-1 历史 region 面板；清洗后派生 `dev_legacy_region_strict` |
| `dev_legacy_core_strict/` | 753 | 仅保留 V3 development 用的 `labels.jsonl`，图像通过相对路径引用主面板资产 | V3 core development；用于 checkpoint 和 decoder 选择 |
| `dev_legacy_region_strict/` | 754 | 仅保留 V3 development 用的 `labels.jsonl`，覆盖教育图、printed 图和 real-world 补充 | V3 region development；用于 crop/region 回归检查 |
| `molrecbench_wild_300/` | 300 | `labels.jsonl` 和 300 张文章结构图；包含官方困难标签和 MolRecBench 来源信息 | 辅助/历史 wild 子集；不是 V3 locked final，标签仍标为 pending manual review |
| `wild_strict_v3/` | 301 | 冻结的论文级留出 `labels.jsonl`；图像引用 `V3/data/assets/molrecbench_wild_v1/images/` | V3 locked final canonical test |
| `wild_strict_scaffold_novel_v3/` | 134 | 从 locked wild 派生的未见 Bemis-Murcko scaffold 子集 `labels.jsonl` | locked 泛化诊断，不单独调参 |
| `wild_symbolic_v3/` | 460 | 独立 symbolic/R-group 文字标签 `labels.jsonl`，保留 symbolic 写法，不做 RDKit canonicalization | 独立 symbolic track，不计入 canonical exact |

最终提交源码仓库只保留评测说明、QC 报告和构建脚本；训练原图、locked 逐样本预测和大体积资产不重复放入 GitHub。可复核的本地标签、图像和证据保留在项目工作区及对应的受许可来源中。

## 4. 构建流程

### 4.1 V2-1 历史面板清洗

V2-1 的 `canonical_smiles_main_v1` 767 条和 `region_panel_770` 770 条先进行结构性检查。每条记录必须有一个图像、一个 assistant 输出、一个目标字段、可打开的图像路径以及可由 RDKit 解析的单分子标签。`canonical_smiles` 面板保留 DECIMER、UOB、USPTO 和 real-world 四类来源；region 面板额外保留 EDU-CHEMC 教育场景。清洗结果形成 753 条 core development 和 754 条 region development，非 canonical、多片段或无法稳定解析的记录不进入主评分。

### 4.2 MolRecBench 论文级留出

原始 MolRecBench-Wild 共 5,008 条记录。构建器先排除 symbolic 或无效候选，再检查与历史 development 分子的重叠，得到 1,428 条 strict pool。随后按 `paper_group` 对整篇论文进行留出，519 篇论文中选择 62 篇作为 evaluation paper groups，每篇最多保留 5 张图，得到 301 条 wild strict。留出论文中剩余的 308 条不进入最终评测，以避免由于每篇论文样本量不均而改变评测权重。训练侧保留 800 条 strict train；训练与 locked test 的 paper group 和 canonical molecule 均不重叠。

### 4.3 Scaffold 与 symbolic 派生

在 301 条 locked wild 中，使用 Bemis-Murcko scaffold 检查训练覆盖，得到 134 条训练未见 scaffold 的诊断子集。另从 MolRecBench 的 R-group、缩写和 symbolic 标签构建 460 条独立轨道。symbolic 标签有意保留其原始转写约定，不经过 RDKit canonicalization，也不把它们的 exact 结果混入主任务。

### 4.4 训练集反向去泄漏

V2-1 clean weighted 训练输入为 23,047 条。V3 构建器删除 273 条多片段、symbolic 或不收敛记录，并删除 12 条与新 held-out molecule 重叠的记录，得到 22,762 条 strict control。MolRecBench 论文级划分还识别出 72 条与 legacy development 分子重叠的候选，以及 19 条需要从训练记录中反向删除的 locked molecule overlap。所有筛选都写入 `evidence/dataset_build_report.json`、`evidence/wild_paper_group_split.jsonl` 和 QC 报告。

## 5. 数量与质量控制证据

构建器记录的关键计数如下：基础训练输入 23,047 条，过滤后 22,762 条；core development 接受 753 条，region development 接受 754 条；MolRecBench strict pool 1,428 条，evaluation paper groups 62 个，wild strict 301 条，scaffold-novel 134 条，symbolic 460 条。训练集中的新评测分子重叠删除数为 12，legacy development 反向重叠识别数为 72，locked molecule 训练侧删除数为 19。

每条正式记录都至少经过路径、图像尺寸、单图单目标、标签格式和 RDKit 解析检查。`canonical_smiles_main_v1` 的 767 条和 `region_panel_770` 的 770 条历史面板在旧 QC 中均通过规则检查；V3 的 locked labels、symbolic labels 和两个 development manifest 由项目 owner 声明已完成离线审核，冻结后没有再剔除或改写标签。该声明不是第二台机器的独立复现，也不等同于双人逐样本审计，限制必须在提交材料中保留。

## 6. 评测指标与结果边界

V3 主指标是 RDKit canonical exact。valid SMILES、token F1、normalized edit similarity 和 fingerprint Tanimoto 仅用于解释结构有效性与错误类型，不能替代 exact。locked wild 的最终结果为 canonical exact 22.92%、valid 84.72%；其中 scaffold-novel 子集 exact 13.43%、valid 75.37%。symbolic 460 条采用独立文字转写口径，whitespace-normalized exact 为 0%，nonempty prediction rate 为 100%，该结果不参与 canonical decoder 选择。

development 面板只承担选择责任：两个面板共 1,507 条，用于选择 `checkpoint-1400`、拒绝 300-step hard replay、采用 beam4/return4，并拒绝 chem-light rerank。locked test 在这些策略全部冻结后运行，结果不回流训练、prompt、checkpoint、beam 或 rerank。

## 7. 来源、许可与公开边界

评测标签中的 `decimer`、`uob`、`uspto`、`real_world` 和 `molrecbench_wild` 均保留 source、上游 ID、来源 URL 或文档字段。MolRecBench-Wild 标签注明 Apache-2.0 来源；项目代码和派生权重按 Apache-2.0/NOTICE 发布，但不把受许可限制的训练原图和 locked 逐样本预测无条件重新分发。`DATA_LICENSES_AND_ATTRIBUTION_zh.md`、`V3/data/eval/ocsr_realworld_mixed_eval_v1p1/SOURCE_PROVENANCE_zh.md` 和各目录 README 是来源与许可的补充证据。

## 8. 可复核文件索引

最终训练与评测证据位于以下文件：`V3/evidence/dataset_build_report.json` 保存输入、过滤、分组和混合统计；`V3/evidence/wild_paper_group_split.jsonl` 保存论文级 split；`V3/evidence/FINAL_RESULTS.json` 保存锁定结果；`V3/evidence/final_checkpoint_selection.json`、`final_vs_hard_replay.json` 和 `generation_policy_beam_selection.json` 保存选模与解码决策；`V3/qc/QC_REPORT_V3_zh.md` 保存质量控制摘要；`V3/runbooks/EVALUATION_PROTOCOL_zh.md` 保存禁止回流和统计口径。

训练侧的可复现 JSONL 位于 `V3/data/sft_materialized/`：`dev_legacy_core_strict_messages.jsonl` 753 条、`dev_legacy_region_strict_messages.jsonl` 754 条、`val_singleline_v1p1_messages.jsonl` 770 条，以及 `train_v3_a_control.jsonl` 22,762 条和其他受控消融混合。`val_singleline_v1p1_messages.jsonl` 是历史诊断面板，不能替代 V3 locked final。

## 9. 限制与提交声明

当前版本没有 private-photo locked test；第二台 GPU clean-download、独立环境从零复现和四 seed confirmatory 训练也尚未完成。V3 的 locked 结果因此应表述为“在冻结 manifest、冻结策略和当前 owner-attested QC 边界下的最终报告”，而不是宣称已经完成独立机器的完全复现。任何在 locked test 之后继续调参的结果都只能标记为 exploratory，不能继续使用本报告中的 confirmatory 口径。

本报告与 `TRAINING_DATA_AND_FINETUNING_REPORT_zh.md` 共同构成训练数据和评测数据的正式说明；答辩 PPT 只展示关键数字，逐文件复核以本报告和 JSON/hash 证据为准。

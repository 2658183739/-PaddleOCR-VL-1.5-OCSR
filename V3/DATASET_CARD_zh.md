# PaddleOCR-VL OCSR V3 数据卡

## 任务定义

每条主任务记录包含一张分子结构图、固定任务 prompt 和一个单分子 canonical SMILES。训练入口拒绝多片段、dummy atom、RDKit 不可解析标签和缺失图片。

## 数据集规模

| 名称 | 记录数 | 用途 |
| --- | ---: | --- |
| A control | 22,762 | wild off，augmentation off |
| D wild-only | 23,562 | 加入 800 条 strict wild train |
| E aug-only | 23,689 | 加入 927 条离线退化图 |
| B wild+aug | 24,489 | 同时加入两种因素 |
| C dose-2 | 25,416 | 额外加入第二档 927 条退化图 |
| hard replay seed | 7,000 | 困难样本短程 continuation |

记录数包含历史重复权重，不能当成独立分子数。详细唯一图片、唯一 canonical molecule、来源和难度统计见 `evidence/dataset_build_report.json` 与 `evidence/mixture_counts.csv`。

## Final control 的七部分来源与配比

为避免把内部自动弱标签子类误写成更多数据集，提交报告按七个上游语义组归并 `A_control`：

| 语义组 | 记录数 | 占比 | 清洗/使用说明 |
| --- | ---: | ---: | --- |
| USPTO | 5,043 | 22.16% | `uspto` 5,035 + 自动弱 USPTO 8；公开标签 canonical 化，保留单分子。 |
| UOB | 4,869 | 21.39% | 公开 benchmark 标签；RDKit 解析、canonicalization、坏图和多片段剔除。 |
| real-world | 4,329 | 19.02% | `real_world` 4,125 + 自动弱 real-world 204；不是一个单独 benchmark 名，模型伪标签不作为真值。 |
| MolGrapher synthetic | 4,000 | 17.57% | 合成/生成图和自带结构标签；只用于训练增强，不计作真实评测。 |
| USPTO-30K clean | 1,501 | 6.59% | 公开标签，cap 约 1,500，避免 printed 子集过采样。 |
| USPTO-30K abbreviated | 1,507 | 6.62% | 公开标签，保留缩写结构长尾。 |
| USPTO-30K large | 1,513 | 6.65% | 公开标签，保留大图和长分子长尾。 |
| **合计** | **22,762** | **100.00%** | V3 strict control，包含 repeat/cap 权重。 |

最终训练清单的源类别统计可以 100% 覆盖到上表，但历史清单没有逐样本
`license/source_url_or_doc/structure_id`。因此公共 release 不分发训练 JSONL 或原图，
只发布聚合统计、构建代码和许可矩阵；逐来源证据与隔离策略见
`DATA_LICENSES_AND_ATTRIBUTION_zh.md`。

配比不是平均采样：UOB/USPTO 保持约 43.5% 的 printed 锚点，real-world 保留约 19.0% 的稀缺场景，MolGrapher 约 17.6% 用于复杂结构，三个 USPTO-30K 子集各限制在约 1,500 条。这个选择由 V2-1 错误分层、数据可信度和 2×2 探索共同约束；两 seed 只能支持“本轮探索下的选择”，不能声称全局最优。

## 划分原则

- 普通 OCSR：按 canonical molecule/`structure_id` 防止同分子跨 split。
- MolRecBench：先按 `paper_group` 留出整篇论文，再在论文内抽样；train/test 论文重叠为 0。
- 自采多视角：同一 `structure_id` 的全部照片必须进入同一 split。
- legacy core/region：由于 V2-1 已多次用于调参，只作为 development。
- locked wild：301 张、301 个唯一 canonical molecule、62 篇留出论文；其中 134 张为训练未见 scaffold。
- symbolic：460 条单独评测，不参与 canonical exact。

`evidence/wild_paper_group_split.jsonl` 保存论文分组清单。构建脚本同时检查 canonical molecule、论文组和图片路径泄漏。

## 评测抽样与比例依据

legacy development 面板的目标配额和严格通过数如下。目标配额先冻结，QC 后不为凑数回填被拒记录：

| 面板 | 目标配额 | strict 通过 | 采样目的 |
| --- | --- | ---: | --- |
| core | DECIMER 150、UOB 200、USPTO 200、real-world 217 | 753（150/193/196/214） | 约四分之一的 UOB/USPTO printed 锚点 + 约五分之一手绘 + 全部可用真实补充样本。 |
| region | EDU-CHEMC 153、UOB 200、USPTO 200、real-world 217 | 754（151/193/196/214） | 保留教育/页面目标区域困难样本，用于 crop、region 和真实退化回归。 |
| wild strict | 1,428 条 strict pool，先留出 62 篇论文 | 301 | 按 `paper_group` 整篇留出，每篇最多 5 图，保证论文外推而非行级随机切分。 |
| scaffold novel | wild strict 子集 | 134 | 训练未见 Bemis-Murcko scaffold 的二级泛化诊断。 |
| symbolic | MolRecBench symbolic/R-group pool | 460 | 单独报告，不混入 canonical SMILES exact。 |

UOB/USPTO 的约 200 条是可比较的公开 benchmark anchor，不代表部署场景自然比例；DECIMER/EDU 约 150 条用于保证手绘/教育难点有足够样本；real-world 清洗后全部保留，避免稀缺场景被随机下采样消失。严格数值来自各面板 `labels.jsonl`，QC 前后统计见 `evidence/v2_1_eval_qc_summary.json`。

## 数据质量

自动 QC 包括图片存在/可读、RDKit 解析、单分子、无 dummy atom、canonicalization 幂等、分组零重叠和 hash 验证。本地自动化测试为 `29/29` 通过；H800 在最终打包前运行同一套测试。

自动 QC 不能证明图片和标签语义一致。项目所有者已经确认 legacy development、
wild strict 和 symbolic 的离线人工审核完成，未报告审核后剔除或标签修订；公开证据
采用 `qc/manual_review_attestation.json` 与冻结清单 SHA256 绑定。历史
`qc/eval_manual_review.csv` 保留为工具模板，不作为本次外部审核的权威完成记录。
公开材料不披露或虚构审核人姓名、签名和逐样本内部决定。

## 来源与许可边界

数据包含公开基准、历史真实场景、合成图、论文裁图和离线算法退化。算法退化只用于训练增强，不计作真实评估实例，也不能写成自行实拍。基础模型为 Apache-2.0；UOB/USPTO 的可追溯分发仓库为 MIT；MolRecBench-Wild 为 Apache-2.0；MolGrapher Synthetic 与 USPTO-30K 的数据卡未声明独立 dataset license，real-world 也没有单一许可证。因此公共仓库隔离全部训练原图和训练清单，只保留统计、构建方法和引用。详见 `DATA_LICENSES_AND_ATTRIBUTION_zh.md` 与 `evidence/release_readiness_audit.md`。

非公开真实手绘/拍照部分目前没有已完成的自采数据：`private_photo_v3` 为 0，不能把历史项目整理图或算法退化写成“自行采集”。发布前的真实采集协议要求至少记录设备、角度、光照、采集人、时间、授权、`structure_id` 和 split，并按同一结构聚类分割；完整流程见 `runbooks/PRIVATE_COLLECTION_PROTOCOL_zh.md` 和 `qc/QC_REPORT_V3_zh.md`。

## 重建

```powershell
& '.\.conda_rdkit\python.exe' V3\scripts\build_v3_datasets.py --project-root .
& '.\.conda_rdkit\python.exe' V3\scripts\verify_v3_workspace.py --project-root .
& '.\.conda_rdkit\python.exe' -m unittest discover -s V3\tests -v
```

完整图像资产不重复放入最终模型包；最终包包含训练/评测 JSONL、来源清单、统计报告和 hash，原始图像由 `V3.tar` 或受许可的数据源提供。

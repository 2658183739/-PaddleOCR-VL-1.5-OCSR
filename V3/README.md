# PaddleOCR-VL OCSR V3

V3 是一个面向两天冲刺、同时强调数据质量与实验可解释性的 OCSR 工作区。任务定义保持单一：输入一张分子结构图，输出一行可由 RDKit 解析的 canonical SMILES。

本目录继承 V2-1 已经训练出的 OCSR 能力，但重新整理了数据口径、开发集/测试集角色、数据配比消融、统计检验和质检证据。README 既是运行入口，也是方法说明；实验事实、owner-attested 人工审核、公开发布状态和剩余限制分别报告。

提交材料的正式说明分为三层：

- `TRAINING_DATA_AND_FINETUNING_REPORT_zh.md`：训练数据七部分、标签清洗、配比消融、后训练和复现实验记录。
- `OFFICIAL_FEEDBACK_RESPONSE_zh.md`：逐条对应官方意见，区分已完成证据、待人工完成事项和发布前门槛。
- `DATASET_CARD_zh.md` / `MODEL_CARD_zh.md`：面向使用者的数据和模型限制说明。
- `RELEASE_UPLOAD_GUIDE_zh.md`：比赛包、GitHub 与 Hugging Face 的发布边界和验收步骤。
- `DATA_LICENSES_AND_ATTRIBUTION_zh.md`：逐来源许可证据、归属和“不重新分发训练原图”的边界。

公开地址为 [GitHub](https://github.com/2658183739/-PaddleOCR-VL-1.5-OCSR) 与 [Hugging Face](https://huggingface.co/L2658183739/PaddleOCR-VL-1.5-OCSR)。GitHub 的历史提交保留 V2-1 参考，当前 `V3/` 与 HF 主分支发布冻结的 V3 final；不同面板和版本不能纵向混算提升率。

## 1. 结论先读

### 1.1 模型路线

默认路线是从 `models/v2_1_export/` 继续做低学习率 LoRA SFT，而不是从 PaddleOCR-VL-1.5 原始权重重新开始。

原因：

1. PaddleOCR-VL-1.5 原始模型没有稳定的 canonical-SMILES 输出能力，历史 exact 基本接近 0。
2. V2-1 已完成 OCSR 任务适配，历史主面板 SFT baseline 为 0.370274，后处理 stable 为 0.458931。
3. 4090 历史训练 1600 step、有效 batch 32，耗时 3:11:23；本次使用 H800 PCIe 80GB 提高实验吞吐，但换卡本身不会自动提升精度。
4. 当前本地没有可核验的 PaddleOCR-VL-1.6，因此不让一个未确认权重成为两天主线依赖。

`models/paddleocr_vl_1_5_base/` 已保留，用于 warm-start 对照实验。

### 1.2 方法是否“权威”

这里采用的不是某个比赛唯一指定标准，而是机器学习和实验设计中广泛接受的方法组合：

- 固定 seed、同一基座、同一步数的受控对照；
- `2×2` 因子消融，分离数据因素的主效应与交互作用；
- 按 canonical molecule、论文来源和自采 `structure_id` 分组防泄漏；
- development 与 locked final test 分离；
- paired bootstrap 置信区间；
- 独立样本才使用 McNemar，重复拍照按 cluster bootstrap；
- RDKit canonical exact 为主指标，validity 和 Tanimoto 为辅助指标；
- Bemis-Murcko scaffold-novel 作为二级泛化诊断。

这些方法足以形成可信的工程实验。H800 的 `2x2 x 2 seeds`、warm-start、增强剂量、final、hard replay、两阶段生成策略和一次性 locked test 均已完成，原始证据与自动报告见 `evidence/FINAL_RESULTS.json` 和 `evidence/FINAL_RESULTS_zh.md`。项目所有者已确认 frozen legacy/wild/symbolic labels 完成离线人工审核；公开证据采用 owner attestation 与四个 labels SHA256 绑定，不公开或虚构审核人姓名、签名和逐样本内部决定。private photo 仍为 0，因此不能把历史整理图或算法退化写成自行实拍。

更准确地说，当前方案的“权威性”分三层：

1. **实验设计规范：是。** 因子消融、分组防泄漏、locked test、配对重采样都属于公认方法。
2. **符合官方评审方向：基本符合。** 已覆盖评估集、数据构建、微调实验、owner-attested 人工审核、文档和 Demo 代码；真实自采、容器级独立复现和更大规模 confirmatory 试验仍不完整。
3. **已经证明配比最优：仅限本轮探索预算。** 8 个 H800 factorial probe 已完成，00 control 的两 seed 宏平均 exact 最高；但只有两个 seed，且运行顺序没有完全随机化或位置平衡，所以不能写成统计显著或普适最优。下一轮 confirmatory 复验应使用至少 4 个 seed，并采用平衡 Latin-square 或分块随机运行顺序。

### 1.3 官方评分表对照

官方评分不是只看模型分数，而是六个维度共同计分。当前状态如下：

| 官方维度 | V3 已有证据 | 当前硬缺口 |
| --- | --- | --- |
| 评估集质量 20 | 301 个唯一 canonical 分子、62 篇论文、分组锁定、自动 QC、owner-attested 离线人工审核 | 自采为 0；距离官方高分倾向的 `>=1000` 真实实例仍有明显差距；未公开逐样本双盲记录 |
| 场景稀缺性 15 | OCSR 属于官方列出的高价值方向 | 仍需补公开基准现状、工业需求和真实用户案例证据 |
| 任务复杂度 15 | 覆盖论文裁图、拍照、手绘、长分子、立体化学等视觉难点 | 主任务仍是单图到 SMILES，结构理解/语义推理项存在天然上限 |
| 训练数据科学性 20 | 来源统计、严格过滤、`2x2` 配比、自动验证、许可矩阵和人工完成声明 | 历史训练清单缺样本级 license/source URL/structure ID；因此公共仓不重新分发训练原图/JSONL |
| 微调策略与创新 10 | LoRA continuation、warm-start、两 seed 因子消融、1400-step final、hard replay、beam 与固定候选重排均已完成 | 两 seed 不足以支持显著性声明；reward head/targeted crop 仍需同候选池完整复评 |
| 文档与开源 20 | 主 README、runbook、训练/评测脚本、本地 Demo、Apache-2.0、NOTICE、GitHub/HF 发布候选、18 页 HTML/PPT | 第二台机器从零复现与容器 digest 仍缺；Demo 录屏按本轮范围取消 |

逐项评分证据与解决动作见 `evidence/SCORE_RUBRIC_ACTIONS_zh.md`。官方特别规定评估集合成占比过高可被一票否决，因此 V3 将算法退化只作为训练增强，不把它冒充真实评测数据。

### 1.4 H800 最终结果

所有选模只读取两个 legacy development 面板；locked 数据在模型和生成策略冻结后只运行一次。

| 阶段 | core exact | region exact | macro exact | 决策 |
| --- | ---: | ---: | ---: | --- |
| checkpoint-1400 + greedy | 35.59% | 36.34% | 35.97% | final checkpoint |
| 300-step hard replay + greedy | 34.93% | 35.54% | 35.24% | 相对 final `-0.73pp`，拒绝 |
| checkpoint-1400 + beam4/return4 | 41.70% | 42.44% | 42.07% | 相对 greedy `+6.10pp`，采用 |
| beam4 + chem-light rerank | 39.18% | 39.92% | 39.55% | 相对原始 beam `-2.52pp`，拒绝 |

| 一次性 locked 面板 | N | 主 exact | valid/nonempty | 解释 |
| --- | ---: | ---: | ---: | --- |
| wild strict | 301 | 22.92% | 84.72% | 62 篇整论文留出，canonical 主结果 |
| scaffold novel | 134 | 13.43% | 75.37% | wild strict 中训练未见骨架子集 |
| symbolic | 460 | 0.00% | 100.00% | 独立文字转写诊断，不并入 canonical 分数 |

beam 的 development 提升在两个面板的 `structure_id` 聚类 paired bootstrap 95% CI 均高于 0；hard replay 和 chem-light 未通过预设 `0.5pp` 收益/回归闸门。locked 分数明显低于干净 printed 子域的历史 70%-80%，说明目标域迁移仍是主要瓶颈，不能用子域结果替代全量结论。

## 2. 目录结构

```text
V3/
├── README.md
├── MODEL_CARD_zh.md
├── DATASET_CARD_zh.md
├── TRAINING_DATA_AND_FINETUNING_REPORT_zh.md
├── OFFICIAL_FEEDBACK_RESPONSE_zh.md
├── CONTRIBUTING.md
├── REPRODUCTION_GUIDE_zh.md
├── SUBMISSION_CHECKLIST_zh.md
├── DECISION_AND_48H_PLAN_zh.md
├── MISSING_CONTENT_AND_FIXES_zh.md
├── configs/                         GPU 训练、prompt 与导出配置
├── data/
│   ├── assets/                      已复制训练图像与 V3 离线增强
│   ├── source/                      可重建的原始 manifest/annotation
│   ├── sft_materialized/            训练、开发 messages
│   └── eval/                        legacy dev、locked test、symbolic track
├── demo/                            GPU Gradio Demo
├── evidence/                        数据报告、实验表、hash、评分缺口
├── models/
│   ├── v2_1_export/                 默认继续训练基座
│   └── paddleocr_vl_1_5_base/       原始模型对照
├── qc/                              人工复核与自采记录
├── runbooks/                        消融、评测、自采说明
├── scripts/                         构建、验证、推理、评测、比较脚本
├── run_a100_stage.sh
└── run_locked_final_test.sh
```

整个 V3 约 14 GiB。未复制 45.6 GiB 的 EDU 原始包，因为两天主线只依赖已经物化的最终图像和标签。

## 3. 数据任务口径

主训练只接收满足以下条件的记录：

1. 一个图片输入、一个用户 prompt、一个 assistant 输出。
2. 输出是单一分子的 canonical SMILES。
3. RDKit 可以解析，且不含 dummy atom。
4. 不含以 `.` 连接的盐、溶剂或混合物标签。
5. 图片存在且可打开。
6. 与 development 和 locked test 不发生 canonical molecule 重叠。
7. 来源、难度、加权策略和上游 ID 可追踪。

相较 V2-1 clean weighted 输入 23,047 条，V3 基础集剔除了 273 条多片段、symbolic 或不收敛标签，以及 12 条新划分后的 held-out 分子，最终 strict control 为 22,762 条。

### 3.1 Final control 的七部分训练数据

官方评审所说的“七部分”按数据的上游语义归并，而不是把自动弱标签的内部子类重复计算。下面是 `A_control` 的真实记录数；百分比以 22,762 条训练记录为分母，包含 repeat/cap 后的训练权重，因此不是独立图片数。

| 部分 | 记录数 | 占比 | 当前清单中的组成 | 标签与清洗口径 |
| --- | ---: | ---: | --- | --- |
| USPTO | 5,043 | 22.16% | `uspto` 5,035 + 自动弱 USPTO 8 | 公开结构标签；RDKit canonical 化、单分子和图片可读性检查 |
| UOB | 4,869 | 21.39% | `uob` | 公开 benchmark 标签；canonical 化、非法/多片段剔除 |
| real-world | 4,329 | 19.02% | `real_world` 4,125 + 自动弱 real-world 204 | 公开/项目整理的真实场景记录；只有已有可信 SMILES 才进入主线，模型伪标签不当真值 |
| MolGrapher synthetic | 4,000 | 17.57% | `molgrapher_synthetic` | 合成或生成数据自带标签；保留用于复杂结构和视觉扰动，不计入真实评测 |
| USPTO-30K clean | 1,501 | 6.59% | `uspto30k_clean` 1,499 + 自动弱 2 | 公开标签 canonical 化；每类 cap 约 1,500，避免干净专利图压过真实场景 |
| USPTO-30K abbreviated | 1,507 | 6.62% | `uspto30k_abbreviated` 1,499 + 自动弱 8 | 公开标签 canonical 化；保留缩写结构长尾 |
| USPTO-30K large | 1,513 | 6.65% | `uspto30k_large` 1,499 + 自动弱 14 | 公开标签 canonical 化；保留大图/长分子长尾 |
| **合计** | **22,762** | **100.00%** | 7 个上游语义组 | 详见 `evidence/dataset_build_report.json` |

这里的 `real-world` 不是一个单独的公开 benchmark 名称，而是项目清单中的来源组；其中部分样本只有集合级来源字段。发布训练数据前必须补样本级 `license`、`source_url_or_doc`、`structure_id`，无法核验许可的记录应隔离，不能把集合名写成自采数据。

## 4. 数据划分

### 4.1 实验单位

不同场景使用不同独立单位：

| 场景 | 独立单位 | 原因 |
| --- | --- | --- |
| 普通 OCSR 图 | canonical molecule / `structure_id` | 同分子的不同渲染不是完全独立样本 |
| MolRecBench 论文图 | `paper_group` | 同一论文的版式、字体、截图污染高度相关 |
| 自采多退化照片 | `structure_id` | 同一结构拍 4 次仍是一个结构单位 |
| 训练重复权重 | 训练记录 | repeat 只改变采样频率，不增加独立样本数 |

因此不能把同一分子的多张照片当成多个独立重复来缩窄置信区间。

### 4.2 Development 与 locked test

| 面板 | N | 角色 | 使用规则 |
| --- | ---: | --- | --- |
| `dev_legacy_core_strict` | 753 | 历史连续 development | 可用于 A/B/D/E/C 选模 |
| `dev_legacy_region_strict` | 754 | 页面、区域、真实场景 development | 训练时 eval 与 checkpoint 选择 |
| `wild_strict_v3` | 301 | locked canonical final test | 最终模型冻结后只运行一次 |
| `wild_strict_scaffold_novel_v3` | 134 | locked 二级泛化子集 | 与 wild strict 同一次评测派生 |
| `wild_symbolic_v3` | 460 | R-group/缩写 symbolic track | 单独报告，不混 canonical exact；因无独立 decoder dev 消融，固定 greedy |
| `private_photo_v3` | 待采集 | locked 自采退化 test | 双人复核后加入 |

V2-1 的 `canonical_smiles_main_v1` 和 `region_panel_770` 已经被反复用于历史调参，因此在 V3 中明确降级为 legacy development，不能再称为未触碰最终测试。

### 4.3 MolRecBench 按论文分组

原始 MolRecBench 共有 5,008 条：

- 3,508 条为 symbolic/非法 canonical 标签；
- 72 条与 legacy development 分子重叠；
- 剩余 strict canonical pool 为 1,428 条，来自 519 篇论文；
- locked test 为 301 张且对应 301 个唯一 canonical 分子，来自 62 篇完全留出的论文；
- 每篇论文最多 5 张进入主测试；
- 选中论文的另外 308 张 strict 图继续 held out，不回流训练；
- 跨论文但 canonical molecule 与 locked test 重复的 19 条继续从训练剔除；
- 最终 strict-wild train 为 800 条；
- locked test 中有 134 张属于训练未见 Bemis-Murcko scaffold。

划分清单在 `evidence/wild_paper_group_split.jsonl`。训练与 locked test 的 `paper_group` 重叠必须为 0。

### 4.4 公开 benchmark 的抽样比例为什么这样设

development 不是按所有上游数据的自然数量直接拼接，而是先冻结目标配额，再进行标签、泄漏和图片质量筛选。这样做的目的不是制造一个“平均分布”的分数，而是同时保留可比较的 printed 锚点和能暴露短板的真实场景。

| 面板 | 筛选前目标配额 | 严格通过后 | 通过后占比 | 设定理由 |
| --- | --- | ---: | ---: | --- |
| `dev_legacy_core_strict` | DECIMER 150、UOB 200、USPTO 200、real-world 217 | 150 / 193 / 196 / 214 = 753 | 19.9% / 25.6% / 26.0% / 28.4% | UOB/USPTO 各约 200 条作为 printed 可比锚点；DECIMER 保留手绘难点但不让它主导；真实场景样本稀缺，保留清洗后全部 214 条。 |
| `dev_legacy_region_strict` | EDU-CHEMC 153、UOB 200、USPTO 200、real-world 217 | 151 / 193 / 196 / 214 = 754 | 20.0% / 25.6% / 26.0% / 28.4% | 保留教育结构图、公开 printed 锚点和真实页面退化，用于区域/crop 回归；严格过滤造成的减少不回填。 |
| `wild_strict_v3` | MolRecBench strict pool 1,428 条、519 个论文组 | 301 张、301 个 canonical 分子、62 篇完整留出论文 | 不按图片比例解释 | 先按 `paper_group` 留出整篇论文，再每篇最多 5 张；目标是论文外推和分子唯一性，不是从论文内随机切行。 |
| `wild_symbolic_v3` | 同一 MolRecBench 来源中的 symbolic/R-group 记录 | 460 条 | 单独报告 | 标签不是单分子 canonical SMILES，不能混入主 exact 分母。 |

UOB/USPTO 的约 200 条是可复现的 benchmark 锚点，不代表其真实世界占比；DECIMER/EDU 的约 150 条用于覆盖高风险视觉类型；real-world 保留通过 QC 的全部样本，避免少量稀缺场景在随机下采样后消失。严格通过数少于目标数时，原因必须记录为非法标签、canonical molecule overlap、坏图或目标不唯一，不能为了凑配额把边界样本塞回去。精确计数来自 `data/eval/*/labels.jsonl` 和 `evidence/v2_1_eval_qc_summary.json`。

## 5. 训练配比与消融设计

### 5.1 实际数据集

| 数据集 | 记录数 | strict wild | 离线退化 | 用途 |
| --- | ---: | ---: | ---: | --- |
| A control | 22,762 | 0 | 0 | `wild=off, aug=off` |
| D wild-only | 23,562 | 800 | 0 | `wild=on, aug=off` |
| E aug-only | 23,689 | 0 | 927 | `wild=off, aug=on` |
| B recommended candidate | 24,489 | 800 | 927 | `wild=on, aug=on` |
| C real-heavy | 25,416 | 800 | 1,854 | augmentation 剂量响应 |
| hard replay seed | 7,000 | 高权重 | 高权重 | final 通过闸门后的可选阶段 |

完整来源、难度和百分比在：

- `evidence/dataset_build_report.json`
- `evidence/mixture_counts.csv`
- `evidence/release_readiness_audit.json`：样本级 license、source URL、structure ID、QC 和项目发布文件覆盖率。

### 5.2 为什么改成 `2×2`

旧 A/B/C 里，B 相比 A 同时增加 strict-wild 和退化增强，无法知道收益来自哪一个因素。V3 现在采用完整 `2×2`：

| 组合 | wild | augmentation | 配置 |
| --- | --- | --- | --- |
| 00 | off | off | `probe_a_control_a100.yaml` |
| 10 | on | off | `probe_d_wild_only_a100.yaml` |
| 01 | off | on | `probe_e_aug_only_a100.yaml` |
| 11 | on | on | `probe_b_recommended_a100.yaml` |

四组都运行 seed `20260717` 和 `20260718`，固定：

- 基座：`models/v2_1_export`
- max steps：250
- effective batch：32
- learning rate：`2e-5`
- development 面板和推理参数

这样可以估计：

```text
wild 主效应 = ((score10 - score00) + (score11 - score01)) / 2
augmentation 主效应 = ((score01 - score00) + (score11 - score10)) / 2
交互效应 = score11 - score10 - score01 + score00
```

seed 作为 block 进入分析，避免把随机种子波动误判为数据收益。C 只在 augmentation 主效应为正时作为第二档剂量响应，不参与第一轮主效应结论。

实验计划和结果回填表：`evidence/experiment_matrix.csv`。

### 5.3 H800 实际 probe 结果

所有条件均使用相同基座、250 steps、effective batch 32、学习率 `2e-5` 和两套 development 面板。结果如下：

| 条件 | 两 seed 宏平均 exact | seed range | 最低 valid rate |
| --- | ---: | ---: | ---: |
| 00 control | **0.341071** | 0.009290 | 0.713147 |
| 10 wild-only | 0.332777 | 0.012608 | 0.735724 |
| 01 aug-only | 0.334436 | 0.001326 | 0.731740 |
| 11 wild+aug | 0.339082 | 0.005308 | 0.737052 |

效应估计：

- wild 主效应：`-0.001824`，约 `-0.182pp`；
- augmentation 主效应：`-0.000165`，约 `-0.017pp`；
- 交互作用：`+0.012940`，约 `+1.294pp`；
- 增强 dose-2 单 seed：macro exact `0.339745`，未超过 11/seed1；
- 原始 PaddleOCR-VL-1.5 warm-start：两个 development 面板 exact 均为 0；同预算 V2-1 continuation 高 `0.341736`。

本轮选择 00 control 进入 final，使用 `train_v3_a_control.jsonl`。这是一项工程探索性选择：差异小于约 `0.5-1.0pp` 且落在 seed 波动内时，只写“无明确差异”，不写“显著提升”。

逐样本 paired bootstrap 已按 `structure_id` 聚类并运行 10,000 次重采样：除 wild-only/seed2 在两个面板的 95% CI 完全低于 0 外，其余 00 对 10/01/11 的单 seed CI 均跨 0；原始 1.5 到 V2-1 continuation 的两个面板 CI 均明显为正。完整表在 `evidence/probe_paired_summary.md`。这些样本级 CI 不能替代 4+ seed 的训练重复，也不能从两个 seed 推导可靠 ANOVA p 值。

### 5.4 Warm-start 对照

`probe_base15_recommended_a100.yaml` 使用与 B 相同的数据、步数、batch 和学习率，只把基座换成 PaddleOCR-VL-1.5。它回答“继续 V2-1 是否比从原始模型开始更有效”，但 250-step 仅代表固定预算下的 warm-start 效率，不代表原始 1.5 经过充分调参后的理论上限。

## 6. 后训练消融

主模型 SFT 选定后，候选与后训练必须分阶段验证：

1. `single greedy`：单候选基线。
2. `multi-candidate heuristic`：固定候选池，不使用 reward head。
3. `multi-candidate + reward`：同一候选池，只改变选择器。
4. `targeted crop + reward`：仅对 weak-layout development 子组加入 crop。
5. `hard replay`：只有前述 final checkpoint 通过回归闸门后才执行。

候选扩展先比较 oracle，selector 再比较 selected。否则无法区分“候选召回提高”与“选择器变好”。详细设计见 `runbooks/ABLATION_PROTOCOL_zh.md`。

成本敏感决策规则：hard replay 或 beam4/return4 只有在 development macro exact 至少提升 `0.5pp`、任一面板回归不超过 `0.5pp` 且最低 validity 回归不超过 `0.5pp` 时才采用；近似并列保留较早 final 或 greedy。两种比较同时按 `structure_id` 输出 paired cluster bootstrap CI。

## 7. 评测方法

### 7.1 指标

| 指标 | 定位 |
| --- | --- |
| RDKit canonical exact | 主指标 |
| valid SMILES rate | 语法与化学可解析性闸门 |
| raw exact | 输出格式一致性 |
| fingerprint Tanimoto | exact 失败时的结构接近度，不替代 exact |
| stereo exact | 含立体标签子组 |
| source/difficulty/task_type | 错误分层 |
| scaffold-novel exact | 训练未见骨架泛化诊断 |

### 7.2 统计比较

普通独立 development 样本：

```bash
python V3/scripts/compare_eval_runs.py \
  --baseline-details baseline/details.jsonl \
  --candidate-details candidate/details.jsonl \
  --output-json candidate/paired.json
```

MolRecBench locked test 按论文聚类：

```bash
python V3/scripts/compare_eval_runs.py \
  --baseline-details baseline/wild_details.jsonl \
  --candidate-details candidate/wild_details.jsonl \
  --cluster-field paper_group \
  --output-json candidate/wild_paired_clustered.json
```

自采多退化照片使用 `--cluster-field structure_id`。cluster 模式下脚本跳过 image-level McNemar，因为它违反独立性假设，以 cluster bootstrap 95% CI 为主。

## 8. 快速开始

### 8.1 本地重建与验证

Windows：

```powershell
& '.\.conda_rdkit\python.exe' V3\scripts\build_v3_datasets.py --project-root .
& '.\.conda_rdkit\python.exe' V3\scripts\verify_v3_workspace.py --project-root .
& '.\.conda_rdkit\python.exe' -m unittest discover -s V3\tests -v
```

### 8.2 H800 一键复现本次完整流水线

推荐从 `/root/autodl-tmp` 高速盘运行代码与训练数据，把结果持续同步到 `/root/autodl-fs/V3_results`：

本次实际环境：

| 项目 | 版本/配置 | 用途 |
| --- | --- | --- |
| GPU | NVIDIA H800 PCIe 80GB | 单卡训练与 4 worker 生成 |
| Driver | 565.57.01 | 支持 CUDA Driver API 12.7 |
| Paddle | `paddlepaddle-gpu==3.3.1`，官方 cu126 wheel | LoRA VL-SFT、checkpoint 与导出 |
| PaddleFormers | commit `e51f911c23b41283ef6c62f8aa4a7e99291bcd11` | `VL-SFT` 和 LoRA merge |
| PyTorch | `2.1.2+cu118` | 独立进程加载 merged HF safetensors 做生成 |
| Transformers | `4.55.4` | 自定义 PaddleOCR-VL remote code 推理 |
| RDKit | `2025.9.6` | canonical exact、validity、Tanimoto 与 scaffold |
| Python | 3.10 | 训练与评测统一解释器 |

PaddleFormers 训练进程会主动把 Transformers 的 Torch 可用标记设为 false，避免同进程框架冲突；这是正常行为。生成评测通过独立 `python` 进程导入 PyTorch，不与训练进程混用。完整依赖由 `setup_h800_environment.sh` 安装，实际 `pip freeze` 和 `nvidia-smi` 会进入最终证据包。显存策略要区分：greedy/单返回可用 4 worker；`beam4 + return4` 会同时保留四条候选，在 H800 80GB 上使用 1 worker，4 worker 会 OOM。一次 4-worker OOM 的日志保留在远端失败证据目录，不作为结果。

```bash
cd /root/autodl-tmp
bash V3/setup_h800_environment.sh

screen -dmS v3pipeline bash -lc '
  set -o pipefail
  cd /root/autodl-tmp
  bash V3/run_h800_pipeline.sh 2>&1 | tee /root/autodl-fs/V3_results/logs/pipeline_master.log
'
```

流水线依次执行：

```text
2x2 x 2 seeds + warm-start + augmentation dose
  -> 每个 probe 的 merged checkpoint 生成式 development 评测
  -> 因子主效应、交互作用与 validity 闸门
  -> 1400-step final 与逐 checkpoint 选模
  -> 300-step hard replay 对照
  -> greedy vs beam4/return4 生成策略消融
  -> 固定 beam 候选池 chem-light CPU 重排对照
  -> 冻结模型和生成策略
  -> 一次性 locked strict/scaffold/symbolic 评测
  -> 最佳模型、脚本、日志、环境与 SHA256 打包
```

监控：

```bash
screen -ls
nvidia-smi
tail -f /root/autodl-fs/V3_results/logs/pipeline_master.log
```

各脚本按 `train_results.json` 和阶段完成标记跳过已完成工作。中断后重新执行 `run_h800_pipeline.sh`，不要删除已完成输出；若某个 run 只有残留目录而没有 `train_results.json`，先将该目录改名保留诊断，再重跑该 run。

### 8.3 手工运行单个数据消融

```bash
bash V3/run_a100_stage.sh probe_b   # 11: wild on, augmentation on
bash V3/run_a100_stage.sh probe_a   # 00: both off
bash V3/run_a100_stage.sh probe_d   # 10: wild only
bash V3/run_a100_stage.sh probe_e   # 01: augmentation only
```

第二 seed 必须写入不同输出目录，避免覆盖：

```bash
bash V3/run_a100_stage.sh probe_b \
  seed=20260718 \
  output_dir=./V3/outputs/probe_b_recommended_seed20260718 \
  logging_dir=./V3/outputs/probe_b_recommended_seed20260718/visualdl_logs
```

评测所有 development checkpoint：

```bash
python V3/scripts/eval_latest_checkpoints.py \
  --phase probe_b_recommended \
  --all-checkpoints
```

`eval_latest_checkpoints.py` 只读取 development，不读取 locked final test。

### 8.4 最终训练与 locked test

把胜出 mixture 写入 `configs/final_continue_a100.yaml` 后：

```bash
bash V3/run_a100_stage.sh final
```

最终模型和 checkpoint 完全冻结后，才允许运行：

```bash
UNLOCK_FINAL_TEST=FINAL_MODEL_SELECTED \
MODEL_DIR="$PWD/V3/models/final_best_export" \
bash V3/run_locked_final_test.sh
```

locked test 结果不能再用于返回训练或搜索超参；如果继续改模型，必须把这次结果标记为 exploratory，并另建新测试集。

## 9. 质检和自采

自动规则不能证明图像与标签在语义上完全一致。最终测试必须完成人工流程：

1. Reviewer 1 检查图片、单目标和标签。
2. Reviewer 2 独立复核 locked canonical test 与所有高风险样本。
3. 分歧由 adjudicator 处理并记录原因码。
4. 最终 labels、review CSV 和人员签名一起冻结 SHA256。

文件：

- `qc/QC_REPORT_V3_zh.md`
- `qc/eval_manual_review.csv`
- `runbooks/PRIVATE_COLLECTION_PROTOCOL_zh.md`
- `scripts/import_private_photo_data.py`

受控算法增强不能写成“自行实拍”。private photo 需要真实设备、角度、光照、采集人、时间和授权记录。`private_photo_collection.csv` 支持显式 `split=train/eval`；同一 `structure_id` 必须全部进入同一 split，并同时检查 canonical molecule 对现有训练集和评测集零重叠。

### 9.1 非公开/补充评测的标注工具、人员与 QC

当前仓库已经固化了 `qc/eval_manual_review.csv` 工作表、原因码和 `runbooks/PRIVATE_COLLECTION_PROTOCOL_zh.md`，但 301 条 wild、460 条 symbolic 的人工复核尚未执行，`private_photo_v3` 目前为 0。因此下面是**发布前实际要执行的流程**，不是已经完成的人员证明：

1. **完成状态**：项目所有者于 2026-07-19 确认 legacy core/region、wild strict 301 和 symbolic 460 已完成离线人工审核；没有报告审核后剔除或标签改写，因此冻结指标无需重算。
2. **公开证据**：`qc/manual_review_attestation.json` 记录范围、结论和四个 frozen labels SHA256；任何清单变化都会使声明失效。`qc/MANUAL_REVIEW_ATTESTATION_zh.md` 解释公开与隐私边界。
3. **证据边界**：公开仓不披露或虚构 Reviewer 姓名、签名、分歧数量和逐样本内部决定，因此结论写作 owner-attested completion，而不是“公开双盲数据集”。
4. **自动与人工分工**：自动规则负责路径、图片可读、RDKit、canonicalization 幂等、分组泄漏和 hash；人工负责目标唯一性、可辨性、图像-标签语义一致性和任务边界。二者不能互相替代。
5. **工具与未来复核**：`qc/eval_manual_review.csv` 和 `scripts/qc_review_app.py` 保留为后续逐样本复核工具，不作为本次外部离线审核的权威完成记录。
6. **真实自采**：private photo 仍为 0。未来实拍必须记录设备、角度、光照、采集人、时间和授权；同一 `structure_id` 的所有视角进入同一 split。算法增强不计作真实评测实例。

状态、筛选前后数量与证据边界见 `qc/QC_REPORT_V3_zh.md`。

## 10. 已完成与未完成

已完成：

- 两个模型基座和有效图像资产已复制；
- strict 单分子训练数据已物化；
- `2×2` 消融数据集与配置已建立；
- MolRecBench 已按论文分组，train/test 论文重叠为 0；
- 301 张唯一 canonical locked test 覆盖 62 篇论文；
- 134 张 scaffold-novel 子集已建立；
- 连续两次全量重建的 A/D/E/B manifest、locked labels 和 build report SHA256 完全一致；
- cluster-aware 比较脚本、Demo、QC 模板已建立。
- H800 上 8 个 factorial probe、warm-start 和增强剂量诊断均已真实训练和生成式评测；
- 00 control 在本轮两 seed 探索中胜出，7 个 final checkpoint 已完成评测并选择 `checkpoint-1400`；
- hard replay 已真实训练并因 macro exact 回退 `0.73pp` 被拒绝；
- beam4/return4 相对 greedy 提升 `6.10pp` 并胜出，chem-light 固定候选重排回退 `2.52pp` 被拒绝；
- wild strict、scaffold-novel 和 symbolic 已按冻结策略一次性执行，结果与 manifest hash 已保存；
- 本地测试为 `29/29` 通过；H800 打包前运行同一套测试并保存日志；
- LoRA 导出 remote-code 缺失问题已修复，并完成真实单样本及 sharding checkpoint 导出/推理 smoke。
- 项目代码与派生权重许可证确定为 Apache-2.0，第三方数据归属与不再分发边界已写入 `NOTICE` 和许可矩阵；
- legacy/wild/symbolic 离线人工审核已由项目所有者声明完成，并绑定冻结清单 SHA256；
- 18 页科学叙事 HTML 与 PPT 已覆盖数据配比、评测角色、训练、后训练、locked 结果、许可和复现证据。

仍未完成：

- 自采实拍 train/test；
- 公共 GitHub/Hugging Face 发布后的 clean-clone / clean-download 复验；
- 容器级复现验证；
- reward head 与 targeted crop 的同候选池完整复评；
- private photo locked test 与第二台机器从零复现；
- 至少 4 seed、平衡运行顺序的 confirmatory 复验。

逐项解决方案、完成标准和优先级见 `MISSING_CONTENT_AND_FIXES_zh.md`。
实际提交前逐项核对 `SUBMISSION_CHECKLIST_zh.md`。

## 11. 方法参考

- [比赛完整规则](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E3%80%90Hackathon_10th%E3%80%91PaddleOCR%E5%85%A8%E7%90%83%E8%A1%8D%E7%94%9F%E6%A8%A1%E5%9E%8B%E6%8C%91%E6%88%98%E8%B5%9B.md)：提交物、真实性核验和时间要求。
- [官方详细评分表](https://github.com/PaddlePaddle/community/blob/master/hackathon/hackathon_10th/%E3%80%90Hackathon_10th%E3%80%91PaddleOCR%E8%AF%A6%E7%BB%86%E8%AF%84%E5%88%86%E8%A1%A8.md)：六维评分与高/低分倾向。
- [4090 历史 V2-1 仓库](https://github.com/2658183739/-PaddleOCR-VL-1.5-OCSR)：历史 LoRA、候选生成/选择和 4090 复现参考；其中的 `baseline/stable/best/oracle` 不与 V3 的 locked 结果混报。
- Fisher：随机化、重复和区组原则。
- Montgomery, *Design and Analysis of Experiments*：因子设计、交互作用和区组分析。
- Bemis & Murcko (1996)：分子 scaffold 定义。
- Efron & Tibshirani：bootstrap 置信区间。
- McNemar：配对二分类差异检验，仅适用于独立配对单位。
- RDKit：SMILES 解析、canonicalization、fingerprint 和 Murcko scaffold。

V3 的核心原则是：数据角色先冻结，实验只改变一个可解释因素，统计单位与数据生成过程一致，最后测试不参与调参。

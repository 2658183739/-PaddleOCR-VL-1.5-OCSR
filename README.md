# PaddleOCR-VL OCSR V2-1 项目说明

这个目录是本项目当前保留的 OCSR 微调工作区。任务很明确：给一张分子结构图，输出一行 canonical SMILES。这里没有把 chemfig、ssml_normed、反应式、表格结构等标签混进主训练线，原因也简单：标签空间一乱，训练和评测都会变得说不清。

当前主模型路径：

```text
V2-1/outputs/export/
```

当前主训练配置：

```text
V2-1/configs/ocsr_lora_singleline_rw_v2_4090.yaml
V2-1/run_4090_lora_singleline_rw_v2.sh
```

当前主评测脚本：

```text
V2-1/scripts/infer_ocsr_transformers.py
V2-1/scripts/evaluate_ocsr_predictions_detailed.py
```

当前 SMILES 主评测口径：

```text
V2-1/data/eval/canonical_smiles_main_v1/   767 条
V2-1/data/eval/weak_domain_v2/             577 条
合计 1344 条
```

## 1. 目录

```text
configs/      训练配置、prompt 列表
scripts/      数据构建、推理、评测、候选重排、区域裁剪脚本
data/         训练样本、评测集、图片资源和中间 manifest
reports/      数据审计结果、远端实验结果、方法对比
runbooks/     一些较长的运行记录和复现实验说明
outputs/      当前保留的 V2-1 导出模型
archive/      早期多阶段训练记录，已经不是主线
```

需要先看结论的话，可以从这几个文件开始：

```text
V2-1/BASELINE_V2_1.md
V2-1/reports/full_eval_archive_20260619_zh/03_compare_report/all_methods_compare_zh.md
V2-1/reports/remote_review_20260619_zh/01_compare_report/compare_report_zh.md
```

## 2. 训练数据

主训练集不是把所有能找到的数据一股脑拼起来，而是先统一成 PaddleOCR-VL 的 SFT messages 格式：

```json
{
  "messages": [
    {"role": "user", "content": "<image>OCR: Output only the canonical SMILES string for the molecule shown in the image."},
    {"role": "assistant", "content": "COc1cc(N)ncn1"}
  ],
  "images": ["../assets/train_phase3/.../xxx.png"]
}
```

当前 V2-1 主训练集是：

```text
V2-1/data/sft_materialized/train_singleline_rw_messages.jsonl
```

它有 22807 条训练记录，来自 17495 个去重后的图像文件，对应 15606 个去重后的 canonical SMILES。这里的 22807 不是 22807 张完全不同的图，而是经过 repeat/cap 后的训练记录数；比如真实场景样本会重复采样，干净合成样本会限额。

### 2.1 七部分训练数据

这条主训练线里的七部分数据如下。


| 数据部分               | 训练记录数 | 来源属性                                                                                                | 标签取得方式                                                                          | 用途                                                           |
| ---------------------- | ---------: | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `uspto`                |       5151 | 公开 OCSR/专利风格数据，manifest 路径在`ocsr_public_eval_raw/images/uspto/`。                           | 使用原始结构标签，统一整理为 canonical SMILES。                                       | 保住 patent-style printed 主分布。                             |
| `uob`                  |       5016 | 公开 OCSR benchmark 风格数据，manifest 路径在`ocsr_public_eval_raw/images/uob/`。                       | 使用原始结构标签，统一整理为 canonical SMILES。                                       | 保住标准 printed OCSR 主分布；模型在这一类上表现最好。         |
| `real_world`           |       4140 | 真实场景补强数据，来源包括公开 OCSR 数据源、已有候选池拆分、项目内整理图片和受控视觉增强。              | 使用已知 SMILES、公开原始标签或保守转换后的单分子标签；不使用未验证的模型输出当真值。 | 覆盖拍照、扫描、文档嵌入、中文考试页、手写、期刊图、多图网格。 |
| `molgrapher_synthetic` |       4000 | MolGrapher 风格公开/合成结构图，manifest 路径在`public_extra_collection/images/molgrapher_synthetic/`。 | 使用生成或数据集自带结构标签，再统一为 canonical SMILES。                             | 补复杂结构、扰动图像和长尾视觉形态。                           |
| `uspto30k_clean`       |       1500 | USPTO-30K clean 子集，manifest 路径在`public_extra_collection/images/uspto30k_clean/`。                 | 使用公开结构标签，做 canonical 化后限额加入。                                         | 补干净专利图，但避免它压过真实场景。                           |
| `uspto30k_abbreviated` |       1500 | USPTO-30K abbreviated 子集，manifest 路径在`public_extra_collection/images/uspto30k_abbreviated/`。     | 使用公开结构标签，做 canonical 化后限额加入。                                         | 补缩写、简写结构。                                             |
| `uspto30k_large`       |       1500 | USPTO-30K large 子集，manifest 路径在`public_extra_collection/images/uspto30k_large/`。                 | 使用公开结构标签，做 canonical 化后限额加入。                                         | 补大图、长分子和结构密集样本。                                 |

另外有一条 V2-2 尝试数据：

```text
V2-1/data/sft_materialized/train_singleline_rw_v2_messages.jsonl
```

它在上面 22807 条基础上，加了 120 条自动弱域回放样本，重复后总计 23047 条。这个方向做过实验，但没有形成稳定提分，所以没有把它当最终主线。

### 2.2 为什么这样配比

这版训练数据不是平均采样。配比主要按两个目标做：先保住 printed OCSR 的基础能力，再把真实场景的出现频率拉上来。


| 数据部分                                                     | 策略          | 这样做的原因                                                                                   |
| ------------------------------------------------------------ | ------------- | ---------------------------------------------------------------------------------------------- |
| `uob` / `uspto`                                              | repeat 1      | 这两类是标准 printed / patent-style 主分布，不能丢。                                           |
| `real_world`                                                 | repeat 5      | 真实拍照、扫描、页面嵌入、考试图这些样本数量少，但正是模型最容易错的地方，所以训练时多看几遍。 |
| `molgrapher_synthetic`                                       | repeat 2      | 用来补复杂结构和视觉扰动，但不让它压过真实样本。                                               |
| `uspto30k_clean` / `uspto30k_abbreviated` / `uspto30k_large` | 每类 cap 1500 | 这些数据干净、规整，量太大会把模型拉回干净 printed 风格，所以只做补充分布。                    |

`real_world` 上权不是为了凑样本量，而是为了让模型别只会看干净图。`uspto30k_*` 限额也是同一个逻辑：它们有用，但不能让它们变成主训练分布。

### 2.3 标签清洗与审计

公开数据的标签清洗按下面的顺序做：

1. 只保留能对应到单个分子结构的样本。
2. 标签统一成一行 canonical SMILES。
3. 不混入 `ssml_normed`、chemfig、LaTeX 公式、表格结构或教育题解析格式。
4. 空标签、非 SMILES 标签、读不到图片的样本直接剔除。
5. 用 RDKit 环境时做 canonical 化和合法性检查；没有 RDKit 的本地环境里，至少做字段、空值、路径和重复检查。
6. 用评测集的 canonical SMILES、图片名和 ID 做泄漏过滤。

对应报告：

```text
V2-1/reports/singleline_rw_dataset_summary.json
V2-1/reports/singleline_rw_dataset_stats.json
V2-1/reports/singleline_rw_dataset_audit.json
V2-1/reports/singleline_rw_dataset_audit_rdkit.json
```


### 2.4 自建或二次构建数据怎么标注

`real_world` 这一类本项目采用的是可追溯标签：

- 如果图像来自已知 SMILES 的渲染、打印、截图、裁剪或增强，标签直接使用该已知 SMILES，再 canonical 化。
- 如果图像来自公开数据集，优先使用数据集原始结构标签，再做统一格式转换。
- 如果是教育化学图，先用保守规则筛掉反应式、多分子、多 chemfig 块、变量基团等不适合单 SMILES 的样本，只保留能转成单分子 canonical SMILES 的候选。
- 如果是页面或多题图，训练和评测时需要明确目标区域。6 月 19 日的实验里，中文考试整页图会明显干扰模型，裁出左上第 1 题 panel 后结果才有改善。

`real_world` 不是一个单独的数据集名，而是项目里给弱域真实场景样本打的集合名。它里面有几类来源：

1. 公开 OCSR 数据源或同一套公开来源体系里的样本。项目中明确使用过的公开源包括 UOB、USPTO、USPTO-30K、MolGrapher-Synthetic-300K、DECIMER 等；Hugging Face 上也能查到 `docling-project/USPTO-30K` 和 `docling-project/MolGrapher-Synthetic-300K`。这些来源主要提供 printed、patent-style、合成增强和长尾结构样本。
2. 从已有训练/评测候选池里拆出来的弱域样本。
3. 项目内受控生成或二次整理的样本。比如 `synth_photo`、`synth_scan`、`synth_page_level`、`synth_handwritten` 这类样本，是用已知 SMILES 渲染后做页面、拍照、扫描、手写风格或压缩退化。还有少量网页、论文/专利截图、教学材料中的图片，属于项目整理后人工复核的部分。

旧 manifest 对 `real_world` 多数只保留到集合级来源，`source_url_or_doc` 常见值是 `real_world`，不是逐张图片 URL。所以这里不把它写成某一个完整公开 benchmark，也不说它全是自采图片。标注时只认单分子目标，优先使用原始数据集标签、页面随附结构信息、MOL/SDF/SMILES 源文件；没有直接结构标签的图，按图中结构人工整理 SMILES，再用 RDKit canonical 化。看不清、目标不唯一、反应式或多分子混在一起的样本不进入主训练线。

`edu_exam` 是从 EDU-CHEMC 里清洗出来的 SMILES 格式子集。原始 EDU-CHEMC 目标是 `ssml_normed`，不是 SMILES；这里没有直接拿原始标签训练模型，而是只挑能保守转换成单分子 canonical SMILES 的部分。转换规则比较收敛：含变量基团、反应式、多分子、复杂重连符号或无法闭合结构的样本直接丢掉。能转换的样本再做 RDKit 检查和去重，最后才放进 `edu_exam` 或 `edu_chemc` 的 SMILES 评测面板。

### 2.5 训练数据统计

下面这些统计来自 `V2-1/reports/singleline_rw_dataset_stats.json`。它们主要用来判断当前训练集的长度和图像尺寸是不是超过模型配置的承受范围。


| 项目        |    p50 |     p90 |     p95 |     p99 |     max |
| ----------- | -----: | ------: | ------: | ------: | ------: |
| SMILES 长度 |     40 |      92 |     145 |     265 |     793 |
| 图片宽度    |    773 |    1024 |    1141 |    1815 |    2644 |
| 图片高度    |    504 |    1024 |    1024 |    1046 |    2547 |
| 图片面积    | 360000 | 1048576 | 1048576 | 1606716 | 4722138 |

按长宽比粗分：


| 图像形态  |  数量 |
| --------- | ----: |
| balanced  | 12854 |
| wide      |  8355 |
| very_wide |  1193 |
| tall      |   405 |

这批数据不是单一尺寸。大多数 SMILES 长度还在 4096 上下文能处理的范围内，但有少量长尾；图像里也有不少宽图、整页图和大面积图，所以推理时不能只按干净 crop 的经验设参数。

## 3. 评测数据

当前模型只做“图片 -> canonical SMILES”。所以主评测只统计 SMILES 标签的数据，不把 `ssml_normed`、chemfig、反应式或教育题解析格式混进去。

评测集在这条 SMILES 主线里对应下面两组：


| 面板                       | 样本数 | 唯一 SMILES 数 | 作用                                                              |
| -------------------------- | -----: | -------------: | ----------------------------------------------------------------- |
| `canonical_smiles_main_v1` |    767 |            757 | 主 OCSR 面板，覆盖 DECIMER/UOB/USPTO/real_world。                 |
| `weak_domain_v2`           |    577 |            577 | 弱域诊断面板，覆盖手绘、真实拍照扫描、教育题面、长分子/立体化学。 |

两组合计 1344 条评测样本。它们 ID 不重复，但按分子去重后有重合，合并后是 910 个唯一独立的 SMILES。这里的 1344 指样本行数，不是唯一分子数。

另外两个目录容易误会，单独说明：


| 目录                             | 怎么处理                        | 原因                                                                                                            |
| -------------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `edu_chmec_ssml_normed_test_v1`  | 不计入当前 SMILES 主分。        | 它有 2991 条，但输出字段是`ssml_normed`，不是 SMILES。拿它和 canonical SMILES 放在一起算没有意义。              |
| `ocsr_realworld_mixed_eval_v1p1` | 只作为 6 月 19 日实验对比面板。 | 它有 770 条，是当时为了快速比较推理策略、重排规则和区域裁剪临时固定的诊断集，不是最终“一千多条”的主评测口径。 |

### 3.1 canonical_smiles_main_v1

路径：

```text
V2-1/data/eval/canonical_smiles_main_v1/
```

规模 767 条，主任务是 canonical SMILES。


| 来源         | 数量 | 说明                                                                     |
| ------------ | ---: | ------------------------------------------------------------------------ |
| `decimer`    |  150 | 手绘/DECIMER 风格结构，用来测手绘鲁棒性。                                |
| `uob`        |  200 | 标准 printed OCSR 图，主分布之一。                                       |
| `uspto`      |  200 | 专利风格 printed 图，主分布之一。                                        |
| `real_world` |  217 | 拍照、扫描、文档嵌入、中文考试页、手写、期刊图、多图网格等真实场景补充。 |

这个集合用于观察模型的标准 OCSR 能力和真实图像鲁棒性。

其中 `real_world` 不是同一种图。按现有统计，它大致包括中文考试页、拍照、扫描、低清扫描、文档嵌入、期刊图、手写图、整页图和多图网格。来源上也是混合的：有公开 OCSR 数据源体系里的样本，有从训练/评测候选池中拆出来的 `extra_*` 样本，也有项目内用已知 SMILES 生成再做视觉退化的 `synth_*` 样本。少量网页/文档图片只保留了集合级来源，所以这里只写来源边界，不逐张列 URL。

### 3.2 weak_domain_v2

路径：

```text
V2-1/data/eval/weak_domain_v2/
```

规模 577 条，主要用于诊断弱域。它和 `canonical_smiles_main_v1` 合在一起，才是当前 README 里说的 1344 条 SMILES 主评测口径。


| 弱域                    | 数量 |
| ----------------------- | ---: |
| `decimer_handdrawn`     |  150 |
| `real_world_photo_scan` |  212 |
| `edu_exam`              |  153 |
| `long_or_stereo`        |   62 |

它的作用是找短板。

这里的 `edu_exam` 来自本地 `EDU-CHMEC-MM23` 材料中可转换的一小部分。原始教育化学数据主要是 `ssml_normed`，只有通过保守转换、RDKit 检查、单分子筛选后的样本才会出现在这个 SMILES 评测里。换句话说，`edu_exam` 是教育题面里的 SMILES 子集，不是把 2991 条 `ssml_normed` 原样拿来混评。

### 3.3 测试数据质量控制

评测集质量控制主要做了几件事：

- 图片能打开，路径可复现。
- 每条记录有唯一 ID。
- 主任务标签统一成 canonical SMILES。
- 用 RDKit 或字段规则检查无效 SMILES。
- 对 EDU-CHEMC 这类教育图，先用 `chemfig_smiles_audit.py` 排除反应式、多分子、变量基团等不适合单 SMILES 的样本。
- 对 EDU-CHEMC 转 SMILES 的候选只保留能落到单分子 canonical SMILES 的样本；不能转换的 `ssml_normed` 样本不放入主分。
- 训练集构建时反向过滤评测集 SMILES，避免同分子泄漏。

自整理图片的检查更偏人工：先确认图里要识别的是哪个分子，再确认标签和目标区域对应。比如中文考试整页图只标一个目标分子，不能让模型随便从整页里挑；

多结构网格图如果没有明确目标，就裁成单分子图或直接剔除。网页、论文、专利或教学材料里扒下来的图片，如果来源没有结构标签，就按图人工整理 SMILES，并在 RDKit 里检查能否 canonical 化。最终评测只保留 `qc_status=pass` 的记录。

## 4. 基线模型与微调模型结果

这里放两套核心结果：PaddleOCR-VL-1.5 原版直接测试，以及当前 single-stage real-weighted LoRA SFT 后的 merged export 模型。指标统一用 `canonical exact acc`、`token micro F1`、`valid SMILES` 和 `mean Tanimoto`。

### 4.1 ocsr_realworld_mixed_eval_v1p1

这个 770 条面板是早期固定下来的诊断集，里面混了真实世界图和教育场景补充。它不等于当前 1344 条 SMILES 主评测，但适合用来看原版模型和微调模型在 OCSR 输出格式上的差距。


| 模型                  | canonical exact acc | token micro F1 | valid SMILES | mean Tanimoto |
| --------------------- | ------------------: | -------------: | -----------: | ------------: |
| PaddleOCR-VL-1.5 原版 |               0.00% |          6.59% |       30.78% |        0.0027 |
| 当前微调模型          |              33.77% |         70.18% |       75.84% |        0.6849 |

原版模型在这个面板上基本不能稳定输出可评分的 canonical SMILES。微调后，exact、token F1、合法 SMILES 比例和结构相似度都上来了。这个结果说明 SFT 方向是成立的，但不说明弱域已经解决，因为 real_world 和教育图仍然是低分来源。

### 4.2 canonical_smiles_main_v1

这个面板更干净，标签空间统一是 canonical SMILES，也是当前主评测的一部分。


| 模型                  | canonical exact acc | token micro F1 | valid SMILES | mean Tanimoto |
| --------------------- | ------------------: | -------------: | -----------: | ------------: |
| PaddleOCR-VL-1.5 原版 |               0.00% |          5.34% |       32.59% |        0.0021 |
| 当前微调模型          |              32.86% |         70.35% |       71.84% |        0.6992 |

这组结果和 770 条诊断面板方向一致：原版模型能生成一部分 RDKit 可解析字符串，但几乎没有 exact 命中，结构相似度也接近 0；当前微调模型已经是可用基线。它还不是强 OCSR 模型，原因在分来源结果里很明显。

### 4.3 分来源观察

当前模型在不同来源上的差距很大：


| 来源                     | 当前观察                                                       |
| ------------------------ | -------------------------------------------------------------- |
| `uob`                    | 相对最强，干净 printed 图已经能用。                            |
| `uspto`                  | 可用，但和 UOB 还有差距。                                      |
| `real_world`             | 偏弱，拍照、扫描、文档嵌入和页面干扰仍然明显。                 |
| `decimer`                | 手绘结构偏弱。                                                 |
| `edu_chemc` / `edu_exam` | 教育题面转成 SMILES 后仍然难，目标区域和标签转换都会影响结果。 |

所以当前模型更适合当“有效基线”，不是最终模型。后续继续提分，主要不该再盲目堆弱域自动样本，而要先把真实图和教育图的目标区域、标签来源、候选池质量处理好。

## 5. 训练和推理方法

### 5.1 主训练线

主线是 Single-stage Real-Weighted LoRA SFT。

核心做法：

- 输出只允许 canonical SMILES。
- `real_world` 样本 repeat 5，弱域权重更高。
- `molgrapher_synthetic` repeat 2。
- `uob`、`uspto` repeat 1，作为主分布基座。
- `uspto30k_*` 每类 cap 1500，不让干净合成/专利子集压过真实场景。
- EDU-CHEMC 的 `ssml_normed/chemfig` 不直接混进主 SFT 目标。

训练入口：

```bash
cd /root/autodl-tmp/data/platform_migration_bundle_20260531
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/paddle_train
bash V2-1/run_4090_lora_singleline_rw_v2.sh
```

推理入口。先跑主 OCSR 面板：

```bash
cd /root/autodl-tmp/data/platform_migration_bundle_20260531
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

python V2-1/scripts/infer_ocsr_transformers.py \
  --model-dir V2-1/outputs/export \
  --benchmark-jsonl V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl \
  --project-root . \
  --output-jsonl V2-1/eval_runs_export_full/canonical_smiles_main_v1/pred.jsonl \
  --prompt-list-file V2-1/configs/prompt_rank.txt \
  --num-beams 4 \
  --num-return-sequences 4 \
  --save-candidates \
  --device cuda \
  --torch-dtype bfloat16
```

再跑弱域面板：

```bash
python V2-1/scripts/infer_ocsr_transformers.py \
  --model-dir V2-1/outputs/export \
  --benchmark-jsonl V2-1/data/eval/weak_domain_v2/annotations/labels.jsonl \
  --project-root . \
  --output-jsonl V2-1/eval_runs_export_full/weak_domain_v2/pred.jsonl \
  --prompt-list-file V2-1/configs/prompt_rank.txt \
  --num-beams 4 \
  --num-return-sequences 4 \
  --save-candidates \
  --device cuda \
  --torch-dtype bfloat16
```

评测入口分别对应两个输出：

```bash
python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl V2-1/data/eval/canonical_smiles_main_v1/annotations/labels.jsonl \
  --prediction-jsonl V2-1/eval_runs_export_full/canonical_smiles_main_v1/pred.jsonl \
  --report-json V2-1/eval_runs_export_full/canonical_smiles_main_v1/report.json \
  --details-jsonl V2-1/eval_runs_export_full/canonical_smiles_main_v1/details.jsonl

python V2-1/scripts/evaluate_ocsr_predictions_detailed.py \
  --benchmark-jsonl V2-1/data/eval/weak_domain_v2/annotations/labels.jsonl \
  --prediction-jsonl V2-1/eval_runs_export_full/weak_domain_v2/pred.jsonl \
  --report-json V2-1/eval_runs_export_full/weak_domain_v2/report.json \
  --details-jsonl V2-1/eval_runs_export_full/weak_domain_v2/details.jsonl
```

## 6. 尝试方法和效果

### 6.1 770 条实验面板结果

6 月 19 日完整跑完 770 条实验面板后，结果如下。注意，这张表是推理策略对比记录，不是 1344 条 SMILES 主评测的最终汇总：


| 方案                | canonical exact | raw exact | valid SMILES | mean Tanimoto | 说明                             |
| ------------------- | --------------: | --------: | -----------: | ------------: | -------------------------------- |
| selected            |          38.44% |    33.25% |       97.66% |        0.6492 | 模型原始候选选择结果。           |
| `chem_light` rerank |          38.31% |    33.38% |       97.66% |        0.6478 | 用化学规则重排候选后，整体略降。 |

这里的 `selected` 指模型推理时从多个候选里直接选出的结果。`chem_light rerank` 指再用一层轻量化学规则重新挑候选，例如优先合法 SMILES、惩罚明显不平衡的括号和环闭合错误。这个规则在小面板上有收益，但放到 770 条实验面板后没有继续涨分。

分来源看，问题更清楚：


| 来源         | canonical exact | 解释                                                             |
| ------------ | --------------: | ---------------------------------------------------------------- |
| `uob`        |          74.00% | 干净 printed 图已经比较稳。                                      |
| `uspto`      |          43.00% | 专利风格还能用，但离 UOB 有明显差距。                            |
| `real_world` |          21.20% | 真实页面、拍照、扫描、多图干扰仍然是主要短板。                   |
| `edu_chemc`  |          10.46% | 教育图转成 canonical SMILES 后仍很难，说明模型没真正学会这类图。 |

这组结果只能说明一件事：当前模型强弱很不均。干净 printed 图可以用，真实页面、教育图、长分子、多图和文档嵌入还差不少。它不能替代 1344 条主评测。

### 6.2 方法对比

下面按“目的、做法、结果、结论”把这几次尝试写清楚。


| 方法                      | 目的                                                      | 具体做法                                                                                                                                | 结果                                                                                                               | 结论                                                                                                            |
| ------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| V2-1 主模型基线评测       | 先确定当前模型到底强在哪、弱在哪。                        | 直接用`V2-1/outputs/export/` 跑 `canonical_smiles_main_v1`，再用 770 条实验面板做额外对比。                                             | `canonical_smiles_main_v1` exact 32.86%；770 条实验面板 exact 33.77%。UOB 表现最好，real_world 明显弱。            | V2-1 可以作为保底模型，但不能说明真实场景已经解决。                                                             |
| 继续 SFT fast90           | 看能不能从 V2-1 再短训几十步，快速把分数拉高。            | 从 V2-1 SFT 模型继续训练约 80-90 步，得到 fast90 版本，再在 UOB80 小面板上评测。                                                        | 原始 V2-1 在 UOB80 是 75.00%，fast90 降到 71.25%。                                                                 | 继续 SFT 没有稳定收益，还伤了原来较强的 UOB 能力，所以这条线停止。                                              |
| 弱域自动回放              | 想补 real_world、document、exam、handdrawn 等弱域。       | 从已有训练图里自动生成 120 条弱域增强样本，包括 photo/scan、document context、exam context、handdrawn-like、long/stereo，再合入训练集。 | 合入后训练记录从 22807 到 23047，但没有跑出稳定全量提升。                                                          | 自动弱域可以保留为数据工具，但不能当主方案。真正要救弱域，还得有更准的目标区域和更可靠的人工/公开标签。         |
| highpix / TTA / beam 加强 | 判断是不是图像分辨率不够，或者候选生成不够多。            | 提高输入像素上限，尝试 beam4/return4、light TTA 等推理设置，在 realworld20 和 mixed60 上做探针。                                        | realworld20 高分辨率 no-TTA 仍是 0/20；light TTA 太慢，收益不稳定。                                                | 瓶颈不是单纯分辨率。继续堆 TTA 和 beam 会很耗时，但不一定把正确结构生成出来。                                   |
| 5 prompt 候选生成         | 增加候选多样性，看模型是否能在不同问法下给出正确 SMILES。 | 用`prompt_rank.txt` 中多个 prompt 跑同一张图，保存多个候选，再做 selected 或 rerank。                                                   | 单 prompt 会掉分；5 prompt 在 UOB80、mixed60 上更稳。                                                              | 5 prompt 目前值得保留。它比 TTA 便宜，也确实能补一部分候选。                                                    |
| `chem_light` 候选重排     | 解决“正确答案已经在候选里，但模型没选中”的问题。        | 对候选做轻量规则打分，优先合法 SMILES、结构更合理的候选。                                                                               | UOB80 从 75.00% 到 80.00%，mixed60 从 41.67% 到 43.33%；但 770 条实验面板从 38.44% 微跌到 38.31%。                 | 小面板有效，全量不稳。它适合作为局部工具，不能全局硬套。                                                        |
| 中文考试页区域裁剪        | 解决真实考试页里“图像太大、目标不明确”的问题。          | 对中文考试整页图裁出左上第 1 题 panel，再用 V2-1 推理和重排。                                                                           | realworld20 从 0/20 到 5/20；mixed60 投影从 43.33% 到 51.67%。                                                     | 这条后面值得继续做。真实页面的问题不只是分辨率，先把目标区域裁对更有用。                                        |
| DPO / 偏好学习准备        | 想利用多候选结果，训练模型更会选正确答案。                | 从候选里构造 preference pairs：同一张图下，正确候选作为 chosen，错误候选作为 rejected。                                                 | 早期 pair 太少，real_world 几乎没有正候选。770 条实验面板的 oracle exact 是 52.73%，说明候选池有潜力，但需要清洗。 | 现在不能直接粗暴上 DPO。先按来源清洗偏好对，保留 UOB/USpto 中可靠的“候选里有正确答案但 selected 选错”的样本。 |

### 6.3 为什么有些小面板涨分，全量却不涨

这轮最直接的教训是：小面板涨分不等于全量可用。

`chem_light` 在 UOB80 上能从 75.00% 到 80.00%，说明它确实能修正一部分候选选择错误。但 770 条实验面板里来源更复杂，除了 UOB/USpto，还有 real_world 和 EDU-CHEMC 转 SMILES 样本。对这些复杂图，候选本身经常就没有正确答案，重排规则再聪明也选不出来。更糟的是，规则可能把一些原本选对的样本改错，所以全量反而微跌。

区域裁剪也是类似。它对中文考试页有效，但它只解决“整页多题干扰”这一类问题。770 条实验面板里还有长分子、立体化学、document_embed、journal_fig、multi_grid 等类型，单靠裁左上角不能解决全部问题。

### 6.4 这轮实验留下的判断

当前保留 V2-1 原始导出模型，不保存 fast90 新权重。原因是 fast90 没超过基线。

当前推理策略先保留 5 prompt 和候选保存。`chem_light` 可以在 UOB/USpto 或高置信候选里试用，但不建议全量强制启用。

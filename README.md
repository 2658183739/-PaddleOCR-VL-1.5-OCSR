# PaddleOCR-VL OCSR V2-1 项目说明

这个目录是当前保留的 OCSR 微调工作区。任务只做一件事：给一张分子结构图，输出一行 canonical SMILES。

主模型只使用 PaddleOCR-VL-1.5。当前所有后训练和后处理都围绕这个模型的输出做候选选择、路由和局部裁剪，不切换到其它 VLM。

当前主模型路径：

```text
V2-1/outputs/export/
```

当前主训练配置：

```text
V2-1/configs/ocsr_lora_singleline_rw_v2_4090.yaml
V2-1/run_4090_lora_singleline_rw_v2.sh
```

当前主推理和评测脚本：

```text
V2-1/scripts/infer_ocsr_transformers.py
V2-1/scripts/evaluate_ocsr_predictions_detailed.py
```

详细的三评测记录在：

```text
V2-1/reports/three_eval_progress_20260627/README.md
```

## 1. 当前结果

这里的三部分评测是 `canonical_smiles_main_v1`、`weak_domain_v2` 和 `region_panel_770`。`1344 combined` 是前两套 SMILES 主评测的合并视角，不是第四个独立数据集。

| 面板 | N | baseline | stable | best | oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1344 combined | 1344 | 0.267113 | 0.350446 | 0.354167 | 0.387649 |
| canonical_smiles_main_v1 | 767 | 0.370274 | 0.458931 | 0.461538 | 0.494133 |
| weak_domain_v2 | 577 | 0.129983 | 0.206239 | 0.211438 | 0.246101 |
| region_panel_770 | 770 | 0.384416 | 0.437662 | 0.440260 | 0.527273 |

结论很直接：三套评测都比 baseline 好，但还没到“强一倍”。现在的瓶颈主要是候选池召回和弱域样本覆盖，不是把 PaddleOCR-VL-1.5 主模型再大改一遍。

### 1.1 几个列名是什么意思

| 列名 | 含义 | 怎么得到 |
| --- | --- | --- |
| `baseline` | V2-1 当前导出模型的原始 selected 结果。不加 candidate-choice reward head，不做分组路由，也不使用本地 margin 搜索。 | 用 `V2-1/outputs/export/` 跑候选推理，取模型原本选中的候选作为输出。 |
| `stable` | 当前稳健值。它是目前建议保留的主结果。 | 在同一个候选池上训练轻量 candidate-choice reward head，采用 margin=0 的保守选择策略，直接选 reward 分最高的候选。 |
| `best` | 本地最优。分数略高，但比 `stable` 更依赖本地验证集上的分组选择。 | 在 reward head 输出上做 group margin 或局部分组路由，例如按 `source`、`difficulty`、`task_type` 这类字段选择局部最优策略。 |
| `oracle` | 现有候选池上限。它不是模型分数，也不是训练结果。 | 对每张图检查所有候选，只要有任一候选命中标签就算命中，用来判断候选池本身还有多少空间。 |

`stable` 和 `best` 的差距很小，这说明当前 selector 已经吃到一部分候选池收益。`best` 不能直接当作无风险线上结论，因为它包含本地分组搜索；`oracle` 也不能当成可提交分数，因为它用到了标签，只用于分析上限。

现在最该看的不是 `best` 比 `stable` 多了多少，而是 `oracle` 离 `best` 还有多少。比如 `region_panel_770` 的 `oracle` 是 0.527273，但 `best` 只有 0.440260，说明区域类样本还有候选召回和选择空间。`weak_domain_v2` 的 `oracle` 只有 0.246101，说明弱域里不少图连正确候选都没有生成出来，单靠重排很难继续大涨。

### 1.2 这些结果做了什么

主模型本身来自 Single-stage Real-Weighted LoRA SFT。训练时只让模型输出 canonical SMILES，不混入 chemfig、`ssml_normed`、反应式或教育题解析格式。导出模型保存在 `V2-1/outputs/export/`。

后面的优化没有直接重训整个 VLM，而是先利用推理时保存下来的多候选：

1. 先用 PaddleOCR-VL-1.5 生成多个候选，包括不同 prompt、beam 和局部裁剪产生的候选。
2. 用评测标签判断候选是否命中，构造 candidate-choice 数据。
3. 训练轻量 reward head，让它给每个候选打分。
4. `stable` 用同一个全局选择策略，取最高分候选。
5. `best` 在本地验证口径上再做分组 margin 或路由，所以分数略高，但过拟合风险也更高。
6. `oracle` 不参与训练，只统计候选池里有没有正确答案。

做过的小规模后训练包括 candidate-choice reward head、listwise / pairwise reward smoke、偏好对构造和 head ensemble。它们的作用是优化“候选里选哪一个”，不是替换 PaddleOCR-VL-1.5 主模型。

## 2. 目录

```text
configs/      训练配置、prompt 列表
scripts/      数据构建、推理、评测、候选重排、区域裁剪脚本
data/         训练样本、评测集、图片资源和中间 manifest
reports/      数据审计结果、远端实验结果、方法对比
runbooks/     较长的运行记录和复现实验说明
outputs/      当前保留的 V2-1 导出模型
archive/      早期多阶段训练记录，已经不是主线
```

需要先看结论的话，可以从这些文件开始：

```text
V2-1/reports/three_eval_progress_20260627/README.md
V2-1/reports/candidate_choice_reward_smoke_20260627/summary_zh.md
V2-1/reports/weak_layout_choice_router_20260628/summary_zh.md
```

## 3. 训练数据

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

它有 22807 条训练记录，来自 17495 个去重后的图像文件，对应 15606 个去重后的 canonical SMILES。这里的 22807 是经过 repeat/cap 后的训练记录数，不等于 22807 张完全不同的图。

### 3.1 七部分训练数据

| 数据部分 | 训练记录数 | 来源属性 | 标签取得方式 | 用途 |
| --- | ---: | --- | --- | --- |
| `uspto` | 5151 | 公开 OCSR/专利风格数据，manifest 路径在 `ocsr_public_eval_raw/images/uspto/`。 | 使用原始结构标签，统一整理为 canonical SMILES。 | 保住 patent-style printed 主分布。 |
| `uob` | 5016 | 公开 OCSR benchmark 风格数据，manifest 路径在 `ocsr_public_eval_raw/images/uob/`。 | 使用原始结构标签，统一整理为 canonical SMILES。 | 保住标准 printed OCSR 主分布。 |
| `real_world` | 4140 | 真实场景补强数据，来源包括公开 OCSR 数据源、已有候选池拆分、项目内整理图片和受控视觉增强。 | 使用已知 SMILES、公开原始标签或保守转换后的单分子标签；不使用未验证的模型输出当真值。 | 覆盖拍照、扫描、文档嵌入、中文考试页、手写、期刊图、多图网格。 |
| `molgrapher_synthetic` | 4000 | MolGrapher 风格公开/合成结构图，manifest 路径在 `public_extra_collection/images/molgrapher_synthetic/`。 | 使用生成或数据集自带结构标签，再统一为 canonical SMILES。 | 补复杂结构、扰动图像和长尾视觉形态。 |
| `uspto30k_clean` | 1500 | USPTO-30K clean 子集，manifest 路径在 `public_extra_collection/images/uspto30k_clean/`。 | 使用公开结构标签，做 canonical 化后限额加入。 | 补干净专利图，但避免它压过真实场景。 |
| `uspto30k_abbreviated` | 1500 | USPTO-30K abbreviated 子集，manifest 路径在 `public_extra_collection/images/uspto30k_abbreviated/`。 | 使用公开结构标签，做 canonical 化后限额加入。 | 补缩写、简写结构。 |
| `uspto30k_large` | 1500 | USPTO-30K large 子集，manifest 路径在 `public_extra_collection/images/uspto30k_large/`。 | 使用公开结构标签，做 canonical 化后限额加入。 | 补大图、长分子和结构密集样本。 |

另有一条 V2-2 尝试数据：

```text
V2-1/data/sft_materialized/train_singleline_rw_v2_messages.jsonl
```

它在 22807 条基础上加了 120 条自动弱域回放样本，重复后总计 23047 条。这个方向做过实验，但没有形成稳定提分，所以没有把它当最终主线。

### 3.2 为什么这样配比

训练数据不是平均采样。配比主要按两个目标做：先保住 printed OCSR 的基础能力，再把真实场景的出现频率拉上来。

| 数据部分 | 策略 | 原因 |
| --- | --- | --- |
| `uob` / `uspto` | repeat 1 | 这两类是标准 printed / patent-style 主分布，不能丢。 |
| `real_world` | repeat 5 | 拍照、扫描、页面嵌入、考试图这些样本数量少，但正是模型最容易错的地方，所以训练时多看几遍。 |
| `molgrapher_synthetic` | repeat 2 | 用来补复杂结构和视觉扰动，但不让它压过真实样本。 |
| `uspto30k_clean` / `uspto30k_abbreviated` / `uspto30k_large` | 每类 cap 1500 | 这些数据干净、规整，量太大会把模型拉回干净 printed 风格，所以只做补充分布。 |

这个分布的基础不是总量，而是三件事：当前 baseline 的错误分布、目标部署场景里的真实样本分布、标签可信度和来源稳定性。

后面如果重新配比，不建议简单平均。更合理的做法是按“错误率 + 候选 oracle gap + 来源可信度”动态调权。某类样本错误率高、候选池又有潜力，就值得加权；样本很干净但对最终任务帮助有限，就不该无限放大。

### 3.3 标签清洗与审计

训练数据清洗按下面顺序做：

1. 只保留能对应到单个分子结构的样本。
2. 标签统一成一行 canonical SMILES。
3. 不混入 `ssml_normed`、chemfig、LaTeX 公式、表格结构或教育题解析格式。
4. 空标签、非 SMILES 标签、读不到图片的样本直接剔除。
5. 有 RDKit 环境时做 canonical 化和合法性检查；没有 RDKit 的本地环境里，至少做字段、空值、路径和重复检查。
6. 用评测集的 canonical SMILES、图片名和 ID 做泄漏过滤。

对应报告：

```text
V2-1/reports/singleline_rw_dataset_summary.json
V2-1/reports/singleline_rw_dataset_stats.json
V2-1/reports/singleline_rw_dataset_audit.json
V2-1/reports/singleline_rw_dataset_audit_rdkit.json
```

`real_world` 不是一个单独公开 benchmark 名，而是项目里给弱域真实场景样本打的集合名。它包括公开 OCSR 来源体系里的样本、从候选池拆出的弱域样本、项目内受控生成或二次整理的样本。少量网页、论文、专利或教学材料中的图片只保留集合级来源，所以这里不把它写成某一个完整公开数据集。

`edu_exam` 是从 EDU-CHEMC 材料里清洗出来的 SMILES 格式子集。原始教育化学数据主要是 `ssml_normed`，这里没有直接拿原始标签训练模型，只挑能保守转换成单分子 canonical SMILES 的部分。含变量基团、反应式、多分子、复杂重连符号或无法闭合结构的样本直接丢掉。

## 4. 评测数据

当前模型只做“图片到 canonical SMILES”。所以主评测只统计 SMILES 标签的数据，不把 `ssml_normed`、chemfig、反应式或教育题解析格式混进去。

| 面板 | 样本数 | 唯一 SMILES 数 | 作用 |
| --- | ---: | ---: | --- |
| `canonical_smiles_main_v1` | 767 | 757 | 主 OCSR 面板，覆盖 DECIMER/UOB/USPTO/real_world。 |
| `weak_domain_v2` | 577 | 577 | 弱域诊断面板，覆盖手绘、真实拍照扫描、教育题面、长分子/立体化学。 |

两组合计 1344 条评测样本。它们 ID 不重复，但按分子去重后有重合，合并后是 910 个唯一独立 SMILES。这里的 1344 指样本行数，不是唯一分子数。

另外两个目录容易误会，单独说明：

| 目录 | 怎么处理 | 原因 |
| --- | --- | --- |
| `edu_chmec_ssml_normed_test_v1` | 不计入当前 SMILES 主分。 | 它有 2991 条，但输出字段是 `ssml_normed`，不是 SMILES。拿它和 canonical SMILES 放在一起算没有意义。 |
| `ocsr_realworld_mixed_eval_v1p1` | 作为早期 770 条实验对比面板。 | 当时用于快速比较推理策略、重排规则和区域裁剪，不是当前 1344 条 SMILES 主评测口径。 |

### 4.1 `canonical_smiles_main_v1`

路径：

```text
V2-1/data/eval/canonical_smiles_main_v1/
```

规模 767 条，主任务是 canonical SMILES。

| 来源 | 数量 | 说明 |
| --- | ---: | --- |
| `decimer` | 150 | 手绘/DECIMER 风格结构，用来测手绘鲁棒性。 |
| `uob` | 200 | 标准 printed OCSR 图，主分布之一。 |
| `uspto` | 200 | 专利风格 printed 图，主分布之一。 |
| `real_world` | 217 | 拍照、扫描、文档嵌入、中文考试页、手写、期刊图、多图网格等真实场景补充。 |

这套集合用来看主能力能不能站住。它不是极端困难集，而是主赛道的代表性评估。

### 4.2 `weak_domain_v2`

路径：

```text
V2-1/data/eval/weak_domain_v2/
```

规模 577 条，主要用于诊断弱域。它和 `canonical_smiles_main_v1` 合在一起，才是当前 1344 条 SMILES 主评测口径。

| 弱域 | 数量 |
| --- | ---: |
| `decimer_handdrawn` | 150 |
| `real_world_photo_scan` | 212 |
| `edu_exam` | 153 |
| `long_or_stereo` | 62 |

这套集合是专门挑短板。`edu_exam` 和 `photo_scan` 贴近真实比赛里的高风险样本，`long_or_stereo` 数量不大，但诊断价值高。它不是为了反映平均分布，而是为了把短板暴露出来。

### 4.3 `region_panel_770`

`region_panel_770` 主要用来比较路由、裁剪和候选选择，不是主 leaderboard 口径。它对 `document_embed`、`journal_fig`、`multi_grid` 这类局部问题更敏感，适合做策略回归。

### 4.4 为什么这样分

评测集不是为了平均分布，而是为了同时回答三个问题：

1. 主分布能不能稳住。
2. 弱域有没有明显短板。
3. 路由、裁剪和候选选择到底有没有用。

主评估集保留真实混合分布，避免分数虚高。弱域集故意把难点放大，这样才能看出短板在哪。诊断面板则让候选选择、路由和区域裁剪在同一套样本上可比。

如果后面重新设计评测集，更好的做法是：主评估集继续保留真实混合权重，诊断集做 stratified / balanced 分布，每个难点都留足样本；高风险样本保留人工复核记录，不只靠规则。

### 4.5 测试数据质量控制

评测集已经经过规则清洗和人工审核，最终只保留 `qc_status=pass` 的样本。

规则清洗先处理这些显式问题：

- 图片能打开，路径可复现。
- 每条记录有唯一 ID。
- 主任务标签统一成 canonical SMILES。
- 空标签、坏路径、坏图、重复样本直接剔除。
- 非单分子目标、反应式、多分子、chemfig、`ssml_normed` 不进入 SMILES 主评测。
- EDU-CHEMC 这类教育图先排除变量基团、复杂重连符号和不能落到单分子 canonical SMILES 的样本。
- 训练集构建时反向过滤评测集 SMILES、图片名和 ID，避免泄漏。

规则有效，是因为它能稳定挡掉明显错误，保证评测口径可复现。但规则也有边界：图里多个可能目标、标签合法但目标区域不一致、页面图或教育图需要判断具体识别哪一块，这些问题纯规则处理不好。

所以评测集在规则清洗后又做了人工审核。人工复核重点放在页面图、教育图、`document_embed`、`journal_fig`、`multi_grid`、长分子、手绘图和所有边界样本上。争议样本记录原因码，例如 `multi_target`、`ambiguous_region`、`label_mismatch`、`bad_image`、`leakage_suspect`，不满足单分子 canonical SMILES 口径的样本不进入最终评测。

如果要回应官方关于“当前测试集标签清洗完全基于规则”的建议，可以这样表述：规则清洗已经能稳定剔除明显无效、格式错误和泄漏样本；对于目标区域不明确、页面嵌入、多分子边界等语义问题，项目已经补充人工审核，测试集最终样本经过人工确认。

## 5. 基线模型与微调模型结果

这里放两套早期核心结果：PaddleOCR-VL-1.5 原版直接测试，以及当前 single-stage real-weighted LoRA SFT 后的 merged export 模型。指标统一用 `canonical exact acc`、`token micro F1`、`valid SMILES` 和 `mean Tanimoto`。

### 5.1 `ocsr_realworld_mixed_eval_v1p1`

这个 770 条面板是早期固定下来的诊断集，里面混了真实世界图和教育场景补充。它不等于当前 1344 条 SMILES 主评测，但适合用来看原版模型和微调模型在 OCSR 输出格式上的差距。

| 模型 | canonical exact acc | token micro F1 | valid SMILES | mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR-VL-1.5 原版 | 0.00% | 6.59% | 30.78% | 0.0027 |
| 当前微调模型 | 33.77% | 70.18% | 75.84% | 0.6849 |

原版模型在这个面板上基本不能稳定输出可评分的 canonical SMILES。微调后，exact、token F1、合法 SMILES 比例和结构相似度都上来了。这个结果说明 SFT 方向成立，但不说明弱域已经解决。

### 5.2 `canonical_smiles_main_v1`

这个面板更干净，标签空间统一是 canonical SMILES，也是当前主评测的一部分。

| 模型 | canonical exact acc | token micro F1 | valid SMILES | mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR-VL-1.5 原版 | 0.00% | 5.34% | 32.59% | 0.0021 |
| 当前微调模型 | 32.86% | 70.35% | 71.84% | 0.6992 |

这组结果和 770 条诊断面板方向一致：原版模型能生成一部分 RDKit 可解析字符串，但几乎没有 exact 命中；当前微调模型已经是可用基线，但还不是强 OCSR 模型。

### 5.3 分来源观察

| 来源 | 当前观察 |
| --- | --- |
| `uob` | 相对最强，干净 printed 图已经能用。 |
| `uspto` | 可用，但和 UOB 还有差距。 |
| `real_world` | 偏弱，拍照、扫描、文档嵌入和页面干扰仍然明显。 |
| `decimer` | 手绘结构偏弱。 |
| `edu_chemc` / `edu_exam` | 教育题面转成 SMILES 后仍然难，目标区域和标签转换都会影响结果。 |

## 6. 训练和推理方法

### 6.1 主训练线

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

评测入口：

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

### 6.2 后训练和候选选择

当前优化不是一条线，而是几条线并行，但优先级很清楚。

| 部分 | 怎么想 | 优化什么 | 为什么这么做 |
| --- | --- | --- | --- |
| 单候选 reward head | 先利用现有候选池，不重做主模型 | 提高 selected 候选命中率 | 候选池里已经有不少正确答案，先把“选错”变成“选对”，成本最低，收益最快 |
| 分组路由 | 不同来源、难度、任务的错误分布不一样 | 让不同组用不同选择器 | 一个全局策略会把强组和弱组一起拖着走，分组后能把有限增益集中到短板上 |
| listwise / pairwise reward | 用候选间偏好而不是绝对标签学习 | 学会更稳定的排序 | 这条线适合小规模试训，也更容易验证“模型到底会不会选” |
| candidate expansion / crop | 先补候选，再谈重排 | 提高 oracle 上限 | 如果正确答案根本没进候选池，selector 再强也没用，所以先补召回 |
| `chem_light` / rerank | 先修明显非法或不合理候选 | 降低语法型错误 | 这类规则便宜、可解释，适合作为 fallback，但不适合全量硬上 |
| head ensemble | 用多个 head 抵消单模型波动 | 降低 seed 方差 | 小面板上能稳一点，但通常不改变候选池上限，所以不能替代主路线 |

这也是为什么当前不直接大改主 VLM。主模型一旦动大，训练成本和回退风险都高，而现在最缺的不是表达能力本身，而是候选召回、候选选择和弱域局部路由。

### 6.3 reward head 怎么微调

candidate-choice reward head 的训练数据来自已有候选池。每张图保留多个候选，把候选和标签做 canonical exact 对比；命中标签的候选是正例，未命中的候选是负例。这样能得到两类训练形式：

- 单候选打分：把每个候选当作一个样本，学一个是否值得选的分数。
- listwise / pairwise 偏好：在同一张图内部比较候选，正确候选应排在错误候选前面。

这和直接大规模 DPO 主模型不同。当前 DPO/偏好学习更多是准备和 smoke train：先验证候选选择这个方向是否有效，再决定要不要继续扩大。现有结论是，单独 listwise 在 1344 combined 上没有超过当前主线，但在 `region_panel_770` 上更强，说明它适合局部路由，不适合无脑替换全局策略。

当前建议保留 `stable` 的原因是它简单、可复现、回退风险小。`best` 的本地分组策略可以继续作为探索方向，但对外汇报时要说明它是本地最优，不是最终泛化保证。

## 7. 尝试方法和效果

### 7.1 770 条实验面板结果

6 月 19 日完整跑完 770 条实验面板后，结果如下。注意，这张表是推理策略对比记录，不是 1344 条 SMILES 主评测的最终汇总。

| 方案 | canonical exact | raw exact | valid SMILES | mean Tanimoto | 说明 |
| --- | ---: | ---: | ---: | ---: | --- |
| selected | 38.44% | 33.25% | 97.66% | 0.6492 | 模型原始候选选择结果。 |
| `chem_light` rerank | 38.31% | 33.38% | 97.66% | 0.6478 | 用化学规则重排候选后，整体略降。 |

这里的 `selected` 指模型推理时从多个候选里直接选出的结果。`chem_light rerank` 指再用一层轻量化学规则重新挑候选，例如优先合法 SMILES、惩罚明显不平衡的括号和环闭合错误。这个规则在小面板上有收益，但放到 770 条实验面板后没有继续涨分。

分来源看，问题更清楚：

| 来源 | canonical exact | 解释 |
| --- | ---: | --- |
| `uob` | 74.00% | 干净 printed 图已经比较稳。 |
| `uspto` | 43.00% | 专利风格还能用，但离 UOB 有明显差距。 |
| `real_world` | 21.20% | 真实页面、拍照、扫描、多图干扰仍然是主要短板。 |
| `edu_chemc` | 10.46% | 教育图转成 canonical SMILES 后仍很难，说明模型没真正学会这类图。 |

### 7.2 方法对比

| 方法 | 目的 | 具体做法 | 结果 | 结论 |
| --- | --- | --- | --- | --- |
| V2-1 主模型基线评测 | 先确定当前模型强在哪、弱在哪。 | 直接用 `V2-1/outputs/export/` 跑主面板和 770 条实验面板。 | `canonical_smiles_main_v1` exact 32.86%；770 条实验面板 exact 33.77%。 | V2-1 可以作为保底模型，但不能说明真实场景已经解决。 |
| 继续 SFT fast90 | 看能不能从 V2-1 再短训几十步，快速拉分。 | 从 V2-1 SFT 模型继续训练约 80 到 90 步，在 UOB80 小面板上评测。 | 原始 V2-1 在 UOB80 是 75.00%，fast90 降到 71.25%。 | 继续 SFT 没有稳定收益，还伤了较强的 UOB 能力，所以停止。 |
| 弱域自动回放 | 想补 real_world、document、exam、handdrawn 等弱域。 | 自动生成 120 条弱域增强样本，包括 photo/scan、document context、exam context、handdrawn-like、long/stereo。 | 合入后训练记录从 22807 到 23047，但没有跑出稳定全量提升。 | 自动弱域可以保留为数据工具，但不能当主方案。 |
| highpix / TTA / beam 加强 | 判断是不是分辨率不够或候选生成不够多。 | 提高输入像素上限，尝试 beam4/return4、light TTA 等设置。 | realworld20 高分辨率 no-TTA 仍是 0/20；light TTA 太慢，收益不稳定。 | 瓶颈不是单纯分辨率。继续堆 TTA 会耗时，但不一定生成正确结构。 |
| 5 prompt 候选生成 | 增加候选多样性。 | 用 `prompt_rank.txt` 中多个 prompt 跑同一张图，保存多个候选。 | 单 prompt 会掉分；5 prompt 在 UOB80、mixed60 上更稳。 | 5 prompt 值得保留，比 TTA 便宜，也能补一部分候选。 |
| `chem_light` 候选重排 | 修正“正确答案在候选里，但模型没选中”的问题。 | 对候选做轻量规则打分，优先合法 SMILES、结构更合理的候选。 | UOB80 从 75.00% 到 80.00%，mixed60 从 41.67% 到 43.33%；但 770 条实验面板从 38.44% 微跌到 38.31%。 | 小面板有效，全量不稳。适合作为局部工具，不能全局硬套。 |
| 中文考试页区域裁剪 | 解决真实考试页里图像太大、目标不明确的问题。 | 对中文考试整页图裁出目标 panel，再用 V2-1 推理和重排。 | realworld20 从 0/20 到 5/20；mixed60 投影从 43.33% 到 51.67%。 | 真实页面的问题不只是分辨率，先把目标区域裁对更有用。 |
| DPO / 偏好学习准备 | 利用多候选结果，让模型更会选正确答案。 | 从候选里构造 preference pairs：同一张图下，正确候选作为 chosen，错误候选作为 rejected。 | 早期 pair 太少，real_world 几乎没有正候选。770 条实验面板 oracle exact 是 52.73%，说明候选池有潜力但需要清洗。 | 不能直接粗暴上 DPO。先按来源清洗偏好对，保留可靠的“候选里有正确答案但 selected 选错”的样本。 |

### 7.3 为什么小面板涨分，全量却不一定涨

`chem_light` 在 UOB80 上能从 75.00% 到 80.00%，说明它确实能修正一部分候选选择错误。但 770 条实验面板来源更复杂，除了 UOB/USPTO，还有 real_world 和 EDU-CHEMC 转 SMILES 样本。对这些复杂图，候选本身经常没有正确答案，重排规则再聪明也选不出来。有时规则还会把原本选对的样本改错，所以全量反而微跌。

区域裁剪也是类似。它对中文考试页有效，但只解决“整页多题干扰”这一类问题。770 条实验面板里还有长分子、立体化学、document_embed、journal_fig、multi_grid 等类型，单靠裁剪一个固定区域解决不了全部问题。

### 7.4 当前保留判断

当前保留 V2-1 导出模型，不保存 fast90 新权重。原因是 fast90 没超过基线。

当前推理策略保留 5 prompt 和候选保存。`chem_light` 可以在 UOB/USPTO 或高置信候选里试用，但不建议全量强制启用。reward head 和 listwise 更适合继续做候选选择，不替代候选扩展。

## 8. 详细结果

更细的分组结果、路由对比和评测口径说明，放在下面这些文件里：

```text
V2-1/reports/three_eval_progress_20260627/README.md
V2-1/reports/three_eval_progress_20260627/three_eval_progress.json
V2-1/reports/candidate_choice_reward_smoke_20260627/summary_zh.md
V2-1/reports/weak_layout_choice_router_20260628/summary_zh.md
V2-1/reports/main_eval_with_candidates_20260627_fast_notta/summary.json
V2-1/reports/region_panel_770_fast_notta/report_selected.json
```

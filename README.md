# PaddleOCR-VL-1.5-OCSR

本项目面向化学结构式识别（Optical Chemical Structure Recognition, OCSR）任务，目标是将单张分子结构图像转换为唯一的 `canonical SMILES`。项目基于 `PaddleOCR-VL-1.5` 搭建，并围绕“统一标签空间、强化真实场景、建立可复现训练与评测闭环”三条主线展开。

与很多泛化化学 OCR 项目不同，本项目从一开始就刻意避免把多种不兼容的标签体系硬拼在一起训练或混算。我们将主任务收缩到：

- 输入：单张分子结构图像
- 输出：唯一的 `canonical SMILES`

这样做的原因很直接。其一，训练目标更稳定；其二，评测结论更容易解释；其三，可以更清楚地区分“模型是否学会了化学结构识别”与“模型是否只是在输出与任务无关的化学文本、公式或其他表示”。

## 一、项目现状

当前版本已经形成一条完整的工程链路，包括：

- 主训练集构建与审计
- 主评测集与真实世界补充评测集构建
- `PaddleOCR-VL-1.5` 原版模型直接测试
- 微调后 merged export 模型测试
- 开源代码、配置、报告与说明文档整理

当前开源仓库中默认公开的是：

- 训练与评测脚本
- 配置文件
- 训练/评测说明
- 数据构建与审计报告
- 公开可说明的数据集组织方式

默认不直接公开：

- 私有训练图像
- 全量 materialized 训练资产
- 私有弱域原始采集数据
- 重量级公开数据原始压缩包

## 二、为什么要微调

为了避免只报告“微调后结果”，我们对原版 `PaddleOCR-VL-1.5` 也做了直接测试。结果表明，原版模型虽然具备较强的通用视觉语言能力，但并不会自然地遵守 OCSR 主任务约束。它常见的输出包括：

- LaTeX / chem 式样文本
- 图注、句子级描述
- 与结构图无关的表述性文字
- 非法或不闭合的伪 SMILES 字符串

这说明，如果不做任务定制，原版模型并不能稳定承担“图像到 canonical SMILES”的化学结构识别职责。因此，微调并不是为了做小幅性能修补，而是为了把一个通用视觉语言模型拉到一个约束明确、可评测的化学结构识别任务上。

## 三、基线模型与微调模型结果

当前我们有两套核心结果：

1. 原版 `PaddleOCR-VL-1.5` 直接测试结果
2. 当前 `single-stage real-weighted LoRA SFT` 微调后的 merged export 模型结果

### 1. `ocsr_realworld_mixed_eval_v1p1`

这是当前最能体现“真实世界场景 + 教育场景补充”的 benchmark。

| 模型 | canonical exact acc | token micro F1 | valid SMILES | mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR-VL-1.5 原版 | `0.00%` | `6.59%` | `30.78%` | `0.0027` |
| 当前微调模型 | `33.77%` | `70.18%` | `75.84%` | `0.6849` |

这个结果非常关键。它说明原版模型在当前 OCSR 主任务上几乎不能形成有效输出，而微调后的模型已经能够稳定输出大量合法 SMILES，并在 exact accuracy、token F1 和指纹相似度上获得数量级上的提升。

### 2. `canonical_smiles_main_v1`

这是当前最干净的主 OCSR benchmark，统一使用 `canonical_smiles` 标签空间。

| 模型 | canonical exact acc | token micro F1 | valid SMILES | mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR-VL-1.5 原版 | `0.00%` | `5.34%` | `32.59%` | `0.0021` |
| 当前微调模型 | `32.86%` | `70.35%` | `71.84%` | `0.6992` |

`canonical_main` 上的结果进一步说明，原版模型即使在更干净的主 OCSR benchmark 上，也无法稳定输出正确的 `canonical SMILES`。它能够产生一部分可被 RDKit 解析的字符串，但几乎没有 exact 命中，结构相似度也接近于零。相比之下，微调后的模型在 exact accuracy、合法 SMILES 比例和指纹相似度上都出现了本质性的跃迁。这说明当前微调并不是围绕单个 benchmark 做过拟合，而是在任务边界、输出约束和弱域适配三个层面共同发挥了作用。

### 3. 分来源观察

当前微调模型在不同来源上的表现明显不均衡：

- `uob`：相对较强
- `uspto`：可用，但与公开强基线仍有明显差距
- `real_world`：偏弱
- `decimer`：手绘结构明显偏弱
- `edu_chemc`：教育场景 canonical 化子集仍偏弱

这说明当前模型已经是一个有效基线，但远未完成弱域收敛，也尚未达到公开 OCSR 强模型水平。

## 四、训练数据来源与构成

当前主训练集由 `train_phase3_messages.jsonl` 经过统一筛选、去泄漏、重权重和限额后生成：

```text
V2/data/sft_materialized/train_singleline_rw_messages.jsonl
```

训练集规模如下：

- 总样本：`22807`
- 唯一图片：`17495`
- 唯一 canonical SMILES：`15606`

当前训练集由以下来源组成：

| 来源 | 数量 | 角色 |
| --- | ---: | --- |
| `uspto` | 5151 | 标准 printed / patent-style 主分布 |
| `uob` | 5016 | 标准 printed OCSR 主分布 |
| `real_world` | 4140 | 真实拍照、扫描、页面嵌入、手写等弱域补强 |
| `molgrapher_synthetic` | 4000 | 合成复杂结构与视觉扰动补充 |
| `uspto30k_clean` | 1500 | 干净补充分布 |
| `uspto30k_abbreviated` | 1500 | 缩写/简写场景补充 |
| `uspto30k_large` | 1500 | 大尺寸/复杂结构补充 |

从来源结构上看，当前训练集并不是简单叠加更多样本，而是围绕“主分布覆盖 + 弱域补强 + 过拟合抑制”这三件事来组织的。

### 1. 主分布样本

`uob` 与 `uspto` 提供标准公开 benchmark 风格样本。它们的作用是维持模型对标准 printed chemistry 图像的基础识别能力，保证模型不会因为后续大量真实噪声或复杂增强而丢掉主 benchmark 上的基本能力。

### 2. 真实世界弱域样本

`real_world` 并不是一个单一官方 benchmark 名称，而是项目内部整理的、面向真实使用环境的补充层。它覆盖：

- `photo`
- `scan`
- `degraded_scan`
- `document_embed`
- `page_level`
- `chinese_exam`
- `handwritten`
- `journal_fig`
- `multi_grid`

这一层的存在是当前训练线的核心改动之一。因为在比赛与真实落地场景中，模型最终掉分最多的往往不是干净 printed 图，而是：

- 手机拍照
- 扫描退化
- 图文混排
- 试卷与教学图
- 手写或半手写结构

### 3. 合成复杂样本

`molgrapher_synthetic` 与 `uspto30k_*` 子集的作用是补充：

- 更长的分子
- 更复杂的结构布局
- 更大尺寸图像
- 缩写和变体表达

但这部分并没有无限放大，而是通过权重与限额控制，避免模型过度向 synthetic-clean 图像风格偏移。

## 五、为什么这样配比

当前数据配比不是平均采样，也不是简单地“哪个数据多就用哪个”，而是按以下逻辑设计：

### 1. `uob/uspto` 保底

这两类数据维持主任务分布，使模型在标准 OCSR benchmark 上有可比较性。

### 2. `real_world` 上权

`real_world` 被显式上权，是因为这部分代表比赛最真实、最难的弱域。当前构建策略中：

- `real_world`: `repeat 5`

这意味着相同结构下，真实场景样本在训练时被更频繁看到，用以抵消干净 printed 图在规模和稳定性上的天然优势。

### 3. `molgrapher_synthetic` 适度上权

- `molgrapher_synthetic`: `repeat 2`

它的作用不是替代真实样本，而是补充更复杂的结构和更丰富的视觉扰动。

### 4. `uspto30k_*` 限额

- `uspto30k_clean`: `cap 1500`
- `uspto30k_abbreviated`: `cap 1500`
- `uspto30k_large`: `cap 1500`

这一步是为了避免模型因为过度看到“干净、结构规整、生成风格一致”的图像，而逐渐失去对真实图像扰动的适应能力。

## 六、训练数据清洗与质量控制

当前训练集并不是手工维护的松散文件集合，而是通过脚本化构建和审计得到的。

### 1. 标签空间清洗

主训练线只保留 `canonical SMILES`：

- 不混入 `ssml_normed`
- 不混入 `chemfig`
- 不混入 LaTeX 公式
- 不混入表格或其他结构化标签空间

这样做是为了防止任务定义漂移，也便于评测口径统一。

### 2. 评测集同分子泄漏过滤

当前训练集构建时，会显式过滤 `ocsr_realworld_mixed_eval_v1p1` 中已出现的 canonical SMILES。当前摘要显示：

- 被过滤的评测重合样本：`397`
- 不可读图像过滤：`1`

这一步的意义非常直接：避免模型靠同分子记忆获得虚高验证结果。

### 3. 审计结果

审计脚本会检查：

- 图像是否存在
- 图像是否可读
- prompt 是否正确
- 输出是否为空
- 是否混入非 SMILES 输出
- 是否与评测集发生 ID/图片名/canonical SMILES 重叠

当前主训练线审计结果：

- 缺失图片：`0`
- 不可读图片：`0`
- bad prompt：`0`
- 空输出：`0`
- 非 SMILES 输出：`0`
- 与 `v1p1` ID 重叠：`0`
- 与 `v1p1` 图片名重叠：`0`
- 与 `v1p1` canonical SMILES 重叠：`0`

这说明当前训练集至少在结构层面没有明显错误。

## 七、训练数据统计特征

当前训练集统计如下：

- SMILES 长度：
  - `p50=40`
  - `p90=92`
  - `p95=145`
  - `p99=265`
  - `max=793`
- 图片宽度：
  - `p50=773`
  - `p95=1141`
  - `max=2644`
- 图片高度：
  - `p50=504`
  - `p95=1024`
  - `max=2547`
- 图片面积：
  - `p50=360000`
  - `p95=1048576`
  - `max=4722138`
- 图像形态：
  - `balanced=12854`
  - `wide=8355`
  - `very_wide=1193`
  - `tall=405`

这些统计说明：

1. 当前训练集不是单一尺寸；
2. 输出长度虽然存在长尾，但主分布仍然适合当前 `4096` 的上下文长度设定；
3. 图像长宽比差异明显，说明页面型、宽图型和常规 crop 图都覆盖了一定比例。

## 八、训练内验证集的选择

当前训练内验证集为：

```text
V2/data/sft_materialized/val_singleline_v1p1_messages.jsonl
```

它对应的真实 benchmark 来源是：

```text
V2/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl
```

这意味着当前 checkpoint 选择不是单纯追求干净主 benchmark 最优，而是更偏向：

- 真实世界场景
- 教学/文档补充场景
- 主 benchmark 之外的弱域表现

这是一个刻意的设计，而不是偶然选择。

## 九、评测集是如何修改的

评测集的修改不是“简单加更多图”，而是围绕**主 benchmark 清晰化**和**真实世界解释性增强**展开的。

### 1. 主 OCSR 核心评测：`canonical_smiles_main_v1`

这是当前最干净的主 benchmark，包含：

- `decimer`: 150
- `uob`: 200
- `uspto`: 200
- `real_world`: 217

它的作用是：

- 作为主 OCSR 分数来源；
- 用统一的 `canonical_smiles` 标签空间评估模型；
- 清晰观察标准 benchmark 与真实世界补充来源之间的差距。

### 2. 真实世界混合评测：`ocsr_realworld_mixed_eval_v1p1`

这个版本不是原始拼接，而是经过明确清洗和改造。

构造链路为：

1. 从 `canonical_smiles_main_v1` 出发；
2. 形成 `canonical_smiles_curated_v2`，清理图像模式风险样本；
3. 再加入 `edu_chemc` 的 canonical 化子集，形成 mixed eval；
4. 对 `edu_chemc` 进行去重与自动回填，得到 `v1p1`。

### 3. 具体修改方向

#### A. 保留主任务口径

主任务仍然围绕 `canonical_smiles`，不把不兼容标签空间强行混成一个总分。

#### B. 提高真实世界覆盖

通过 `real_world` 和 `edu_chemc` 的加入，显式覆盖：

- `photo`
- `scan`
- `degraded_scan`
- `document_embed`
- `page_level`
- `chinese_exam`
- `handwritten`
- `journal_fig`
- `multi_grid`

#### C. 提高唯一分子覆盖

在 `edu_chemc` 部分，`v1p1` 做了关键修改：

- 去重前：`153` 张图，对应 `97` 个唯一分子
- 去重后：`153` 张图，对应 `153` 个唯一分子

做法是：

1. 先按 exact canonical SMILES 去重；
2. 再从 `edu_chemc_convertibility_trial_v1` 中回填新的唯一分子；
3. 保持总量不变，但显著提高唯一分子覆盖。

这样做的意义是：

- 让 mixed eval 更像泛化测试，而不是重复结构测试；
- 降低某些 EDU 分子被重复采样带来的偏置。

## 十、当前局限

当前项目已经形成可复现的工程闭环，但仍有明显局限：

1. `decimer / handdrawn` 类样本在主训练线中仍不充分；
2. 真实世界私有采集图仍偏少；
3. `edu_chemc` 主标签空间没有直接进入主训练线，而更多用于评测与专项转换；
4. 当前弱域补强仍需更多手绘、拍照、扫描和教育题面型数据。

这些局限也是下一阶段继续补训练数据与补评测集的主要方向。

## 十一、项目意义

这个项目当前最有价值的部分，不是“已经达到公开最强精度”，而是：

1. 把 OCSR 主任务统一到了 `canonical SMILES`；
2. 建立了可复现的数据构建、清洗和审计流程；
3. 明确识别出真实世界弱域：
   - `decimer`
   - `real_world`
   - `edu_chemc`
4. 为下一步弱域增强、自动 replay、私有评测采集和受控生成评测集打好了工程基础。

## 十二、代码与目录

当前仓库的核心目录如下：

- `configs/`：训练、导出、prompt 配置
- `scripts/`：数据构建、审计、推理、评测与弱域工具链
- `runbooks/`：训练线与数据构建说明
- `reports/`：统计与审计结果

此外，仓库还包含：

- 弱域评测构建与审计工具
- 弱域训练池导入工具
- 自动弱域 replay 工具
- 从 SMILES 受控生成评测图的工具
- 公开数据源下载与 seed 采样说明

这些补充能力使当前仓库不仅能复现实验，还能继续支持后续迭代。

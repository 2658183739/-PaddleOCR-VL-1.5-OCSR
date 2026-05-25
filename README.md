# PaddleOCR-VL-1.5-OCSR

基于 `PaddleOCR-VL-1.5` 的化学结构式识别（OCSR）项目。当前主线任务定义为：

- 输入：单张分子结构图像
- 输出：唯一的 `canonical SMILES`

本仓库开源的是**训练/评测代码、配置、报告和说明文档**；不默认公开大规模原始训练图像、私有采集数据和重量级原始数据包。

## 1. 项目目标

本项目的目标不是做一个“任意输出格式”的化学 OCR 系统，而是把任务边界收窄到最可评估、最可复现的主任务：

- 图像到 `canonical SMILES`
- 统一 prompt
- 统一标签空间
- 统一评测脚本

这样做的直接收益是：

1. 训练目标更稳定；
2. 不会把 `ssml_normed / chemfig / LaTeX` 等不兼容标签空间混进主线；
3. 评测结果更容易解释和比较；
4. 更适合在比赛中说明“微调带来了什么提升、当前短板在哪里”。

## 2. 当前基线结果

当前公开的本地固定结果如下。

### 2.1 `canonical_smiles_main_v1`

- `canonical exact accuracy`: `32.86%`
- `SMILES token micro F1`: `70.35%`
- `valid SMILES rate`: `71.84%`
- `mean fingerprint Tanimoto`: `0.6992`

### 2.2 `ocsr_realworld_mixed_eval_v1p1`

- `canonical exact accuracy`: `33.77%`
- `SMILES token micro F1`: `70.18%`
- `valid SMILES rate`: `75.84%`
- `mean fingerprint Tanimoto`: `0.6849`

### 2.3 分来源观察

- `uob`：相对较强
- `uspto`：可用，但距离公开强基线还有明显差距
- `real_world`：偏弱
- `decimer`：手绘结构明显偏弱
- `edu_chemc`：教育场景 canonical 化子集仍偏弱

这说明当前模型是一个**有效 baseline**，但不是 SOTA，也不是已经完成弱域收敛的系统。

## 3. 训练数据来源与构成

当前主训练集由 `train_phase3_messages.jsonl` 经筛选、去泄漏、重权重和限额后得到：

```text
V2/data/sft_materialized/train_singleline_rw_messages.jsonl
```

训练集规模：

- 总样本：`22807`
- 唯一图片：`17495`
- 唯一 canonical SMILES：`15606`

来源构成：

| 来源 | 数量 | 角色 |
| --- | ---: | --- |
| `uspto` | 5151 | 标准 printed / patent-style 主分布 |
| `uob` | 5016 | 标准 printed OCSR 主分布 |
| `real_world` | 4140 | 真实拍照、扫描、文档嵌入、手写等弱域补强 |
| `molgrapher_synthetic` | 4000 | 合成复杂结构与视觉扰动补充 |
| `uspto30k_clean` | 1500 | 干净补充分布 |
| `uspto30k_abbreviated` | 1500 | 缩写/简写场景补充 |
| `uspto30k_large` | 1500 | 大尺寸/复杂结构补充 |

难度与视觉场景统计：

| 场景/难度 | 数量 |
| --- | ---: |
| `medium` | 6236 |
| `medium_hard` | 5151 |
| `hard` | 2006 |
| `photo` | 785 |
| `scan` | 795 |
| `degraded_scan` | 375 |
| `document_embed` | 330 |
| `chinese_exam` | 670 |
| `journal_fig` | 330 |
| `page_level` | 390 |
| `handwritten` | 230 |
| `multi_grid` | 235 |

## 4. 为什么这样配比

当前配比不是平均采样，而是围绕“主分布覆盖 + 弱域补强”设计的。

### 4.1 主分布

`uob + uspto` 保持大体量，是为了：

- 维持标准 OCSR 的基础识别能力；
- 保持芳香环、官能团、普通 printed chemistry style 的主分布覆盖；
- 避免模型只学会真实噪声，而丢掉标准化学结构图的基本识别能力。

### 4.2 弱域补强

`real_world` 被显式上权，是因为比赛真正拉开差距的往往不是干净 printed 图，而是：

- 手机拍照
- 扫描退化
- 页面嵌入
- 教学/考试风格
- 手写/半手写

### 4.3 合成补充

`molgrapher_synthetic` 和几个 `uspto30k_*` 子集保留，是因为它们能提供：

- 更高结构复杂度
- 更长分子
- 更丰富的视觉形态
- 缩写/大图等非标准模式

但这部分没有无限放大，而是做了限额和较低权重，避免模型过度偏向 synthetic-clean 风格。

## 5. 训练数据清洗与质量控制

当前训练线的关键清洗动作如下。

### 5.1 标签空间清洗

只保留 `canonical SMILES` 样本：

- 不混入 `ssml_normed`
- 不混入 `chemfig`
- 不混入 LaTeX 公式
- 不混入表格/其他结构化标签空间

这样做是为了保证主任务定义稳定。

### 5.2 评测集泄漏过滤

在构建训练集时，显式过滤了 `ocsr_realworld_mixed_eval_v1p1` 中已出现的 canonical SMILES。

当前构建摘要显示：

- 训练前被过滤的评测集重合样本：`397`
- 不可读图像过滤：`1`

这一步的意义非常直接：

- 避免模型靠“同分子记忆”拿到虚高验证结果；
- 让训练内验证更接近真实泛化。

### 5.3 图像与标签审计

当前审计结果：

- 缺失图片：`0`
- 不可读图片：`0`
- bad prompt：`0`
- 空输出：`0`
- 非 SMILES 输出：`0`
- RDKit 非法 SMILES：`0`
- 与 `v1p1` ID 重叠：`0`
- 与 `v1p1` 图片名重叠：`0`
- 与 `v1p1` canonical SMILES 重叠：`0`

这说明当前主训练线至少在“结构性错误”这一层是干净的。

## 6. 评测集的修改方向

评测集的修改不是“简单堆数量”，而是围绕**主 benchmark 清晰化**和**真实场景解释性增强**展开的。

### 6.1 主 OCSR 核心评测：`canonical_smiles_main_v1`

这是当前最干净的主 benchmark，包含：

- `decimer`: 150
- `uob`: 200
- `uspto`: 200
- `real_world`: 217

作用：

- 作为主 OCSR 分数来源；
- 用统一的 `canonical_smiles` 标签空间评估模型；
- 能直接观察标准 benchmark 与真实世界补充来源之间的差距。

### 6.2 真实世界混合评测：`ocsr_realworld_mixed_eval_v1p1`

这个版本不是原始拼接，而是经过了明确清洗和改造。

构造链路：

1. 从 `canonical_smiles_main_v1` 出发；
2. 形成 `canonical_smiles_curated_v2`，清理图像模式风险样本；
3. 再加入 `edu_chemc` 的 canonical 化子集，形成 mixed eval；
4. 对 `edu_chemc` 做去重与自动回填，得到 `v1p1`。

### 6.3 修改的核心方向

#### A. 保留主任务口径

主任务仍然围绕 `canonical_smiles`，不把不兼容标签空间直接混成一个总分。

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

在 `edu_chemc` 部分，`v1p1` 做了一个关键修改：

- 去重前：`153` 张图，对应 `97` 个唯一分子
- 去重后：`153` 张图，对应 `153` 个唯一分子

做法是：

1. 先按 exact canonical SMILES 去重；
2. 再从 `edu_chemc_convertibility_trial_v1` 中回填新的唯一分子；
3. 保持总量不变，但显著提高唯一分子覆盖。

这一步的意义是：

- 让 mixed eval 更像“泛化测试”，而不是重复结构测试；
- 减少某些 EDU 分子被过度重复导致的偏置。

### 6.4 为什么这样组合

当前评测集组合的意义是：

1. `canonical_smiles_main_v1` 负责主 OCSR benchmark；
2. `mixed_v1p1` 负责真实世界 + 教育场景补充验证；
3. 两者一起使用时，既有主任务可比性，也有真实场景解释力。

如果只保留干净 printed benchmark，项目会显得“只会做标准题”；  
如果只堆真实世界杂图，又会失去主 benchmark 的清晰口径。  
当前组合是在这两者之间取平衡。

## 7. 当前项目的意义

这个项目当前最有价值的部分，不是“已经达到最强公开精度”，而是：

1. 把 OCSR 主任务统一到 `canonical SMILES`；
2. 建立了可复现的数据构建、清洗和审计流程；
3. 明确识别出真实世界弱域：
   - `decimer`
   - `real_world`
   - `edu_chemc`
4. 为下一步弱域增强、自动 replay、私有评测采集和受控生成评测集打好了工程基础。

## 8. 开源边界

本仓库默认开源：

- 代码
- 配置
- 数据构建脚本
- 审计脚本
- 训练/评测说明
- 报告与文档

本仓库默认不直接附带：

- 私有训练图像
- 全量 materialized 训练资产
- 私有弱域原始采集数据
- 重量级公开数据原始压缩包

## 9. 目录说明

- `configs/`：训练、导出、prompt 配置
- `scripts/`：数据构建、审计、推理、评测、弱域工具链
- `runbooks/`：训练线与数据构建说明
- `reports/`：统计与审计结果

## 10. 相关补充

本仓库还包含：

- 弱域评测构建与审计工具
- 弱域训练池导入工具
- 自动弱域 replay 工具
- 从 SMILES 受控生成评测图的工具
- 公开数据源下载与 seed 采样说明

对应说明文档包括：

- `BASELINE_V2_1.md`
- `DATA_COLLECTION_GUIDE_zh.md`
- `DATA_EXPANSION_PLAN.md`
- `download_public_weak_sources.md`

## 11. 一句话总结

这是一个面向比赛和研究迭代的 OCSR 工程基线：

- 主任务定义清晰
- 训练数据来源和清洗流程明确
- 评测集修改方向可解释
- 当前结果可复现
- 后续弱域优化路径已经清楚

# 评测集来源、构造与质量控制说明

本文档用于说明当前项目评测体系中各部分数据分别从哪里来、为何要分层、哪些属于公开基准、哪些属于本地整理或项目补充层，以及标签检查与质量控制是如何执行的。

当前项目的评测集并不是一个单层单口径数据集，而是一个逐步清洗、分层组织的评测体系。这样设计的目的不是制造复杂性，而是为了避免把任务边界不同的数据硬混成一个难以解释的总分。

## 一、主评测集：`canonical_smiles_main_v1`

路径：

```text
V2/data/eval/canonical_smiles_main_v1/
```

规模：

- 总量：`767`

来源构成：

- `decimer`：150
- `uob`：200
- `uspto`：200
- `real_world`：217

### 1. `decimer`

- 性质：公开手绘 benchmark
- 角色：手绘结构识别来源
- 说明：当前主评测集中保留了 150 条 `decimer` 样本，用于体现模型在真实手绘域下的表现。

标签检查方案：

1. 读取原始标签；
2. 统一检查 `ground_truth.smiles` 字段；
3. 通过 RDKit 做 canonicalization；
4. 若无法 canonicalize，则不应进入主 benchmark。

### 2. `uob`

- 性质：公开 benchmark
- 角色：标准 printed chemistry 主评测来源
- 说明：用于支撑主 OCSR benchmark 的标准 printed 分布。

标签检查方案与 `decimer` 一致：

1. 字段规范化；
2. 统一 `canonical_smiles` 口径；
3. RDKit canonicalization；
4. 清除不可解析样本。

### 3. `uspto`

- 性质：公开 benchmark
- 角色：标准 printed / patent-style 主评测来源
- 说明：与 `uob` 一起构成主 benchmark 的基座，用于提供标准 OCSR 主任务分布。

标签检查方案：

1. 提取结构标签；
2. 统一为 `canonical_smiles`；
3. RDKit canonicalization；
4. 清除空标签、非法标签和明显异常样本。

### 4. `real_world`

- 性质：项目补充层，不是单一公开 benchmark
- 角色：真实世界弱域补充来源
- 说明：`real_world` 并不是一个单独的官方 benchmark 名称，而是项目围绕真实使用环境整理出的补充层。它用于补足主 benchmark 在以下场景上的不足：
  - `photo`
  - `scan`
  - `degraded_scan`
  - `document_embed`
  - `page_level`
  - `handwritten`
  - `journal_fig`
  - `multi_grid`
  - `chinese_exam`

从当前仓库的来源链来看，这一层更适合描述为：

> 项目自建的 real-world supplementary robustness subset，其中既包含可追溯公开结构图像的再组织，也包含 synthetic-realistic 页面/弱域补充层，而不是单一官方 benchmark。

标签检查方案：

1. 使用已知结构标签作为主标签来源；
2. 统一落成 `canonical_smiles`；
3. 通过 RDKit 检查可解析性；
4. 结合文件存在性、图像可读性和重复检查做结构性 QC。

## 二、清洗层：`canonical_smiles_curated_v2`

路径：

```text
V2/data/eval/canonical_smiles_curated_v2/
```

它不是最终提交层，但很重要，因为它决定了后续 mixed 评测集的主干部分。

当前已知清洗动作：

1. 从 `canonical_smiles_main_v1` 出发；
2. 对图像模式风险样本做清理；
3. 被移除的 150 条样本全部来自 `decimer`；
4. 清洗后保留：
   - `uob`: 200
   - `uspto`: 200
   - `real_world`: 217

这一步的意义不是“降低难度”，而是先把 mixed 集的 canonical 主干部分清理成更稳定、图像模式风险更低的主层。

## 三、混合评测集：`ocsr_realworld_mixed_eval_v1p1`

路径：

```text
V2/data/eval/ocsr_realworld_mixed_eval_v1p1/
```

规模：

- 总量：`770`

构造链：

1. `canonical_smiles_main_v1`
2. `canonical_smiles_curated_v2`
3. `ocsr_realworld_mixed_eval_v1`
4. `ocsr_realworld_mixed_eval_v1p1`

也就是说，`v1p1` 不是平地起一个数据集，而是主 benchmark 清洗后，再加入教育场景 canonical 化子集，并进一步去重和回填得到的版本。

### 1. `canonical_main`

在 `v1p1` 中，`canonical_main` 的主干仍然来自清洗后的 `canonical_smiles_curated_v2`，也就是：

- `uob`
- `uspto`
- `real_world`

当前数量：`617`

### 2. `edu_chemc`

- 性质：本地整理的教育场景测试子集
- 当前数量：`153`
- 当前角色：教育场景补充层

这里最关键的是：当前 `edu_chemc` 并不是完全自行重新采集构建的，而是基于本地已整理的 `EDU-CHEMC` 测试子集进行筛选、转换、去重和回填。

因此，更准确的写法应当是：

> 教育场景中的手写/教学结构图当前主要来自本地整理的 `EDU-CHEMC` 测试子集，而不是完全由项目组自行重新采集。当前工作的重点在于统一标签口径、提高唯一分子覆盖度，并构造成与主 OCSR 结果可以并列解释的补充 benchmark。

## 四、`edu_chemc` 的转换、去重与回填

这是当前评测体系中最关键的一步优化之一。

### 1. 原始问题

在 `ocsr_realworld_mixed_eval_v1` 中，`edu_chemc` 子集：

- 总量：153
- 唯一分子数：97

也就是说，存在明显的重复分子现象，这会导致评测更像“重复结构再识别”，而不是对教育场景泛化的评估。

### 2. 当前优化方向

`v1p1` 的处理链是：

1. 对当前 `edu_chemc` 子集按 exact canonical SMILES 分组；
2. 每组保留 1 条现有样本，其余重复样本移除；
3. 从 `edu_chemc_convertibility_trial_v1` 中重新做 `ssml_normed -> canonical_smiles` 自动转换；
4. 只从转换成功且此前未出现过的分子中补回新的样本；
5. 保持总量仍为 153，但把唯一分子覆盖度提升到 153。

### 3. 标签转换方案

对候选池中的教育场景样本，当前使用的是可复现自动流程，而不是手工逐条猜测：

1. 读取 `ssml_normed`
2. 提取单个 `chemfig` 块
3. 进行本地 token / branch / reconnect 解析
4. 映射到最小可支持的原子/片段集合
5. 构建 RDKit 分子
6. `SanitizeMol`
7. 输出 `canonical_smiles`

### 4. 候选过滤

为避免脏样本直接补回，候选池并不是全进，而是做了筛选：

- 单 `chemfig`
- 无反应箭头
- 无 `+` 多组分
- `ssml_len <= 220`
- `max_side <= 768`
- 无显式变量占位
- 无 `circle`
- 无复杂 ring reconnect

### 5. 失败样本处理

对无法稳定转换的候选样本：

- 直接丢弃
- 不做猜测性修补
- 不手工补 SMILES

这使得当前 `edu_chemc` 的提升更偏向“标签口径统一”和“唯一分子数提升”，而不是人为修饰后的数据堆积。

## 五、正式评测集总览 `eval/`

路径：

```text
V2/data/eval/
```

当前正式提交版在顶层目录中被组织为一个 **evaluation collection**，由两个正式子基准组成：

1. `canonical_smiles_main_v1`
2. `edu_chmec_ssml_normed_test_v1`

这种结构不是为了显得复杂，而是为了避免任务边界混乱。因为：

- `canonical_smiles` 与 `ssml_normed` 不是同一标签空间；
- 如果硬混成一个总分，结果会不清晰；
- 分开解释更容易让评审理解各子基准的作用。

## 六、当前质量控制机制

评测集当前已经有的质控并不只是一句“基于规则调分布”，而是至少包含以下层次：

### 1. 结构性 QC

- 清理实验性候选层与临时目录
- 删除无效文件
- 保证 `annotations/` 与 `images/` 对齐
- 保证 `stats.json` 与目录结构一致
- 保证主任务与教育场景标签空间隔离

### 2. 标签层 QC

- 主 OCSR 核心集统一为 `canonical_smiles`
- EDU 候选层保留 `ssml_normed` 原格式，但在 mixed benchmark 中只使用经过稳定转换和去重后的 canonical 子集
- 通过 RDKit 进行 canonicalization 与可解析性检查

### 3. 去重与唯一分子控制

- `edu_chemc` 在 `v1p1` 中进行了 exact canonical 去重
- 通过 trial 候选池回填，保持样本总量不变，同时提升唯一分子数

### 4. 泄漏边界

- 核心集与训练数据之间做了显式隔离
- 训练集构建时会过滤 `v1p1` 中已出现的 canonical SMILES

## 七、当前仍需增强的部分

如果想进一步提高评测集质量和对应评分，最值得补的不是继续无差别加样本，而是补：

1. 每个来源的可视化抽检页
2. 更明确的人工复核记录
3. 更明确的来源表与标签来源链
4. 对 `real_world` 层中哪些是真实采集、哪些是 synthetic-realistic 的边界进一步写清楚
5. 对教育场景候选层补更正式的 QC 与许可证说明

## 八、如何让这一部分评分继续提高

如果目标是把“评估集质量”和“训练/评测数据科学性”继续往上拉，最现实的增分方式是：

1. 把来源表、转换链、去重链写成正式文档；
2. 对每个来源抽样做人工复核记录；
3. 增加分来源统计与图表；
4. 把“规则”提升为“可审计的数据处理流程”。

评审通常并不要求你所有数据都来自纯人工采集，但会要求你：

- 说清楚哪些来自公开数据；
- 哪些来自本地整理；
- 哪些是项目补充层；
- 标签如何检查；
- 数据如何清洗；
- 错误样本如何处理。

把这些讲清楚，评分会比单纯再补几十张图更有效。

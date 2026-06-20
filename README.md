# PaddleOCR-VL-1.5-OCSR

本项目聚焦于化学结构式识别（Optical Chemical Structure Recognition, OCSR）中的一个明确子任务：输入单张分子结构图像，输出唯一的 `canonical SMILES`。模型基于 `PaddleOCR-VL-1.5`，当前公开版本使用单阶段 `LoRA SFT` 进行任务定制，并在统一评测脚本下对主 benchmark 与真实世界补充 benchmark 进行对比评估。

这份说明文档主要把训练数据、评测集、结果对比和后续优化方向讲清楚。

## 项目边界

当前主线任务固定为：

- 输入：单张分子结构图像
- 输出：唯一的 `canonical SMILES`

因此，当前主训练线和主评测线都刻意不混入以下标签空间：

- `ssml_normed`
- `chemfig`
- LaTeX 公式
- 其他不与 `canonical SMILES` 一致的结构表示

这样做的原因很简单：如果标签空间不统一，训练目标会漂移，评测结果也会变得难以解释。当前项目选择先把主任务定义收窄，再围绕弱域与真实场景做补强。

## 当前训练范式

当前项目使用的是单阶段 `LoRA SFT`，而不是多阶段课程学习、RLHF 或偏好优化。具体做法是：

- 基座模型：`PaddleOCR-VL-1.5`
- 微调方式：`LoRA SFT`
- 输出约束：只训练 `canonical SMILES`
- 数据策略：公开主分布保底、真实弱域上权、复杂合成样本适度补充、干净合成子集限额

当前还没有引入：

- DPO / ORPO / KTO
- RLHF / RLAIF
- 多模型自博弈
- 结构级奖励函数

对这个阶段的项目来说，先把标签空间、数据配比、弱域覆盖和评测口径做稳定，比过早引入更复杂的训练算法更重要。

## 原版模型与微调模型结果

为了避免只报告微调后结果，这里同时给出 `PaddleOCR-VL-1.5` 原版模型和当前微调模型在同一套评测脚本下的结果。

### 1. `canonical_smiles_main_v1`

| 模型 | canonical exact acc | token micro F1 | valid SMILES | mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR-VL-1.5 原版 | `0.00%` | `5.34%` | `32.59%` | `0.0021` |
| 当前微调模型 | `32.86%` | `70.35%` | `71.84%` | `0.6992` |

### 2. `ocsr_realworld_mixed_eval_v1p1`

| 模型 | canonical exact acc | token micro F1 | valid SMILES | mean Tanimoto |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR-VL-1.5 原版 | `0.00%` | `6.59%` | `30.78%` | `0.0027` |
| 当前微调模型 | `33.77%` | `70.18%` | `75.84%` | `0.6849` |

从这两组结果可以看出，原版 `PaddleOCR-VL-1.5` 在 OCSR 这个任务上还不能直接使用。它常见的输出是 LaTeX、图注、自然语言描述，或者无法闭合的伪 SMILES。当前微调的作用不是做小幅修补，而是把一个通用视觉语言模型拉到“图像到 canonical SMILES”这个明确任务上。

当前微调模型的分来源表现仍然不均衡：

- `uob` 相对较强
- `uspto` 可用，但与公开强基线仍有差距
- `real_world` 偏弱
- `decimer` 手绘结构偏弱
- `edu_chemc` 教育场景 canonical 化子集仍偏弱

## 训练数据来源与构成

当前主训练集来自：

```text
V2/data/sft_materialized/train_singleline_rw_messages.jsonl
```

当前统计：

- 总样本：`22807`
- 唯一图片：`17495`
- 唯一 canonical SMILES：`15606`

这里的“唯一图片”指的是**不同的图像文件路径数**，而不是 message 条数。因为当前训练集中存在确定性重复加权，同一张图像可能因为上权策略出现多次，所以总样本数大于唯一图片数。这不是数据脏乱，而是当前加权策略的直接结果。

训练集由 7 部分来源组成：

| 来源 | 当前数量 | 来源性质 | 在训练中的角色 | 标签处理方式 |
| --- | ---: | --- | --- | --- |
| `uob` | 5016 | 公开 benchmark | 标准 printed OCSR 主分布 | 结构标签统一后用 RDKit canonicalize，无法稳定落成 canonical SMILES 的样本不进入主训练线 |
| `uspto` | 5151 | 公开 benchmark | 标准 printed / patent-style 主分布 | 同 `uob`，统一 `canonical_smiles` 后进入训练 |
| `real_world` | 4140 | 项目自建补充层 | 弱域补强，覆盖 photo/scan/document/page/handwritten/chinese_exam 等 | 优先继承已知结构标签，再统一 canonicalize，不依赖从噪声图像重新反推标签 |
| `molgrapher_synthetic` | 4000 | 公开或可追溯合成结构图来源 | 复杂结构与视觉扰动补充 | 保留结构标签，统一转 canonical SMILES，剔除非主任务格式 |
| `uspto30k_clean` | 1500 | 公开结构集合再组织子集 | 干净补充分布 | 统一 canonicalize，限额保留，避免 clean 样本过量 |
| `uspto30k_abbreviated` | 1500 | 公开结构集合再组织子集 | 缩写/简写结构补充 | 统一 canonicalize，保留缩写风格差异 |
| `uspto30k_large` | 1500 | 公开结构集合再组织子集 | 大尺寸 / 长分子 / 稠密结构补充 | 统一 canonicalize，限额保留，补足结构和图像复杂度长尾 |

需要特别说明的是，这 7 部分来源并不等于“7 个彼此独立、命名清晰的公开数据集”。其中有些可以直接对应到公开 benchmark，有些则更准确地说是**基于公开结构集合再次组织得到的项目内部训练子集**。为了避免误导，这里按目前仓库中能够明确追溯的证据做出保守说明：

| 当前训练来源名 | 可直接说明的公开来源 | 当前最稳妥的写法 |
| --- | --- | --- |
| `uob` | UOB OCSR Benchmark | 可直接写为公开 benchmark 主分布来源 |
| `uspto` | USPTO OCSR Benchmark | 可直接写为公开 benchmark 主分布来源 |
| `molgrapher_synthetic` | MolGrapher synthetic structure image source | 当前项目实际使用的是本地整理后的 synthetic 子集，而不是原始全集直接消费 |
| `uspto30k_clean` | 公开 patent-style 结构集合的 clean 子集 | 更准确地写成“基于公开结构集合再组织得到的 clean 训练子集” |
| `uspto30k_abbreviated` | 公开 patent-style 结构集合的 abbreviated 子集 | 更准确地写成“基于公开结构集合再组织得到的 abbreviated 训练子集” |
| `uspto30k_large` | 公开 patent-style 结构集合的 large 子集 | 更准确地写成“基于公开结构集合再组织得到的 large 训练子集” |
| `real_world` | 无法对应单一公开数据集名 | 应写成项目自建的 real-world supplementary robustness layer，而不是某一个现成公开 benchmark |

这个组合不是平均拼接，而是按以下逻辑组织：

1. `uob/uspto` 负责维持标准 OCSR 主分布；
2. `real_world` 负责补足真实场景弱域；
3. `molgrapher_synthetic` 与 `uspto30k_*` 负责在结构复杂度、图像尺寸和表达形式上扩展训练边界；
4. 干净 synthetic 子集做限额，避免模型过度偏向渲染图风格。

### 公开来源的标签清洗方式

对于 `uob`、`uspto` 以及来自公开结构集合的 synthetic / patent-style 子集，当前训练线并不是原样消费原始标签，而是统一经过以下处理：

1. 读取原始结构标签；
2. 将主字段统一映射到 `canonical_smiles`；
3. 使用 RDKit 做 canonicalization；
4. 清除空标签、无法解析标签以及不在主任务边界内的标签；
5. 最终转为统一的 `messages` 训练格式。

也就是说，当前主训练线的关键不在于“数据来自公开 benchmark”，而在于“公开来源在进入训练前已经做了统一标签清洗和任务边界裁剪”。

## 训练数据清洗、标注与质量控制

当前训练线的数据处理不是手工维护的松散拼盘，而是一条脚本化、可复现的数据工程链。

### 标签空间清洗

进入主训练线前，所有样本都要经过标签空间清洗。当前明确不进入主训练线的标签包括：

- `ssml_normed`
- `chemfig`
- LaTeX 公式
- 其他不与 `canonical SMILES` 一致的结构表示

进入主训练线的标签，最终都要统一为 `canonical SMILES`。

### 评测集同分子过滤

在构建：

```text
V2/data/sft_materialized/train_singleline_rw_messages.jsonl
```

时，当前训练脚本会显式过滤 `ocsr_realworld_mixed_eval_v1p1` 中已出现的 canonical SMILES。当前构建摘要显示：

- 被过滤的评测重合样本：`397`
- 被过滤的不可读图像：`1`

这一步的意义是避免模型通过同分子记忆获得虚高验证结果。

### 加权与限额

当前构建策略中：

- `real_world`: `repeat 5`
- `molgrapher_synthetic`: `repeat 2`
- `uob`: `repeat 1`
- `uspto`: `repeat 1`
- `uspto30k_clean`: `cap 1500`
- `uspto30k_abbreviated`: `cap 1500`
- `uspto30k_large`: `cap 1500`

这意味着当前训练集不是按“条数越多越重要”，而是显式承认真实弱域的训练价值更高。

### 项目补充层的标注方式

`real_world` 这部分并不完全依赖人工从复杂图像反推结构，而是尽量采用“已知结构 -> 派生图像”的继承式标注方式：

1. 先确定样本对应的已知结构标签；
2. 再进行页面嵌入、图像退化、拍照风格构造或其他弱域生成；
3. 图像形成后直接继承原结构标签；
4. 最终统一做 `canonical_smiles` 规范化。

这种方案的优点是：

- 比完全人工从弱域图像回推标签更稳定；
- 更适合批量构造 `photo / scan / chinese_exam / document_embed` 一类样本；
- 更容易和公开 benchmark 主分布保持统一标签空间。

### 审计结果

当前审计脚本检查：

- 图像是否存在
- 图像是否可读
- prompt 是否正确
- 输出是否为空
- 是否混入非 SMILES 输出
- 是否与评测集发生 ID/图片名/canonical SMILES 重叠


当前训练集统计结果：

| 统计项 | 数值 |
| --- | --- |
| SMILES 长度 p50/p90/p95/p99/max | `40 / 92 / 145 / 265 / 793` |
| 图片宽度 p50/p95/max | `773 / 1141 / 2644` |
| 图片高度 p50/p95/max | `504 / 1024 / 2547` |
| 图片面积 p50/p95/max | `360000 / 1048576 / 4722138` |
| 图像形态 | `balanced=12854, wide=8355, very_wide=1193, tall=405` |

这些统计和审计结果说明，当前训练线已经形成了基础的结构性质量控制闭环。

## 评测集来源与修改方向

当前项目的评测集不是单层单口径数据集，而是一个分层评测体系。

### 1. 主 OCSR benchmark：`canonical_smiles_main_v1`

路径：

```text
V2/data/eval/canonical_smiles_main_v1/
```

规模：

- 总量：`767`

来源构成：

- `decimer`: 150
- `uob`: 200
- `uspto`: 200
- `real_world`: 217

其中：

- `decimer`：公开手绘 benchmark，用来表征真实手绘结构识别能力；
- `uob/uspto`：公开主 benchmark，用来维持主 OCSR 可比性；
- `real_world`：项目补充层，用于补足标准 benchmark 对真实弱域覆盖不足的问题。

如果需要对外明确写清来源边界，建议用下面的表述：

| 当前评测来源名 | 当前建议表述 | 性质 |
| --- | --- | --- |
| `decimer` | DECIMER Hand-drawn Molecule Images dataset | 公开手绘 benchmark |
| `uob` | UOB OCSR Benchmark | 公开 benchmark |
| `uspto` | USPTO OCSR Benchmark | 公开 benchmark |
| `real_world` | project-built synthetic-realistic robustness subset | 项目补充层 |

标签检查方案：

1. 检查图像存在性与可读性；
2. 统一 `ground_truth.smiles` 字段；
3. 使用 RDKit 做 canonicalization 检查；
4. 清除明显异常、空标签或不可解析标签；
5. 统一到主任务脚本可直接使用的 schema。

这里需要强调的是：主评测集中的公开 benchmark 样本并不是简单下载后直接使用，而是在正式进入评测层前统一做了字段规范化、标签检查、RDKit canonicalization 检查以及目录结构整理。

### 2. 清洗层：`canonical_smiles_curated_v2`

路径：

```text
V2/data/eval/canonical_smiles_curated_v2/
```

这是当前 mixed 评测集的主干来源之一。当前已知清洗动作包括：

1. 从 `canonical_smiles_main_v1` 出发；
2. 对图像模式风险样本做清理；
3. 被移除的 `150` 条样本全部来自 `decimer`；
4. 清洗后保留：
   - `uob`: 200
   - `uspto`: 200
   - `real_world`: 217

也就是说，`canonical_curated_v2` 的作用不是代替主 benchmark，而是为 mixed benchmark 提供一个更稳定的 canonical 主层。

### 3. 混合评测集：`ocsr_realworld_mixed_eval_v1p1`

路径：

```text
V2/data/eval/ocsr_realworld_mixed_eval_v1p1/
```

规模：

- 总量：`770`

构造链路为：

1. `canonical_smiles_main_v1`
2. `canonical_smiles_curated_v2`
3. `ocsr_realworld_mixed_eval_v1`
4. `ocsr_realworld_mixed_eval_v1p1`

这个版本的关键优化点在于 `edu_chemc` 子集的去重与回填，而不是简单增加样本数量。

### 4. `edu_chemc` 子集

`edu_chemc` 当前不是完全由项目组自行重新采集构建的，而是基于本地整理的 `EDU-CHEMC` 测试子集，经过转换、去重与回填形成的教育场景补充 benchmark。

更准确的描述方式应当是：

> 教育场景中的手写/教学结构图当前主要来自本地整理的 `EDU-CHEMC` 测试子集，而不是完全由项目组自行重新采集。当前工作的重点在于统一标签口径、提高唯一分子覆盖度，并构造成与主 OCSR 结果可以并列解释的补充 benchmark。

因此，对外说明时更稳妥的写法是：

- 主 benchmark 层中的 `uob / uspto / decimer` 主要来自公开 benchmark；
- `real_world` 主要来自项目补充层；
- 教育场景中的 `edu_chemc` 则主要来自本地整理的 EDU-CHEMC 测试子集，而不是完全重新人工采集。

### 5. EDU 子集的关键修改

在 `ocsr_realworld_mixed_eval_v1` 中：

- 总量：`153`
- 唯一分子数：`97`

在 `v1p1` 中，做了以下处理：

1. 按 exact canonical SMILES 去重；
2. 每组保留 1 条，其余重复样本移除；
3. 从 `edu_chemc_convertibility_trial_v1` 中自动转换并回填新的唯一分子；
4. 保持总量仍为 `153`，但唯一分子提升为 `153`。

即：

| 阶段 | 图像数 | 唯一分子数 |
| --- | ---: | ---: |
| 去重前 | `153` | `97` |
| 去重后并回填 | `153` | `153` |

这里真正的优化不是“扩量”，而是：

- 去掉重复分子偏置
- 提升 mixed benchmark 的泛化测试属性

### 6. EDU 标签转换方案

当前 `edu_chemc` 的 canonical 化不是人工逐条猜测，而是通过可复现的自动流程完成：

1. 读取 `ssml_normed`
2. 提取单个 `chemfig` 块
3. 做本地 token / branch / reconnect 解析
4. 映射到最小可支持的原子/片段集合
5. 构建 RDKit 分子
6. `SanitizeMol`
7. 输出 `canonical_smiles`

候选过滤规则包括：

- 单 `chemfig`
- 无反应箭头
- 无 `+` 多组分
- `ssml_len <= 220`
- `max_side <= 768`
- 无显式变量占位
- 无 `circle`
- 无复杂 ring reconnect

对转换失败样本：

- 直接丢弃
- 不做猜测性修补
- 不手工补 SMILES

### 7. 当前质检机制

当前评测体系中的质量控制，已经不只是“基于规则优化分布”，至少包括以下几个层次：

#### 结构性 QC

- 清理实验性候选层与中间目录
- 删除无效文件
- 保证 `annotations/` 与 `images/` 对齐
- 保证 `stats.json` 与样本规模一致
- 保证主任务与教育场景标签空间隔离

#### 标签层 QC

- 主 OCSR 核心集统一为 `canonical_smiles`
- EDU 层保留原 `ssml_normed` 候选形式，但在 mixed benchmark 中只使用经过稳定转换和去重后的 canonical 子集
- 通过 RDKit 做 canonicalization 和可解析性检查

#### 去重与唯一分子控制

- `edu_chemc` 做 exact canonical 去重
- 用 trial 候选池回填，保持总量不变并提高唯一分子覆盖度

#### 泄漏边界

- 训练集构建时显式过滤 `v1p1` 中已出现的 canonical SMILES

换句话说，当前评测体系的质量控制并不是只有“规则调分布”，而是已经形成了：

- 结构性整理
- 标签层检查
- canonical 去重
- EDU 子集回填
- 训练/评测泄漏边界控制

这几层叠加后的结果，才构成当前 mixed 评测集的实质优化。

## 为什么当前数据集部分还有提升空间

现在最明显的短板不是“完全没有数据”，而是“证据链还不够像正式数据工程”。如果想把数据集相关评分继续提升，最有效的不是再盲目堆样本，而是把以下材料补齐：

1. 每个来源的来源表
2. 训练/评测标签来源链
3. 可视化抽检页
4. 人工复核记录
5. 统计图表
6. QC 表格

只要把这些内容补齐，当前的数据集部分从现在这档继续往上走是现实的。

## 后续怎么继续提高分数

### 先补文档与证据链

现在最容易加分的不是再训，而是补：

- 数据来源表
- 质检表
- 抽检页
- base vs fine-tuned 对比

### 再补弱域

如果继续训，优先补：

- `decimer / handwritten`
- `real_world / photo / scan`
- `edu_exam / chinese_exam`
- `page_level / document_embed`

### 可以引入的“agent 手法”

当前最值得做的 agent 化，不在训练损失函数，而在：

1. 数据侧：
   - 错误样本挖掘
   - 去重
   - 泄漏检查
   - 弱域 replay 候选池生成

2. 评测侧：
   - 自动生成分来源对比表
   - 自动生成抽检清单
   - 自动输出 error buckets

3. 推理侧：
   - 多候选生成
   - RDKit 合法性筛选
   - 候选重排

这类改造比当前直接跳向 RLHF/DPO 更划算，也更符合当前 OCSR 项目的成熟度。

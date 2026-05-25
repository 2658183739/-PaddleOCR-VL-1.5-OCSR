# 训练数据集构建报告

## 1. 任务目标

本项目面向 OCSR（Optical Chemical Structure Recognition）任务，目标是：

- 输入：单张分子结构图像
- 输出：唯一的 `canonical SMILES`

本训练线的设计原则是：

1. 主任务标签空间必须统一；
2. 训练目标必须尽量稳定；
3. 训练集既要覆盖标准 OCSR 主分布，也要显式补强真实场景弱域；
4. 必须避免评测集同分子泄漏；
5. 所有构建过程应可复现、可审计、可继续扩展。

因此，当前主训练线不追求“把所有化学相关标签都揉进来”，而是明确收缩到：

> 图像到 `canonical SMILES`

## 2. 数据格式与标注规范

训练样本采用 PaddleOCR-VL 官方 SFT `messages` 格式：

```json
{
  "messages": [
    {"role": "user", "content": "<image>OCR: Output only the canonical SMILES string for the molecule shown in the image."},
    {"role": "assistant", "content": "COc1cc(N)ncn1"}
  ],
  "images": ["../assets/train_phase3/.../xxx.png"]
}
```

统一规范如下：

- 输入端必须包含 `<image>` 占位符；
- prompt 固定为：
  - `OCR: Output only the canonical SMILES string for the molecule shown in the image.`
- 输出端只允许 `canonical SMILES`；
- 不混入 `ssml_normed`、`chemfig`、LaTeX、表格结构等其他标签空间；
- 图像路径使用相对 `data/sft_materialized/` 的相对路径，便于迁移和打包。

这样定义的好处是：

1. 标签空间单一；
2. 评测脚本与训练脚本更容易保持一致；
3. 不会因为“同一图像存在多种等价表示”导致训练目标漂移。

## 3. 数据来源

当前主训练集由以下来源构成：

| 来源 | 数量 | 角色 |
| --- | ---: | --- |
| `uspto` | 5151 | 标准 printed / patent-style 主分布 |
| `uob` | 5016 | 标准 printed OCSR 主分布 |
| `real_world` | 4140 | 真实拍照、扫描、文档嵌入、教学/考试、手写等弱域补强 |
| `molgrapher_synthetic` | 4000 | 合成复杂结构与视觉扰动补充 |
| `uspto30k_clean` | 1500 | 干净补充分布 |
| `uspto30k_abbreviated` | 1500 | 缩写/简写结构补充 |
| `uspto30k_large` | 1500 | 大尺寸 / 复杂结构补充 |

总量：

- 总样本：`22807`
- 唯一 ID：`17495`
- 唯一图片：`17495`
- 唯一 canonical SMILES：`15606`

## 4. 为什么这样组合

当前组合不是平均拼接，而是围绕“主分布覆盖 + 弱域补强 + 过拟合抑制”设计的。

### 4.1 主分布覆盖

`uob + uspto` 是标准 OCSR 公开 benchmark 风格数据，保留它们的大体量是为了：

- 保持模型对标准 printed chemistry 图的基础识别能力；
- 保持主 benchmark 上的可比较性；
- 防止模型因为过度偏向真实场景噪声而丢掉主任务基础能力。

### 4.2 真实世界弱域补强

`real_world` 被显式上权，原因是当前项目在这些方向上最容易掉分：

- `photo`
- `scan`
- `degraded_scan`
- `document_embed`
- `page_level`
- `chinese_exam`
- `handwritten`
- `journal_fig`
- `multi_grid`

这些子场景更接近真实使用环境，也是比赛中更有区分度的部分。

### 4.3 合成数据的角色

`molgrapher_synthetic` 与 `uspto30k_*` 子集主要用于补：

- 复杂结构
- 长分子
- 缩写与变体表示
- 大图和结构稠密样本

但这部分并没有无限放大，而是通过上限控制和较低权重，避免模型过度适配 synthetic-clean 风格。

## 5. 当前数据配比策略

当前训练集由 `train_phase3_messages.jsonl` 派生，经过过滤、重权重和限额后生成：

```text
V2/data/sft_materialized/train_singleline_rw_messages.jsonl
```

对应构建脚本：

```text
V2/scripts/build_singleline_rw_sft_dataset.py
```

核心策略如下：

### 5.1 只保留 canonical-SMILES 样本

- 剔除非主任务标签空间；
- 保证主训练线不被 `ssml_normed / chemfig / LaTeX` 污染。

### 5.2 评测集同分子过滤

显式过滤 `ocsr_realworld_mixed_eval_v1p1` 中已出现的 canonical SMILES。

当前摘要显示：

- 被过滤的评测重合样本：`397`

这一步的意义是：

- 避免验证集虚高；
- 避免模型通过同分子记忆获得不真实的提升。

### 5.3 来源权重

当前权重策略：

- `real_world`: `repeat 5`
- `molgrapher_synthetic`: `repeat 2`
- `uob`: `repeat 1`
- `uspto`: `repeat 1`
- `uspto30k_clean`: `cap 1500`
- `uspto30k_abbreviated`: `cap 1500`
- `uspto30k_large`: `cap 1500`

这表示：

- 真实世界弱域被显式强化；
- synthetic 复杂结构被适度加强；
- `uob/uspto` 维持主分布基座；
- 清洁补充分布被限额，防止“越训越像干净渲染图”。

## 6. 场景与难度分布

当前数据集中比较关键的视觉/任务子场景如下：

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
| `abbreviated` | 1500 |
| `large` | 1500 |
| `clean` | 1500 |

这说明当前训练线并不是只用标准 printed chemistry 图，而是已经引入了明显的场景复杂度和视觉复杂度差异。

## 7. 数据清洗与质量控制

当前训练集的质量控制不是“人工感觉没问题”，而是通过脚本化审计完成。

审计脚本：

```text
V2/scripts/audit_singleline_training_dataset.py
```

统计脚本：

```text
V2/scripts/summarize_singleline_dataset_stats.py
```

### 7.1 审计结果

当前审计报告给出的关键结果：

- 缺失图片：`0`
- 不可读图片：`0`
- bad prompt：`0`
- 空输出：`0`
- 非 SMILES 输出：`0`
- 与 `v1p1` ID 重叠：`0`
- 与 `v1p1` 图片名重叠：`0`
- 与 `v1p1` canonical SMILES 重叠：`0`

这说明当前主训练集在结构层面是干净的，至少没有明显的：

- 图像缺失
- 标签为空
- prompt 错误
- 主任务空间被混标
- 与当前验证集发生同分子泄漏

### 7.2 统计结果

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
- 图片形态：
  - `balanced=12854`
  - `wide=8355`
  - `very_wide=1193`
  - `tall=405`

这些统计说明：

1. 当前数据不是单一尺寸分布；
2. 输出长度虽然有长尾，但主分布仍适合 `4096` 上下文；
3. 图像长宽比存在明显差异，说明页面型、宽图型和常规 crop 图都被覆盖了一定比例。

## 8. 训练内验证集

当前训练内验证集是：

```text
V2/data/sft_materialized/val_singleline_v1p1_messages.jsonl
```

它对应的真实 benchmark 来源是：

```text
V2/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl
```

这意味着当前 checkpoint 选择更偏向：

- 真实世界场景
- 教学/文档补充场景
- 主 benchmark 之外的弱域表现

这是一个刻意设计，而不是偶然选择。

## 9. 数据构建的科学性

从“训练数据集构建科学性”的角度，当前方案主要体现在以下几点：

### 9.1 任务边界清晰

不试图把多种不兼容标签空间硬揉成一条训练线，而是把主任务明确定义为 `canonical SMILES`。

### 9.2 来源组合有明确分工

- `uob/uspto` 负责主分布
- `real_world` 负责弱域
- `molgrapher_synthetic` 与 `uspto30k_*` 负责结构复杂度与风格补充

### 9.3 有显式防泄漏机制

训练集构建中直接过滤验证集同分子。

### 9.4 有脚本化审计

不是依赖人工 spot-check，而是用脚本化检查来验证：

- 图像可读
- prompt 正确
- 标签非空
- 主任务空间不被污染
- 验证集无交叉

### 9.5 有可继续扩展的入口

后续新增数据时，只要保持：

- `id`
- `source`
- `difficulty`
- `task_type`
- `canonical_smiles`
- `image_path`
- `license/source_url_or_doc`

并转成统一 `messages` 格式，就可以继续并入同一训练线。

## 10. 当前局限

虽然当前数据构建已经形成可复现闭环，但仍有局限：

1. `decimer / handdrawn` 类样本在主训练线中并不充分；
2. 真实世界私有采集图仍偏少；
3. `edu_chemc` 主标签空间没有直接进入主训练线，而是更多用于评测与专项转换；
4. 当前弱域补强仍需要更多手绘、拍照、扫描和教育题面型数据。

这些局限也是下一阶段继续补数据和补评测的主要方向。

## 11. 可复现性

构建命令：

```bash
python V2/scripts/build_singleline_rw_sft_dataset.py --project-root .
python V2/scripts/audit_singleline_training_dataset.py --project-root .
python V2/scripts/summarize_singleline_dataset_stats.py --project-root .
```

固定项：

- 随机种子：`20260512`
- 采样方式：确定性 shuffle + cap
- 评测集同分子过滤文件：
  - `V2/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl`

## 12. 一句话总结

当前训练数据集并不是简单拼接数据，而是围绕以下目标构造的：

> 用标准 OCSR 主分布保底，用真实世界与复杂结构弱域补强，通过显式去泄漏和脚本化审计，构建一条可复现、可解释、可继续扩展的 canonical-SMILES 主训练线。

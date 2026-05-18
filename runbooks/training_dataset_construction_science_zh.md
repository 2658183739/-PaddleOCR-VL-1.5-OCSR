# 训练数据集构建科学性说明

## 目标

本训练集服务于 OCSR（Optical Chemical Structure Recognition）任务：输入分子结构图像，输出 canonical SMILES。数据构建目标是提升 PaddleOCR-VL-0.9B 在真实拍照、扫描退化、文档嵌入、手写/教学图和复杂结构图上的鲁棒性，同时避免输出格式混乱和评测集泄漏。

## 格式规范

训练样本遵循 PaddleOCR-VL 官方 SFT `messages` 格式：

```json
{
  "messages": [
    {"role": "user", "content": "<image>OCR: Output only the canonical SMILES string for the molecule shown in the image."},
    {"role": "assistant", "content": "COc1cc(N)ncn1"}
  ],
  "images": ["../assets/train_phase3/.../xxx.png"]
}
```

统一约束：

- 输入端必须包含 `<image>` 占位符和固定 OCSR prompt。
- 输出端只允许 canonical SMILES。
- 不混入 `ssml_normed`、`chemfig`、表格 OTSL、公式 LaTeX 等其他标签空间。
- 图像路径使用相对 `V2/data/sft_materialized/` 的路径，便于迁移和打包。

## 数据来源与配比

当前训练集由 `V2/data/sft_materialized/train_phase3_messages.jsonl` 派生，经过同分子过滤、重权重和合成样本限额后生成：

```text
V2/data/sft_materialized/train_singleline_rw_messages.jsonl
```

当前规模：

- 总样本：`22807`
- 唯一 ID：`17495`
- 唯一 canonical SMILES：`15606`
- 唯一图片：`17495`
- 训练内验证集：`770`

来源分布：

- `uob`: `5016`
- `uspto`: `5151`
- `real_world`: `4140`
- `molgrapher_synthetic`: `4000`
- `uspto30k_clean`: `1500`
- `uspto30k_abbreviated`: `1500`
- `uspto30k_large`: `1500`

难度与视觉场景：

- `photo`: `785`
- `scan`: `795`
- `degraded_scan`: `375`
- `document_embed`: `330`
- `chinese_exam`: `670`
- `journal_fig`: `330`
- `page_level`: `390`
- `handwritten`: `230`
- `multi_grid`: `235`
- `hard`: `2006`
- `medium_hard`: `5151`
- `large`: `1500`
- `abbreviated`: `1500`
- `clean`: `1500`

## 构建策略

构建脚本：

```text
V2/scripts/build_singleline_rw_sft_dataset.py
```

策略：

- 只保留 canonical-SMILES 样本，剔除非主任务标签空间。
- 过滤 `ocsr_realworld_mixed_eval_v1p1` 中已出现的 canonical SMILES，避免同分子记忆影响验证可信度。
- `real_world` 样本 repeat 5，提高真实拍照、扫描、文档嵌入、手写等场景权重。
- `molgrapher_synthetic` repeat 2，补充视觉扰动和复杂结构。
- UOB/USPTO 保持 repeat 1，用于维持主分布覆盖。
- 干净合成数据设定 cap 1500，避免模型过度偏向清晰渲染风格。

该策略不是随机堆数据，而是围绕评分要求中的真实视觉复杂度、任务稀缺性和泛化能力做配比。

## 可复现性

复现命令：

```bash
python V2/scripts/build_singleline_rw_sft_dataset.py --project-root .
```

固定项：

- 随机种子：`20260512`
- 采样方式：确定性 shuffle + cap
- 评测集同分子过滤文件：`V2/data/eval/ocsr_realworld_mixed_eval_v1p1/annotations/labels.jsonl`
- 输出摘要：`V2/reports/singleline_rw_dataset_summary.json`
- 输出训练集：`V2/data/sft_materialized/train_singleline_rw_messages.jsonl`
- 输出验证集：`V2/data/sft_materialized/val_singleline_v1p1_messages.jsonl`

## 质量控制

审计脚本：

```text
V2/scripts/audit_singleline_training_dataset.py
```

复现命令：

```bash
python V2/scripts/audit_singleline_training_dataset.py --project-root .
```

已生成审计报告：

```text
V2/reports/singleline_rw_dataset_audit.json
```

关键结果：

- 缺失图片：`0`
- 不可读图片：`0`
- bad prompt：`0`
- 空输出：`0`
- 非 SMILES 输出：`0`
- RDKit 非法 SMILES：`0`
- 与 `v1p1` ID 重叠：`0`
- 与 `v1p1` 图片名重叠：`0`
- 与 `v1p1` canonical SMILES 重叠：`0`

统计报告：

```text
V2/reports/singleline_rw_dataset_stats.json
```

关键统计：

- SMILES 长度：p50=`40`，p90=`92`，p95=`145`，p99=`265`，max=`793`
- 图片宽度：p50=`773`，p95=`1141`，max=`2644`
- 图片高度：p50=`504`，p95=`1024`，max=`2547`
- 图片面积：p50=`360000`，p95=`1048576`，max=`4722138`
- 图片形态：balanced=`12854`，wide=`8355`，very_wide=`1193`，tall=`405`

这说明本任务输出长度远短于通用页面级 OCR，因此训练配置使用 `max_seq_len=4096` 是合理的 4090 单卡折中；若训练日志出现截断或丢样本告警，再提高到 `8192`。

## 可扩展性

后续新增数据时，只需要把新样本转为相同 `messages` 格式，并加入上游 materialized 或 manifest 构建流程。建议新增来源时必须提供：

- `id`
- `source`
- `difficulty`
- `task_type`
- `canonical_smiles`
- `image_path`
- `license/source_url_or_doc`

扩展后重新运行：

```bash
python V2/scripts/build_singleline_rw_sft_dataset.py --project-root .
python V2/scripts/audit_singleline_training_dataset.py --project-root .
python V2/scripts/summarize_singleline_dataset_stats.py --project-root .
```

只要审计报告通过，即可直接进入同一条 4090 LoRA 训练线。

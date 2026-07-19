# 数据许可与归属矩阵

## 发布结论

V3 项目代码和本项目产生的模型权重采用 Apache License 2.0。基础模型
`PaddlePaddle/PaddleOCR-VL-1.5` 的官方模型页明确标注 Apache-2.0。该项目
许可证只覆盖本项目代码、文档和派生权重，不会替代或扩大第三方数据许可。

公共 GitHub 和 Hugging Face 模型仓不上传第三方训练原图、训练 JSONL、论文
裁图或教育材料。数据名称、数量、处理规则和评测结果可以公开；如需取得原始
数据，使用者必须前往对应上游地址并遵守其条款。

本矩阵于 2026-07-19 重新核验上游公开页面。核验优先级为：作品自身的 LICENSE
文件或模型/数据卡明确 metadata，其次才是引用它的代码仓许可证。核验时基础模型
页显示 `apache-2.0`，OCSR_Review 显示 MIT，MolRecBench-Wild 显示 Apache-2.0；
MolGrapher-Synthetic-300K 与 USPTO-30K 数据页未显示独立 `License:` metadata。
“未显示”按未知/未声明处理，不从代码仓许可证推导数据许可证。

## 可审计矩阵

| 语义组 | V3 final 记录/占比 | 上游证据 | 上游许可状态 | V3 公开策略 |
| --- | ---: | --- | --- | --- |
| USPTO | 5,043 / 22.16% | [OCSR_Review](https://github.com/Kohulan/OCSR_Review)；其 README 说明 5,719 张基于 USPTO 的图像，仓库标注 MIT | OCSR_Review 仓库为 MIT；原始专利来源为 USPTO | 公开来源、统计和处理代码；不再分发训练图 |
| UOB | 4,869 / 21.39% | [OCSR_Review](https://github.com/Kohulan/OCSR_Review)；其 README 说明 University of Birmingham/MolRec 的 5,740 图 benchmark | OCSR_Review 仓库为 MIT；原始 UOB benchmark 未在本项目中发现更细的逐文件授权 | 公开来源、统计和处理代码；不再分发训练图 |
| real-world | 4,329 / 19.02% | 项目历史整理的拍照、扫描、页面、手写、教育和受控退化混合集合 | 无单一许可证；部分记录仅有集合级来源 | 不公开原图或训练清单；仅公开聚合统计、清洗规则和限制 |
| MolGrapher synthetic | 4,000 / 17.57% | [MolGrapher](https://github.com/DS4SD/MolGrapher) 和 [MolGrapher-Synthetic-300K](https://huggingface.co/datasets/docling-project/MolGrapher-Synthetic-300K) | 代码仓为 MIT；数据卡说明由 PubChem SMILES 和 RDKit 渲染，但未声明独立 dataset license | 仅公开来源与统计；不再分发训练图 |
| USPTO-30K clean | 1,501 / 6.59% | [USPTO-30K](https://huggingface.co/datasets/docling-project/USPTO-30K) | 数据卡说明来自 USPTO image/MolFile 对；页面未声明独立 dataset license | 仅公开来源、子集定义和统计；不再分发数据 |
| USPTO-30K abbreviated | 1,507 / 6.62% | 同上 | 同上 | 同上 |
| USPTO-30K large | 1,513 / 6.65% | 同上 | 同上 | 同上 |
| MolRecBench-Wild | train probe 800；locked canonical 301；symbolic 460 | [MolRecBench-Wild](https://github.com/opendatalab/MolRecBench-Wild) | 上游仓库明确为 Apache-2.0 | 公共源码只保留划分/评测方法与统计；比赛完整包按规则保留评测材料 |
| PaddleOCR-VL-1.5 | 基础模型 | [官方模型页](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | Apache-2.0 | 保留原始归属、NOTICE 和 remote-code 声明 |

## 解释边界

1. “公开可访问”不等于“可以由本项目重新打包分发”。上游数据卡未给出明确
   dataset license 时，本项目采用不再分发原图的保守策略。
2. MIT/Apache-2.0 是对应仓库或作品的许可，不自动覆盖其引用的所有上游内容。
3. `real-world` 是内部语义组而不是单一公开数据集；不能把其中所有图片写成
   自行实拍，也不能给它统一补写一个并不存在的第三方许可证。
4. 离线模糊、扫描、透视和压缩增强继承父样本的来源限制；算法变换不会产生
   新的数据授权。
5. 本矩阵用于工程发布审计，不构成法律意见。商业或高风险使用者应再次核对
   上游条款和适用司法辖区。

## 项目许可证选择理由

Apache-2.0 与 PaddleOCR-VL-1.5 的上游许可一致，明确包含版权和专利授权，适合
训练/评测代码与派生模型权重的公开复现。由于本项目不重新分发许可不明确的
训练原图，Apache-2.0 不会被误写成第三方数据的统一许可证；所有第三方归属与
限制继续由本文件和 `NOTICE` 保留。

# V3 人工审核完成声明

## 声明

项目所有者于 2026-07-19 确认：V3 冻结评测清单已经完成离线人工审核。审核覆盖
753 条 legacy core development、754 条 legacy region development、301 条 wild
strict locked canonical 样本以及 460 条 wild symbolic 样本；134 条
scaffold-novel 是 wild strict 的预定义子集，不重复计数。

项目所有者确认当前冻结 labels 可用于已经完成的评测，没有报告需要剔除的样本
或需要改写的标签，因此不需要重新选模或重新运行推理，现有 locked 指标保持不变。
本声明绑定的四个清单 SHA256 写在 `manual_review_attestation.json` 中；只要任一
清单发生变化，本声明即失效，必须重新审核或重新出具声明。

## 公开证据边界

公开仓库保留项目所有者完成声明、审核范围、冻结清单哈希和结果边界。审核人员
身份、签名以及逐样本内部工作表不随公共仓库发布。仓库不会虚构 Reviewer 姓名、
签名、分歧数量或逐样本决定。

`eval_manual_review.csv` 是此前生成的本地审核工具模板，里面的 pending 字段不再
作为本次离线人工审核的权威状态来源。对外状态以本声明及其 JSON 哈希绑定记录为准。

## 与自动 QC 的关系

自动 QC 负责图片可读性、RDKit 解析、单分子约束、canonicalization、泄漏、分组
和哈希；人工审核负责图像目标、标签语义和任务边界。两者互补，人工完成声明不替代
自动验证，自动通过也不被写成“人工通过”。

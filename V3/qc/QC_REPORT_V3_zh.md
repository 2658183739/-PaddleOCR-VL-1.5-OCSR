# V3 测试集人工质检报告

状态：**自动 QC 已完成；项目所有者确认离线人工审核已完成。公开证据采用 owner attestation，并与冻结清单 SHA256 绑定。**

## 1. 面板角色

| 面板 | 数量 | 角色 | 当前状态 |
| --- | ---: | --- | --- |
| legacy core strict | 753 | development | 历史面板，非最终测试 |
| legacy region strict | 754 | development | 历史面板，非最终测试 |
| wild strict | 301 | locked canonical final test | 自动 QC 通过；人工审核完成（owner-attested） |
| wild scaffold-novel | 134 | wild strict 子集 | 随 wild strict 完成人工审核 |
| wild symbolic | 460 | locked symbolic track | 与 canonical 分开；人工审核完成（owner-attested） |
| private photo | 0 | locked 自采 final test | waiting for collection |

## 2. 人工审核证据

| 证据 | 公开状态 | 作用 |
| --- | --- | --- |
| `qc/manual_review_attestation.json` | 公开 | 项目所有者确认审核完成、范围、结果边界和冻结清单 SHA256 |
| `qc/MANUAL_REVIEW_ATTESTATION_zh.md` | 公开 | 人工审核完成声明及证据边界 |
| 审核人员身份、签名、逐样本内部记录 | 不公开 | 隐私与内部审计材料，不在公共仓库虚构或披露 |
| `build_v3_datasets.py` + `verify_v3_workspace.py` | 公开 | RDKit、路径、分组、泄漏和 hash 自动 QC |

当前公开材料证明的是“项目所有者确认离线人工审核已经完成”，不是公开逐样本双盲
数据集。任何文档都不得补写不存在的姓名、签名、分歧数量或逐样本决定。

## 3. 审查步骤

1. 自动检查图片存在/可读、标签非空、RDKit 可解析、单分子、无 dummy atom。
2. 按 canonical molecule 检查 train/development/locked test 重叠。
3. MolRecBench 按 `paper_group` 留出整篇论文，每篇最多 5 张进入 locked canonical test。
4. 离线人工审核检查目标是否唯一、结构图是否可辨、标签是否对应和任务边界。
5. 项目所有者确认冻结 labels 是审核后可用于评测的版本，未报告需要剔除或改写的标签。
6. 将最终 labels 的 SHA256 写入 attestation；清单一旦变化，完成声明自动失效。
7. 只有在人工审核改变 labels 时才对冻结预测重新统计；不得据此返回调参或重新选模。

历史逐样本工具模板：`qc/eval_manual_review.csv`。它不是本次外部离线审核的权威
完成记录；对外状态以 `manual_review_attestation.json` 为准。

本地逐图工具：

```bash
# Reviewer 1 和 Reviewer 2 必须使用各自真实编号、独立运行
python V3/scripts/qc_review_app.py --project-root . --reviewer 1 --reviewer-id R1
python V3/scripts/qc_review_app.py --project-root . --reviewer 2 --reviewer-id R2 --port 7862

# 只在两名 reviewer 完成后处理分歧
python V3/scripts/qc_review_app.py --project-root . --reviewer adjudicator --reviewer-id ADJ --port 7863
```

工具只显示图片和预先冻结标签，分别写入 reviewer 列或 `final_decision`，不允许在审查界面修改标签。它保留为后续复核工具，不被用来补造本次已完成的外部审核记录。

## 4. 自动筛选前后差异

### 4.1 V2-1 基础训练清单

| 阶段 | 数量 | 差异原因 |
| --- | ---: | --- |
| V2-1 clean weighted 输入 | 23,047 | 历史 RDKit 可解析口径 |
| 多片段、symbolic 或不收敛标签剔除 | -273 | 不符合稳定单分子 canonical 口径 |
| 与新 development/locked test 分子重叠剔除 | -12 | canonical molecule overlap |
| V3 strict control | 22,762 | 2×2 消融共同底座 |

### 4.2 MolRecBench 全量 5,008 条

| 阶段 | 数量 | 去向/原因 |
| --- | ---: | --- |
| 原始 annotation | 5,008 | 自动检查输入 |
| symbolic 或非法 canonical | 3,508 | 不进入 canonical train/test |
| 与 legacy development 分子重叠 | 72 | 从 V3 wild pool 剔除 |
| strict canonical pool | 1,428 | 519 篇论文 |
| locked canonical test | 301 | 62 篇论文，每篇最多 5 张，分子内部唯一 |
| 同批 test 论文额外 held-out | 308 | 防止同论文回流训练 |
| 跨论文 canonical 重复剔除 | 19 | 与 locked test 分子重叠 |
| strict wild train | 800 | 与 test paper group、canonical molecule 均为 0 重叠 |

locked canonical test 中有 301 个唯一 canonical molecule，134 张属于 scaffold-novel 子集。

### 4.3 当前人工状态

| 面板 | 自动输入 | 自动通过 | 人工通过 | 人工拒绝 | 最终状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| legacy core development | 767 | 753 | 753（owner-attested） | 14（自动口径剔除） | complete development QC |
| wild strict locked test | 301 | 301 | 301（owner-attested） | 0（未报告审核后剔除） | complete, frozen |
| wild symbolic locked | 460 | 460 | 460（owner-attested） | 0（未报告审核后剔除） | complete, independent track |
| private photo locked test | 0 | 0 | 0 | 0 | waiting for collection |

## 5. 独立性说明

wild strict 的图片 N 和唯一分子 N 均为 301，但论文来源 N 为 62。统计比较必须按 `paper_group` 做 cluster bootstrap，不能把 301 张都当独立来源。

官方高分倾向是至少 1000 个真实评测实例。当前 301 张公开论文真实图满足真实性和来源分组要求，但规模尚未达到最高档；新增自采评测图必须单独记录真实设备与授权，算法增强不得计入真实评测规模。

自采照片如果 80 个结构各拍 4 张，图片 N 为 320，独立结构 N 仍为 80；必须按 `structure_id` 聚类。

## 6. 剩余证据边界

- 人工审核完成状态已经由项目所有者声明并绑定冻结 labels SHA256。
- 公共材料不提供审核人身份、签名或逐样本内部决定，因此不得宣称公开双盲复核数据集。
- private photo 仍为 0；这与现有公开论文真实图审核是两件事，不能混写。
- 自动规则不能替代人工判断“图像与结构标签是否真的一致”；owner attestation 也不能替代后续清单变更后的重新审核。

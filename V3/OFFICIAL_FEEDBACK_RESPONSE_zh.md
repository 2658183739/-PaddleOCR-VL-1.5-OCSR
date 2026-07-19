# 官方反馈逐项响应

本文把官方意见映射到 V3 当前证据和实际行动。`已完成` 只表示仓库中有可复核文件；`待人工` 需要真实人员、设备或签名，不能用自动脚本代替；`发布前` 表示即使模型训练结束也不能跳过。

| 官方意见 | 当前证据 | 状态 | 具体解决方式 |
| --- | --- | --- | --- |
| 补充非公开手绘/真实拍照的标注工具、人员、QC | `scripts/qc_review_app.py` 提供逐图工具；项目所有者确认 legacy/wild/symbolic 离线人工审核完成，并用 attestation 绑定 frozen labels SHA256；private photo 仍为 0 | 已完成现有面板审核；自采待补 | 公开 owner attestation 与 QC 流程，不虚构审核人身份或逐样本记录；未来自采另行记录设备/角度/光照/授权并重新双审 |
| 解释公开 benchmark 抽样比例 | core 753、region 754 的目标配额、严格通过数和理由已写入 README/数据卡；UOB/USPTO 约 200 条是比较锚点，DECIMER/EDU 约 150 条覆盖困难类型，real-world 保留全部通过样本 | 已完成（方法） | 提交时附 `labels.jsonl`、`v2_1_eval_qc_summary.json` 和筛选前后统计；不能把目标配额写成最终通过数 |
| 增加训练数据说明和模型微调 | `TRAINING_DATA_AND_FINETUNING_REPORT_zh.md`、数据卡、模型卡；final control 22,762 条七部分配比、LoRA continuation、2×2×2 probe 和 hard replay gate | 已完成（文档/实验） | 把报告随最终开源包提交；补样本级许可字段后再公开训练清单 |
| 加强社区/技术影响力 | 方法、脚本、Demo、18 页 HTML/PPT、GitHub `a68b434`、HF `e496110` 和历史 4090 参考已公开 | 已完成（发布） | 已验证公开页面、Apache-2.0、文件清单、revision 与权重 SHA256；第二台机器 clean-download/GPU smoke 保持为限制，不把 Demo 录屏作为本轮已完成证据 |
| README 补充训练数据来源和构成 | README 新增七部分精确数量/比例、source counts 和许可缺口 | 已完成（文档） | 公开前逐样本补 `license/source_url_or_doc/structure_id`，未知来源隔离 |
| 增加 PaddleOCR-VL-1.5 直接 baseline | 4090 历史 770 条面板：原始权重 exact 0.00%、valid 30.78%、token F1 6.59%、Tanimoto 0.0027；V2-1 export exact 33.77% | 已完成（历史统一面板） | V3 final 结果只使用 `FINAL_RESULTS.json`，不要把 V2-1 stable/best/oracle 当 V3 baseline |
| 解释评测集修改方向 | 按论文分组、canonical 唯一性、公开锚点/弱域诊断、symbolic 独立 track、自动 QC 和 owner-attested 人工审核均已记录 | 已完成（规则与报告） | 若未来清单变化，attestation 自动失效并重新审核；当前不补造不存在的一致率或分歧统计 |
| 避免“唯一图片”等含糊表述 | 文档统一区分 `records`、`unique_image_refs`、`unique_canonical_smiles`、`structure_id`、`paper_group` | 已完成（文档） | 提交前做一次人工术语审校；每个 N 同时说明独立单位 |
| 描述七部分训练数据来源、公开标签清洗、自采标注 | 七部分表格和清洗顺序已写入报告；当前没有完成的自采训练/评测数据，不写成自采 | 已完成（已有数据范围） | 为历史 real-world 逐样本补来源/许可；新增自采按协议双审 |
| 解释测试数据来源和人工 QC | core/region/wild 的 source、paper group、strict 数量和 QC 状态已写入数据卡与 QC 报告；现有 frozen labels 审核完成 | 已完成现有面板 | public evidence 为 owner attestation + 四个 labels SHA256；private photo 需真实采集后才能关闭规模缺口 |
| 说明规则 QC 有效性并引入人工 | 自动规则剔除坏图、非法/多片段、泄漏和不幂等标签；项目所有者确认人工语义审核完成 | 已完成（证据边界公开） | 自动与人工结果分别报告；未公开逐样本双盲记录，不宣称公开 reviewer 一致率 |
| 报告审查人员构成、步骤、筛选前后差异 | `qc/QC_REPORT_V3_zh.md` 已给出自动筛选前后数量、人工步骤、审核范围和 owner-attested 结论 | 已完成可公开范围 | 审核身份、签名和逐样本内部表不公开或虚构；清单变化时必须重新出具声明 |
| 补真实手绘、拍照不同退化 | 当前 private photo=0；算法增强 927/1854 只属于训练 augmentation | 待采集 | 至少两台设备、正面/斜拍/低光/反光阴影四条件；eval-only 结构与 train 结构完全隔离；官方 `>=1000` 规模还差约 699 张真实图 |
| 证明训练配比不是主观意见 | `2×2×2` 两 seed、主效应、交互、dose-2、paired bootstrap 和失败条件均有证据 | 已完成（探索性） | 提交时明确只有两个 seed，不能写显著最优；后续确认实验至少 4 seed、分块随机顺序 |
| 继续探索后训练 | final checkpoint、300-step hard replay、greedy/beam4/return4 和 chem-light 固定候选重排均已完成；最终采用 `checkpoint-1400 + beam4/return4` | 已完成本轮 | hard replay 与 chem-light 均因回退被拒绝；locked test 冻结后未回调调参 |
| 提交训练数据构建报告、GitHub、Hugging Face、答辩 PPT | 训练报告、模型卡、数据卡、Apache-2.0/NOTICE、最终模型和 18 页 HTML/PPT 已发布 | 已完成（发布） | GitHub `a68b434`、HF `e496110`、权重 SHA256 与材料 hash 已回填；Demo 录屏按本轮范围取消 |

## 不应写入提交材料的结论

1. 不把只有两个 seed 的 probe 写成统计显著或全局最优。
2. 不把 301 张论文图、算法增强图或历史整理图写成“1000 张真实自采”。
3. 不把自动 QC 写成双人审核，不补写不存在的姓名、设备或授权。
4. 不把 PaddleOCR-VL-1.5 原始模型、V2-1 LoRA baseline、V3 final 和 oracle 混在一张“提升率”表中。
5. 不根据 locked test 分数返回修改模型、prompt、beam、crop 或阈值。

## 提交前最小关闭条件

- `all_pipeline_complete.txt`、最终 `FINAL_RESULTS.json/.md`、checkpoint/模型/hash 和日志存在。
- final merged model 可以独立加载，推理 smoke 和测试通过。
- wild/symbolic 的 owner-attested 完成声明、范围、结论与 frozen labels SHA256 一致；不把未公开逐样本表写成公开双盲证据。
- 训练样本级许可/来源覆盖率经过审计；未知来源数据不进入公共 release。
- GitHub、Hugging Face、Apache-2.0/NOTICE、CONTRIBUTING、环境锁与 HTML/PPT 按实际状态发布；Docker/第二机复现保持 limitation。

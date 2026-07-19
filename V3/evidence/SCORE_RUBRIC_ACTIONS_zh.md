# 官方评分表到 V3 证据映射

历史反馈总分为 43 分。本表不预测最终得分，只按官方六维评分表区分“已有可核验证据”和“仍需真实完成的工作”。

官方规则：

- 评估集中合成数据占比过高可被直接判 0 分；
- 评估集高分倾向为 `>=1000` 个真实实例、来源清晰、无版权问题；
- 获奖候选会核验代码、模型、文档、评估集真实性和结果一致性；
- 完整开源项目需要 GitHub 训练/评估代码、文档、Demo，以及 Hugging Face 微调模型和模型卡。

## 1. 评估集质量（20）

| 子项 | V3 已有证据 | 当前缺口 | 解决与完成标准 |
| --- | --- | --- | --- |
| 1.1 数据规模 | 301 张 locked canonical 真实论文图，301 个唯一分子、62 篇论文 | 距离官方高分倾向的 1000 真实实例仍差 699 张；private 为 0 | 准备约 195-200 个 eval-only 结构，每结构 4 视角，预留质检淘汰后至少保留 699 张；最终 locked 总数 `>=1000` |
| 1.2 标注准确性 | RDKit 稳定单分子过滤、自动验证；项目所有者确认 legacy/wild/symbolic 离线人工审核完成并绑定 frozen labels SHA256 | 未公开逐样本双盲记录与 reviewer 一致率 | 公开 owner attestation、范围和 hash；不虚构姓名/签名/分歧，清单变化后重新审核 |
| 1.3 数据多样性 | 论文截图、页面裁图、拍照/扫描、手绘、长分子、stereo 分层 | 自采不同设备、角度、光照和遮挡仍为 0 | 至少 4 种实拍退化、2 台设备，报告条件分布；算法增强不计入真实评测规模 |
| 1.4 难度合理性 | 已有 difficulty、atom_count、scaffold-novel 字段 | 尚无最终 easy/medium/hard 审核后分布图 | 双审后生成难度统计和可视化，说明分层规则与真实业务分布 |

## 2. 场景稀缺性（15）

| 子项 | V3 已有证据 | 当前缺口 | 解决与完成标准 |
| --- | --- | --- | --- |
| 2.1 研究稀缺性 | OCSR 是官方列出的高价值长尾方向 | 尚无正式文献/公开 benchmark 对比表 | 列出 DECIMER、MolRecBench 等公开基准的任务边界，并说明真实拍照、手绘、中文教育文档仍缺什么 |
| 2.2 工业需求价值 | 药化、专利、科研文献、教育题目均有结构数字化需求 | 缺真实用户、工作流和成本证据 | 补 2-3 个可核验业务流程、人工录入成本、错误风险和模型价值 |
| 2.3 场景独特性 | 输出是化学结构语言，不是普通字符 OCR | 与通用 OCSR 的差异仍需更清晰 | 明确“真实文档退化 + canonical 单分子转写 + 化学有效性验证”的独特任务定义 |

## 3. 任务复杂度（15）

| 子项 | V3 已有证据 | 当前缺口 | 解决与完成标准 |
| --- | --- | --- | --- |
| 3.1 视觉复杂度 | journal figure、photo、scan、handwritten、crop、glare 等分层 | 私有实拍尚未产生 | 至少 4 种真实实拍条件且比例可解释，报告各条件 exact/validity |
| 3.2 结构复杂度 | 有 targeted crop、候选生成和 selector 链路 | 主任务仍是单图到 SMILES，非多任务联合 | 若追求更高分，新增独立 page-level track：分子区域检测/裁切 + OCSR + 页面结构回填；不要与主 canonical exact 混分 |
| 3.3 理解复杂度 | canonicalization、结构有效性和候选选择需要化学约束 | 语义推理仍弱 | 可增加反应图中角色/步骤或文档上下文关联的辅助任务；两天主线不应为此破坏 OCSR 稳定性 |

## 4. 训练数据集构建科学性（20）

| 子项 | V3 已有证据 | 当前缺口 | 解决与完成标准 |
| --- | --- | --- | --- |
| 4.1 采集流程 | builder、private protocol、设备/人员/时间字段 | 部分历史 `real_world` 只有集合级来源 | 补齐样本级 URL、license、collector、capture_time、transform_parent；未知来源隔离 |
| 4.2 标注规范 | 单分子 canonical 口径、symbolic track、原因码 | 缺独立 annotation guideline 示例页 | 补正例/反例、盐/多片段、R-group、stereo、不可读图、多目标图处理规则 |
| 4.3 质量控制 | 自动 RDKit/路径/泄漏/hash 验证，双审模板 | 人工签名和争议统计未产生 | 执行双审并报告筛选前后数量、拒绝原因和一致率 |
| 4.4 数据统计 | `dataset_build_report.json`、`mixture_counts.csv`、精确 `2x2` 数据量 | 缺正式可视化分析 | 输出来源、难度、原子数、图像尺寸、退化、许可证覆盖率图表，数字可回溯到 JSON/CSV |

## 5. 模型微调策略与创新（10）

| 子项 | V3 已有证据 | 当前缺口 | 解决与完成标准 |
| --- | --- | --- | --- |
| 5.1 策略合理性 | V2-1 continuation、1.5 warm-start 对照、BF16 LoRA、回归闸门 | 无 A100 实际日志 | 保存 8 个 probe、final、可选 hard replay 的配置/日志/checkpoint hash |
| 5.2 实验充分性 | `2x2` wild x augmentation、两个 seed、seed block、paired/cluster bootstrap | 结果表仍为空 | 回填主效应、交互、seed 波动、95% CI 和失败实验；不只报最好一次 |
| 5.3 技术创新 | candidate oracle、reward selector、targeted crop、hard replay 分阶段设计 | P0-P4 未在同一 final 模型上完成 | 固定候选池逐阶段只改一个机制，区分召回、selector、crop 和 replay 收益 |

## 6. 技术文档与开源贡献（20）

| 子项 | V3 已有证据 | 当前缺口 | 解决与完成标准 |
| --- | --- | --- | --- |
| 6.1 文档质量 | 主 README、48h 计划、消融/评测/自采协议、缺失项清单 | 最终结果和限制未回填 | 所有表格只写真实完成结果，提供从安装到复现的完整命令 |
| 6.2 代码可复现 | builder、verifier、11 个测试模块/29 项测试、训练/推理/评测脚本、artifact hash | CUDA/Paddle 环境锁和公共版本号缺失 | `environment.yml`/`requirements-lock.txt`、nvidia-smi、Paddle commit、公共 Git commit 齐全，第二台 A100 通过 smoke |
| 6.3 Demo 完整性 | 本地 Gradio 代码存在，H800 已保存 final model 单样本与 sharding 推理 smoke | 未提供交互式 Demo 截图/录屏 | 本轮明确取消录屏，不把该项写成已完成；后续可在同一 final hash 上补正常图、坏图与超大图演示 |
| 6.4 社区价值 | 方法、数据构建流程、Apache-2.0/NOTICE、CONTRIBUTING、模型/数据卡、许可矩阵、复现 issue 模板、GitHub `a68b434` 与 HF `e496110` 已公开 | 第二台机器 clean-download/GPU smoke 仍缺 | 已验证远端 commit/revision/权重 SHA256；训练原图和 JSONL 因样本级许可不足而隔离 |

最重要的边界：V3 已完成 H800 实验、owner-attested 现有面板人工审核、项目许可和公共页面级发布验收；自采实拍、第二机复现与 clean-download 后 GPU smoke 仍必须按真实证据计分，不能由计划或占位文本替代。

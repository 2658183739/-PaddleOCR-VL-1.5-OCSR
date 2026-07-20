# PaddleOCR-VL OCSR V3：训练数据构建与微调报告

## 摘要

本报告记录 V3 如何把分子结构图转换为单分子 canonical SMILES，以及训练数据、评测划分、LoRA 微调、消融实验和后训练的可复现实验口径。报告只把已经写入清单、日志、评测报告、项目所有者审核声明或远端 revision 的内容称为结果；真实自采、独立机器复现等尚未完成的事项单独列为限制，不用计划替代证据。

评测集的逐目录构建、V2-1 历史面板降级、论文级留出、去泄漏和许可边界见 `EVAL_DATASET_CONSTRUCTION_REPORT_zh.md`。

任务定义是：输入一张分子结构图和固定 prompt，输出一行可以由 RDKit 解析的单分子 canonical SMILES。主指标是 RDKit canonical exact，valid SMILES、token F1、Tanimoto 和 scaffold-novel exact 用来解释错误类型，不能互相替代。

## 阅读路线：从数据到结论

整条实验链先冻结任务和数据角色，再逐层作出可撤销的工程决定，最后只打开一次 locked test。这样做的核心不是让每一步都“涨分”，而是让失败方案也能被明确拒绝，并且不把最终测试偷换成调参集。

| 阶段 | 输入与约束 | 允许作出的决定 | 通过证据 |
| --- | --- | --- | --- |
| 1. 定义任务 | 单图、固定 prompt、单分子 canonical SMILES | 统一标签空间和主指标 | RDKit 解析与 canonicalization 规则固定 |
| 2. 冻结数据角色 | training、legacy development、paper-group locked、symbolic 分离 | 哪些数据可训练、可选模或只能最终报告 | canonical molecule 与 paper group 零泄漏 |
| 3. 构建训练混合 | 七个语义组、repeat/cap、来源可信度 | 形成 A/D/E/B/C 五个受控混合 | 过滤前后数量、唯一图/分子和 source counts 可追踪 |
| 4. 探索数据因素 | 同一 V2-1 基座、同预算、`2x2x2` | 选择本轮 final 数据混合 | 两 seed 均值、seed 范围、主效应/交互和 paired CI；不宣称显著最优 |
| 5. 训练与选 checkpoint | 00 control、1400 steps、7 个 checkpoint | 选择 `checkpoint-1400` | development macro exact 优先，validity 不触发回归闸门 |
| 6. 后训练与解码 | hard replay、greedy、beam4/return4、同候选池 rerank | 采用或拒绝复杂策略 | 预设 `+0.5pp` 收益/回归闸门；hard replay 与 rerank 被拒，beam 被采用 |
| 7. 一次性 locked test | 模型、prompt、decoder 和 hash 全部冻结 | 只回答论文外推与未见 scaffold 泛化 | wild strict/scaffold/symbolic 分轨报告，结果不回流 |
| 8. 发布与审计 | Apache-2.0/NOTICE、许可矩阵、owner attestation、模型 hash | 哪些文件可公开与如何复验 | 训练原图隔离、公开 commit/revision、clean-download 与 SHA256 |

这套流程把三种不确定性分开：训练 seed 不确定性需要更多独立训练重复，development 样本不确定性使用按 `structure_id` 聚类的 paired bootstrap，论文域外推则由按 `paper_group` 留出的 locked test 回答。三个层次不能用同一个置信区间替代。

## 1. 数据构建范围

### 1.1 七部分训练数据

V3 final control 的 22,762 条记录按上游语义归并为七部分。记录数包含 repeat/cap 权重，不等于独立图片数。

| 部分 | 记录数 | 占比 | 来源属性 | 标签清洗 |
| --- | ---: | ---: | --- | --- |
| USPTO | 5,043 | 22.16% | 公开专利/OCSR 风格数据 | 原始结构标签统一 canonical SMILES；剔除坏图、多片段、非法标签 |
| UOB | 4,869 | 21.39% | 公开 OCSR benchmark 风格数据 | RDKit 解析、canonicalization、路径和重复检查 |
| real-world | 4,329 | 19.02% | 公开/项目整理的拍照、扫描、页面和手写场景集合 | 只接受已有可信 SMILES；自动生成标签不作为真值 |
| MolGrapher synthetic | 4,000 | 17.57% | 合成/生成结构图与配套标签 | 检查单分子和标签合法性；只作为训练增强 |
| USPTO-30K clean | 1,501 | 6.59% | 公开 clean 子集 | canonical 化并 cap 约 1,500 |
| USPTO-30K abbreviated | 1,507 | 6.62% | 公开 abbreviated 子集 | canonical 化并保留缩写长尾 |
| USPTO-30K large | 1,513 | 6.65% | 公开 large 子集 | canonical 化并保留大图/长分子长尾 |
| **合计** | **22,762** | **100.00%** | 7 个上游语义组 | V3 strict control |

`real-world` 是项目内部的来源组，不是一个可以单独引用的公开数据集名称。它的内部 source counts、难度计数和重复策略见 `evidence/dataset_build_report.json`。当前 final training manifest 只有集合级 source 类别覆盖，样本级 `license`、`source_url_or_doc`、`structure_id` 覆盖为 0%；公共 release 前必须补齐或隔离未知来源。

### 1.2 为什么不平均采样

UOB 和 USPTO 共同提供约 43.5% 的 printed 锚点，确保模型不会因为弱域增强而丢失基本 OCSR 能力。real-world 保留约 19.0% 以提高拍照、扫描、页面嵌入和手写场景出现频率；MolGrapher synthetic 约 17.6% 用于复杂结构和视觉扰动；USPTO-30K 三个子集各限制约 1,500 条，防止干净专利图吞掉真实场景权重。

这个比例来自三个可审计因素：V2-1 错误分层、各来源的标签可信度和 2×2 探索预算。它不是经过充分搜索得到的全局最优。当前 2×2 只有两个 seed，运行顺序也未完全平衡，因此结论只能写为“本轮探索下选择 control”，不能写成统计显著或普适最优。

### 1.3 标签清洗顺序

构建器按以下顺序处理每条候选记录：

1. 验证图片存在、可打开且路径可复现。
2. 规范化为一行单分子 canonical SMILES。
3. 使用 RDKit 检查可解析性、canonicalization 幂等性、dummy atom、多片段和明显无效标签。
4. 拒绝反应式、盐/溶剂混合物、R-group/symbolic、chemfig、`ssml_normed` 和无法稳定转换的教育标签；它们若需报告，进入独立 symbolic track。
5. 用 canonical molecule、`structure_id`、图片 ID 和 paper group 反向过滤 development/locked test，防止泄漏。
6. 保存 source、difficulty、task_type、上游 ID 和 repeat/cap 策略；缺少样本级来源的记录不能直接进入公共 release。

输入由 V2-1 clean weighted 的 23,047 条减少到 22,762 条：273 条多片段/symbolic/不收敛标签被剔除，12 条与新 held-out 分子重叠的记录被剔除。筛选前后统计和 SHA256 在 `evidence/dataset_build_report.json`、`qc/QC_REPORT_V3_zh.md` 和 `evidence/workspace_verification.json`。

## 2. 评测集划分

### 2.1 独立单位和泄漏控制

普通 OCSR 用 canonical molecule/`structure_id` 作为独立单位；MolRecBench 用 `paper_group`；同一结构的多视角照片不能被当成多个独立重复。MolRecBench 先留出整篇论文，再从留出论文中最多取 5 张进入 wild strict，训练和 locked test 的 paper group 与 canonical molecule 均为零重叠。

### 2.2 公开 benchmark 抽样比例

development 面板不是按上游数据的原始数量直接拼接，而是预先冻结目标配额，完成严格 QC 后接受剩余样本：

| 面板 | 目标配额 | strict 通过 | 选择原因 |
| --- | --- | ---: | --- |
| core | DECIMER 150、UOB 200、USPTO 200、real-world 217 | 150/193/196/214 = 753 | UOB/USPTO 作为 printed 可比锚点；DECIMER 覆盖手绘；real-world 保留稀缺场景。 |
| region | EDU-CHEMC 153、UOB 200、USPTO 200、real-world 217 | 151/193/196/214 = 754 | 在 core 基础上专门覆盖教育图和页面目标区域，供 crop/region 回归。 |
| wild strict | strict pool 1,428 条、519 篇论文 | 301 图/301 分子/62 篇论文 | 论文级留出和每篇最多 5 图，测试论文外推而非行级记忆。 |
| scaffold novel | wild strict 子集 | 134 | 诊断训练未见 Bemis-Murcko scaffold 的泛化。 |
| symbolic | MolRecBench symbolic/R-group | 460 | 标签不属于 canonical SMILES，独立报告。 |

UOB/USPTO 各约 200 条是比较锚点，不是部署场景真实比例；DECIMER/EDU 约 150 条保证手绘/教育困难样本有统计可见度；real-world 清洗后全部保留，避免稀缺场景被下采样消失。目标数与通过数的差异必须由 QC/泄漏原因解释，不能为凑数把拒绝样本放回去。准确 source counts 来自各面板 `labels.jsonl`，自动 QC 汇总见 `evidence/v2_1_eval_qc_summary.json`。

### 2.3 非公开数据的标注和 QC 状态

当前 private photo 数据仍为 0。项目所有者确认 wild strict 301、symbolic 460 以及两个 legacy development 面板已经完成离线人工审核，冻结 labels 没有审核后剔除或修订，因此现有指标无需重新计算。公开证据如下：

- 声明：`qc/MANUAL_REVIEW_ATTESTATION_zh.md` 说明审核范围、结果与公开证据边界。
- 机器记录：`qc/manual_review_attestation.json` 绑定四个冻结 labels SHA256；清单变化时声明自动失效。
- 工具边界：`qc/eval_manual_review.csv` 与 Gradio 查看器保留为后续复核模板，不被用来补造本次外部审核的逐样本记录。
- 隐私边界：公开材料不披露或虚构 reviewer 姓名、签名、分歧数量和逐样本决定。
- 自采：至少记录设备、角度、光照、采集人、时间、授权、`structure_id` 和 split；同一结构的所有视角必须留在同一 split。算法增强不能计作真实自采。

人工审核状态和证据边界见 `qc/QC_REPORT_V3_zh.md`；自动 QC 与 owner-attested 人工审核分别报告，不互相替代。

## 3. 微调和消融

### 3.1 训练策略

主线是从 `models/v2_1_export/` 继续做低学习率 LoRA SFT，而不是从原始 PaddleOCR-VL-1.5 重新开始。V2-1 已完成 OCSR 任务适配；固定预算 warm-start probe 显示原始 1.5 基座两个 development 面板 exact 均为 0，而 V2-1 continuation 宏平均 exact 为 0.341736。该对照只代表固定预算效率，不代表充分调参的原始模型上限。

LoRA 选择的工程原因是：保留已学习的 OCSR 输出格式，减少单卡 H800 的训练时间和显存，能够把预算用在数据因素、checkpoint 和生成策略的可解释对照上。训练只输出 canonical SMILES，不把 `ssml_normed`、chemfig 或反应式混入同一目标。

### 3.2 `2×2×2` 数据因素消融

因素为 strict-wild（off/on）与离线视觉增强（off/on），每个组合使用两个 seed、相同基座、250 steps、effective batch 32、学习率 `2e-5` 和相同 development 推理参数：

| 条件 | wild | augmentation | 两 seed 宏平均 exact |
| --- | --- | --- | ---: |
| 00 control | off | off | **0.341071** |
| 10 wild-only | on | off | 0.332777 |
| 01 aug-only | off | on | 0.334436 |
| 11 wild+aug | on | on | 0.339082 |

估计主效应为 wild `-0.182pp`、augmentation `-0.017pp`、交互 `+1.294pp`；dose-2 单 seed 为 0.339745。paired bootstrap 按 `structure_id` 聚类，10,000 次重采样，CI 结果在 `evidence/probe_paired_summary.md`。由于只有两个 seed、运行顺序未完全平衡，这不是显著性检验，也不是全局最优证明；本轮选择 00 control 进入 final 只是探索性模型选择。

### 3.3 Final、hard replay 和生成策略

final 使用 00 control、1400 steps、effective batch 32、学习率 `3e-5`，完整评测 200/400/600/800/1000/1200/1400 七个 checkpoint 后选择 `checkpoint-1400`：development core/region exact 为 35.59%/36.34%，macro 35.97%。300-step hard replay macro 为 35.24%，相对 final 回退 0.73pp 且 validity 同时回退，因此拒绝。

hard replay 是 300-step、学习率 `8e-6` 的困难样本 continuation，目标是修复错误分层而不是全量重训。生成阶段先比较 greedy 与 beam4/return4：macro exact 从 35.97% 提升至 42.07%，两个面板 paired cluster bootstrap 95% CI 均高于 0，因此采用 beam。随后在同一 beam 候选池运行 CPU-only `chem_light` 重排，macro 降至 39.55%，相对原始 beam 回退 2.52pp，因此拒绝。symbolic 是独立文字转写 track，未参与 canonical decoder 选模，locked symbolic 预先固定 greedy，不把 canonical beam/reranker 决策无验证地迁移过去。

一次性 locked test 在全部策略冻结后运行：wild strict 301 张 canonical exact 22.92%、valid 84.72%；其中 134 张 scaffold-novel exact 13.43%、valid 75.37%；symbolic 460 张 whitespace-normalized exact 0%、nonempty 100%。locked 结果不回流选模。完整数字与分层见 `evidence/FINAL_RESULTS.json`。

## 4. 原始模型基线与历史 4090 参考

同一历史 770 条 OCSR 诊断面板上，PaddleOCR-VL-1.5 原始权重 canonical exact 为 0.00%、valid SMILES 30.78%、token micro F1 6.59%、mean Tanimoto 0.0027；V2-1 LoRA export 为 33.77%、75.84%、70.18%、0.6849。该表来自 4090 历史仓库的同口径评测，不能和 V3 locked test 混报。

4090 代码、候选生成和 reward selector 的历史参考：[2658183739/-PaddleOCR-VL-1.5-OCSR](https://github.com/2658183739/-PaddleOCR-VL-1.5-OCSR)。V3 使用相同任务定义但重新冻结数据角色、分组泄漏和统计单位。

## 5. 复现环境和运行

本次 H800 记录：NVIDIA H800 PCIe 80GB、Driver 565.57.01、PaddlePaddle GPU 3.3.1 cu126、PaddleFormers commit `e51f911c23b41283ef6c62f8aa4a7e99291bcd11`、PyTorch 2.1.2+cu118、Transformers 4.55.4、RDKit 2025.9.6、Python 3.10。训练在 `/root/autodl-tmp`，结果持续同步到 `/root/autodl-fs/V3_results`，日志为 `logs/pipeline_master.log`。greedy 可用 4 worker；beam4/return4 因四候选显存开销改用 1 worker，4 worker OOM 的失败日志被保留为工程证据。

最小复现：

```bash
cd /root/autodl-tmp
bash V3/setup_h800_environment.sh
screen -dmS v3pipeline bash -lc 'set -o pipefail; cd /root/autodl-tmp; bash V3/run_h800_pipeline.sh 2>&1 | tee /root/autodl-fs/V3_results/logs/pipeline_master.log'
```

流程顺序固定为 `2×2×2 probe -> final checkpoint 选择 -> hard replay gate -> greedy/beam 对照 -> locked test -> artifact package`。中断后依靠阶段完成标记恢复，不删除已有输出。H800 训练日志、checkpoint hash、环境快照和最终包路径必须一起保留。

## 6. MolTrace Agent 前后端与应用闭环

为了把离线模型证据变成可复核的使用流程，V3 增加 `V3/agent_demo/` 前后端工作台。浏览器端负责图片上传与预览、图像质量诊断、beam/return/TTA 参数设置、候选比较、六步决策轨迹、运行历史和 JSON 证据导出；零运行依赖的 Node 后端负责大小与格式限制、模型状态探针、临时文件生命周期、真实 V3 推理适配和结果汇总。真实模式固定调用 `V3/scripts/infer_ocsr_transformers.py`，从同一图像召回多个候选，再将可解析性、canonicalization、片段/dummy 风险和跨视图一致性写入 trace；Agent 的作用是公开决策证据，而不是用启发式结果替代冻结评测。

系统明确区分三种状态：设置 `V3_MODEL_DIR` 时运行真实 GPU 模型；内置咖啡因样例以 `guided demo` 标记展示交互和已知标签；任意上传图在模型不可用时返回 `needs model`，不会伪造识别结果。默认不持久化原图，推理输入仅写入系统临时目录并在子进程退出后删除，历史只保留文件指纹和结果摘要。前后端已在 1440×1050 和 390×844 浏览器视口验收，并通过 5/5 Node 自动测试；这些测试证明接口、边界和 trace 可运行，不替代模型精度评测。

最小启动方式：

```bash
cd V3/agent_demo
npm start
# http://127.0.0.1:8787
```

## 7. 限制与发布状态

公共发布已完成：训练与模型证据冻结基线为 GitHub `a68b434f2a905562929c545470192b4b11f1c66c`，Hugging Face 模型 revision 为 `e496110ec222c1a70ebca287990c07dae47a2daa`，远端 Xet 权重 SHA256 为 `2a7ac278677ff56379e67933d6d81481991b755b93355fca5902cc36a7b1cc13`；包含 Agent、最终报告和答辩稿的 GitHub 交付提交记录在比赛包 `最终交付清单_V3_final.md`。当前仍缺 private photo、Docker/第二台机器独立复现、clean-download 后的 GPU smoke 和至少四 seed 的 confirmatory 复验。项目采用 Apache-2.0，并用许可矩阵隔离不适合再分发的训练原图；人工审核由项目所有者声明完成。Demo 录屏不再作为本轮交付项。完整远端验收边界见 `evidence/PUBLIC_RELEASE_VERIFICATION_zh.md`。

不得把 UOB 小子集的 70%-80% 目标当作全量保证。当前证据显示干净 printed 子域可以明显高于真实页面、手绘和教育图；V3 的工程目标是先保证口径清楚、可复现和不夸大，再用候选召回、crop 和 hard replay 在固定回归闸门内争取增益。

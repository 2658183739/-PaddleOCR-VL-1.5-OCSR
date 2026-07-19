# V3 当前缺失内容与解决方案

这份清单区分“代码已经准备好”和“证据已经真实产生”。优先级按是否阻止正式提交结论排序。

## P0：阻止正式结论

### 1. 两 seed 因子消融已完成，但 confirmatory 复验仍不足

H800 上 8 个 `2x2` factorial probe、warm-start 和增强剂量诊断均已完成，实验矩阵、训练日志、checkpoint、生成结果和 `probe_analysis.json` 已产生。00 control 的两 seed 宏平均 exact 为 `0.341071`，高于 11 的 `0.339082`；wild 主效应 `-0.182pp`、augmentation 主效应 `-0.017pp`、交互 `+1.294pp`。

仍缺：

1. 只有两个 seed，无法稳定估计 seed 方差或支持 ANOVA/p 值。
2. 运行顺序没有完全随机化或位置平衡：11 两次均先运行，01 两次均最后运行，时间、缓存和热状态可能与条件混杂。
3. 逐样本 paired CI 已输出到独立的 `probe_paired_summary.json/.md`；但这只量化固定 checkpoint 在 development 样本上的不确定性，不能替代训练 seed 重复。

解决：下一轮至少使用 4 个 seed；用平衡 Latin-square 或分块随机顺序，使每个条件均匀出现在各位置；保留同一基座、步数、batch、像素和推理参数；预先定义 `0.5pp` 或业务可接受阈值，报告效应、seed 分布和 paired bootstrap CI。

当前结论口径：本轮属于工程探索性选模，只能说“00 在本轮预算下均值最高”，不能说“统计显著最优”。

### 2. Frozen evaluation 人工审核（已按 owner attestation 关闭）

项目所有者于 2026-07-19 确认 legacy core/region、wild strict 301 和 symbolic 460 已完成离线人工审核；没有报告审核后剔除或标签修订，因此 frozen 指标无需重新统计。`qc/manual_review_attestation.json` 记录审核范围、结论和四个 labels SHA256；任何清单变化都会使声明失效。

公开证据边界是 owner-attested completion，不是公开逐样本双盲数据集。Reviewer 身份、签名、分歧统计和内部逐样本工作表不公开，也不补造。历史 `eval_manual_review.csv` 保留为后续工具模板，不作为本次外部审核的权威完成记录。

### 3. 自采实拍数据缺失

当前算法退化不能替代官方要求的自行实拍，且官方评估集高分倾向是至少 1000 个真实实例。

解决：

1. 最低可信版本按 `runbooks/PRIVATE_COLLECTION_PROTOCOL_zh.md` 采集至少 80 个结构、每结构 4 条件，覆盖至少 2 台设备和 4 类真实退化。
2. 若目标是官方数据规模最高档，当前 locked wild 只有 301 张，还需至少 699 张审核通过的真实 eval 图；按每结构 4 视角计算，至少需要 175 个 eval-only 结构，考虑淘汰应准备约 195-200 个。
3. 在 `private_photo_collection.csv` 预先填写 `split=train/eval`。eval-only 批次不得进入训练；同一结构所有照片必须同 split。
4. 运行 `import_private_photo_data.py`，脚本会同时检查自采 eval 对现有训练、自采 train 对现有评测以及自采内部的 canonical 分子零重叠。

完成标准：最低版本不少于 80 个独立结构；高分规模版本 locked 真实实例总数 `>=1000`；train/test 结构与 canonical 分子重叠均为 0；双人复核；算法增强与真实照片在来源字段中严格分开。

### 4. Final model 已冻结（已解决）

1400-step final 已在 H800 上完成，数据为 `train_v3_a_control.jsonl`。200-1400 的 7 个 checkpoint 均完成 development 评测，最终选择 `checkpoint-1400`，macro exact 35.97%。merged export、remote code、模型 hash、20 张 runtime smoke 和可恢复 checkpoint 均已进入最终包。

300-step hard replay macro exact 35.24%，相对 final 回退 0.73pp 且 validity 回退，因此按预设闸门拒绝。最终模型目录为 `models/final_best_export/`，证据见 `final_checkpoint_selection.json`、`final_vs_hard_replay.json` 和 `training_artifacts/`。

状态：机器训练、选模与导出已完成；第二台机器从零复现仍归入环境复现缺口。

### 5. Locked test 与现有人工审核均已冻结（已解决）

模型、prompt 和生成策略冻结后，locked run `20260719_000225` 已一次性执行。wild strict 301 张 exact 22.92%、valid 84.72%；scaffold-novel 134 张 exact 13.43%、valid 75.37%；symbolic 460 张独立 track exact 0%、nonempty 100%。模型/标签/prompt/策略 hash 已写入 `locked_test_manifest.sha256`，结果未回流调参。

人工审核完成状态由项目所有者声明并与 frozen labels SHA256 绑定。公开仓不披露审核身份或内部逐样本记录，因此不宣称公开双盲一致率。现有 manifest、运行日志、attestation 与 labels hash 必须共同保留；若未来修改 labels，只允许对冻结预测重新统计，不得返回重新选模。

### 6. 官方提交包与真实性核验材料缺失

官方要求的最终材料不只是 README 和模型目录。可能获奖作品会核验实验可复现性、评估集真实性以及评测结果与提交材料的一致性。

解决：

1. 评估集包：images/documents、annotations、task description、evaluation script、来源/规模/类别/难度说明。
2. 训练数据构建报告：采集/合成关键代码、标注规范、工具、QC 流程和统计图。
3. 公共 GitHub：训练/评测代码、文档、Demo、Apache-2.0/NOTICE；模型权重不进入普通 Git history。
4. Hugging Face：final merged model、完整 model card、使用限制、评测结果和文件 hash。
5. 决赛材料：18 页 HTML/PPT；Demo 录屏按本轮范围取消。
6. 真实性包：每次训练的 config、stdout、trainer state、环境版本、checkpoint/model/data hash 和评测命令。

完成标准：在一台新 A100 环境按公开说明完成 20-step smoke 和固定小样本复评；所有报告数字能追溯到 JSON/CSV/日志；正式材料在官方 2026-07-20 截止前提交。

## P1：影响技术可信度和评分

### 7. 后训练消融部分已纳入流水线，reward/crop 完整复评仍缺失

本轮已真实执行 300-step hard replay、greedy 与 beam4/return4 对照，以及同一 beam 候选池上的 CPU-only chem-light 重排。hard replay 回退 0.73pp，被拒绝；beam 相对 greedy 提升 6.10pp，被采用；chem-light 相对 beam 回退 2.52pp，被拒绝。现有 reward head/targeted crop 尚未在 V3 final 上按固定候选池做完整 P1-P3 消融。

已补成本敏感门槛：复杂候选必须至少提升 `0.5pp`，且各面板和 validity 回归不超过 `0.5pp`；否则保留更早 final 或 greedy。final vs hard replay、greedy vs beam 均会输出 `structure_id` 聚类 paired CI。

解决：先比较 candidate oracle，再在同一候选池比较 heuristic 与 reward；crop 只在 weak-layout dev 开启；所有 margin/router 用 cross-validation。

完成标准：每一阶段只改变一个机制，能回答收益来自召回、选择器还是 crop。

### 8. 实验环境已记录，但容器级环境锁仍缺失

当前已有 `setup_h800_environment.sh`、实际 H800/driver/Paddle/PaddleFormers/Transformers/RDKit 版本和运行诊断；最终包已保存 `evidence/runtime/pip-freeze-final.txt` 与 `nvidia-smi-final.txt`。仍没有 Dockerfile/镜像 digest，也没有第二台机器从零复现证据。

解决：训练机导出 `conda env export` 或 `pip freeze`，保存 `nvidia-smi`、CUDA/cuDNN、Paddle commit；补 Dockerfile 或启动脚本。

完成标准：第二台 A100 能从零加载数据并完成 20-step smoke。

### 9. 样本级来源与许可证仍不完整

自动审计结果：final control 22,762 条记录的 source 类别覆盖为 100%，但样本级 `license`、`source_url_or_doc` 和显式 `structure_id` 覆盖均为 0%。两个 legacy dev 面板以及 wild strict 301、symbolic 460 的三项覆盖均为 100%。这说明训练清单当前只能追踪到集合/来源类别，不能逐样本证明为自采或明确公开许可。

解决：为每条样本补 `source_type/source_url/license/collector/capture_time/transform_parent`；无法证明许可的样本不进入公开 release。

完成标准：训练、development、test 均有来源覆盖率统计；未知来源为 0 或被单独隔离。

### 10. 自动化测试覆盖仍不完整

本地测试为 `29/29` 通过，H800 在打包前运行同一套测试，覆盖 checkpoint 近似并列优先较早模型、validity-floor 排除、release-readiness/BOM 人工复核统计、复杂候选 0.5pp 门槛、paper train/test 零重叠、locked canonical 唯一性、canonicalization 幂等与非法标签一致拒绝、cluster bootstrap 独立单位、private structure 确定性/显式分组、推理分片完整性、runtime export 元数据、实验矩阵回填、paired 汇总、最终报告渲染、胜出 checkpoint 可恢复文件收集，以及人工 QC 工具的 reviewer/adjudicator 原子写入。同时已有 py_compile、YAML、图片路径、RDKit、分子/论文泄漏、每篇上限和 hash 验证。本次已人工连续运行两次全量 builder，A/D/E/B manifest、locked labels 和 build report 的 SHA256 完全一致。仍缺完整 builder 的小型端到端 fixture、2x2 manifest 计数和自动化 CI 重建 hash 测试。

解决：继续增加不依赖全量 14 GiB 资产的小型 end-to-end fixture，并在 CI 中运行。

完成标准：一条命令运行全部测试；重建两次的关键 manifest hash 一致。

### 11. Development 的历史调参偏差

legacy core/region 已被 V2-1 多次使用，只适合作为 development 和历史连续对比。

解决：所有 confirmatory 结论只在 paper-group locked wild 与新 private test 上报告；README 和技术报告持续标明 legacy role。

完成标准：任何最终表格不会把 legacy dev 标为“unseen test”。

### 12. Probe 与 beam paired 统计已完成，final/hard paired 统计仍缺

两 seed 条件均值、seed range、主效应和交互已经产生。16 组 per-seed/per-panel 比较已按 `structure_id` 做 10,000 次 paired cluster bootstrap；除 wild-only/seed2 明确负向外，其余主因子比较 CI 均跨 0。greedy 对 beam 的两个 development 面板也已完成 paired cluster bootstrap，exact delta 的 95% CI 分别为 `[4.17pp, 8.01pp]` 和 `[4.10pp, 8.06pp]`。由于只有两个训练 seed 且运行顺序不平衡，仍不报告 ANOVA p 值或配比“显著最优”。final 对 hard replay 的逐样本 details 未进入最终包，因此当前只能报告面板级回退，不能补造 paired CI。

解决：development 用 `structure_id`，wild 用 `paper_group`，private 用 `structure_id`；报告 image N 和 cluster N、95% CI、净新增正确与回归样本。

完成标准：所有主要涨分声明都有 paired CI；聚类数据不使用 image-level 显著性。

### 13. 模型卡、数据卡、复现指南与最终数字已建立（已解决）

`MODEL_CARD_zh.md`、`DATASET_CARD_zh.md` 和 `REPRODUCTION_GUIDE_zh.md` 已建立。最终训练、hard replay、生成和 locked test 的数字已由 `scripts/build_final_report.py` 从 JSON 证据生成到 `evidence/FINAL_RESULTS_zh.md`，并修复了二阶段 generation baseline/candidate 标签硬编码问题。

完成标准：报告数字可追溯到 JSON/CSV；不写未执行实验；公开内容不泄漏受限数据。

## P2：影响展示和长期复现

### 14. Demo 代码保留，录屏不在本轮范围

`demo/app.py` 已有。最终模型真实加载、单样本与 sharding 推理 smoke 已在 H800 保存证据；交互式 Gradio 录屏按项目所有者决定取消，因此不能把录屏列作已完成加分证据。

### 15. Git/版本发布状态不完整

当前工作区根目录没有可用 Git repository 状态，无法用 commit 精确指向版本。

解决：修复或新建受控仓库，提交 V3 代码/小型 manifest；大模型和大数据用 release hash 或 LFS/对象存储管理。

### 16. 社区与开源材料缺失

当前 model card、dataset card、复现指南、`CONTRIBUTING.md`、Apache-2.0、NOTICE、许可矩阵和复现 issue 模板已建立。逐样本训练数据许可清单仍不完整，因此训练原图/JSONL 不进入公共 release；模型下载地址在 GitHub/HF 发布完成后回填 revision。

### 17. PaddleOCR-VL-1.6 仍未核验

解决：只接受官方可验证权重和文档；先跑 200 样本兼容性与同预算 probe。没有至少 +2pp development 提升且 validity 不降，不切主线。

## 建议完成顺序

```text
人工自采/QC（并行）
    +
H800 2x2 两 seed（已完成）
    -> development 选模与 final SFT（已完成）
    -> hard replay/beam/chem-light（已完成并按闸门取舍）
    -> 导出、冻结 hash、一次性 locked test（已完成）
    -> owner-attested 人工审核 + Apache-2.0/NOTICE（已完成）
    -> GitHub/HF 公开发布（进行中）
    -> 自采实拍与第二机复现（后续限制）
```

最容易犯的错误是先看 locked test，再根据结果继续改模型。这个动作会让 locked test 变成 development，之前的“最终分数”失去无偏含义。

# PaddleOCR-VL OCSR V3 模型卡

## 模型用途

该模型接收单张分子结构图，输出一行 SMILES。目标是稳定生成可由 RDKit 解析的单分子 canonical SMILES，不适用于谱图解析、反应预测、药效预测或医疗决策。

## 模型来源与后训练

- 原始架构：PaddleOCR-VL-1.5 系列 remote-code 模型。
- 任务基座：`models/v2_1_export/`，即已经完成 OCSR 任务适配的 V2-1 导出模型。
- 主训练：低学习率 LoRA continuation SFT。
- 数据配比选择：`2x2` 因子消融，比较 strict-wild 与离线退化增强的主效应和交互。
- 辅助对照：原始 1.5 warm-start、增强 dose-2。
- 后训练：300-step hard replay 已执行，development macro exact 从 35.97% 降至 35.24%，因此拒绝并保留 `checkpoint-1400`。
- 生成策略：beam4/return4 相对 greedy 从 35.97% 提升至 42.07%，通过闸门；同候选池 chem-light 重排降至 39.55%，因此最终使用原始 beam。

hard replay 和 beam4 都是更昂贵的候选，只有 macro exact 至少提升 `0.5pp`、单面板和 validity 不出现超过 `0.5pp` 回归时才替换简单基线；否则保留较早 final 或 greedy。

本轮两 seed 探索选择 00 control 数据进入 final。完整数字见 `evidence/probe_analysis.json` 和最终自动生成的 `evidence/FINAL_RESULTS_zh.md`。

## 与 PaddleOCR-VL-1.5 原始模型的直接基线

为了避免把“V2-1 微调基线”误写成原始模型基线，下面保留同一历史 770 条 OCSR 诊断面板上的直接对照。原始 PaddleOCR-VL-1.5 没有 OCSR canonical-SMILES 任务适配；V2-1 export 则已经完成 LoRA SFT，因此两者用途不同。

| 模型 | canonical exact | valid SMILES | token micro F1 | mean Tanimoto | 口径 |
| --- | ---: | ---: | ---: | ---: | --- |
| PaddleOCR-VL-1.5 原始权重 | 0.00% | 30.78% | 6.59% | 0.0027 | 4090 历史 `ocsr_realworld_mixed_eval_v1p1`，未做 OCSR SFT |
| V2-1 LoRA export | 33.77% | 75.84% | 70.18% | 0.6849 | 同面板、同评测脚本，已做 single-line OCSR SFT |
| V3 final + beam | 22.92% | 84.72% | 77.02% | 0.5715 | 一次性 `wild_strict_v3` locked，301 张/62 篇留出论文 |

上表前两行是历史 770 条 mixed development，第三行是新的 paper-group locked test，不能纵向计算“提升”。当前同口径 V3 development 对照为 greedy 35.97%、beam4/return4 42.07%，beam 的净提升为 `+6.10pp`。

另有 H800 warm-start probe：在相同 250-step 预算下，原始 1.5 基座两个 development 面板 exact 均为 0；这只能说明固定预算下 continuation 更有效，不能当作“原始模型充分调参后的理论上限”。4090 参考实现和候选选择定义见 [历史 V2-1 仓库](https://github.com/2658183739/-PaddleOCR-VL-1.5-OCSR)。

## 加载与推理

最终 merged 模型位于 `models/final_best_export/`，包含 tokenizer、processor、generation config、PaddleOCR-VL remote code 和 safetensors 权重。

```bash
python V3/scripts/infer_ocsr_transformers.py \
  --model-dir V3/models/final_best_export \
  --benchmark-jsonl your_labels_or_inputs.jsonl \
  --project-root . \
  --output-jsonl predictions.jsonl \
  --prompt-file V3/configs/prompt.txt \
  --device cuda \
  --torch-dtype bfloat16 \
  --num-beams 4 \
  --num-return-sequences 4 \
  --save-candidates
```

最终 canonical 策略使用 beam4/return4，H800 80GB 上固定单 worker；四个 beam worker 会 OOM。若只做 greedy 功能验证，可使用 `run_sharded_inference.py --workers 4`，但该结果不等同于最终冻结策略。symbolic 是独立 track，因没有独立 decoder development 消融而预先固定 greedy、4 workers。

## 评测口径

- 主指标：RDKit canonical exact。
- 闸门指标：valid SMILES rate。
- 辅助指标：raw exact、Tanimoto、stereo exact、来源/难度分层。
- 泛化诊断：Bemis-Murcko scaffold-novel。
- symbolic 标签独立使用文字规范化 exact，不混入 canonical 主分数。

legacy core/region 是历史 development，不是未触碰测试集。wild strict 仅在模型与生成策略冻结后运行一次，并按 `paper_group` 聚类解释。

最终 locked 结果：wild strict canonical exact 22.92%、valid 84.72%；scaffold-novel exact 13.43%、valid 75.37%；symbolic whitespace-normalized exact 0%、nonempty 100%。symbolic 不混入 canonical 主分数。完整分层、F1、Tanimoto 和来源统计见 `evidence/FINAL_RESULTS.json`。

## 已知限制

- 只有两个 seed，且运行顺序未完全平衡；probe 结论是探索性的。
- locked wild/symbolic 已由项目所有者确认完成离线人工审核；公开仓库只提供 owner attestation 与冻结清单 hash，不披露审核人身份和逐样本内部记录。
- private-photo 真实自采评测为 0，算法退化不能替代实拍。
- 部分历史训练样本缺样本级许可证与来源字段，不应在未清理前直接公开全部数据。
- 模型可能输出合法但结构错误的 SMILES；必须由下游化学工具和人工流程复核。

人工审核的公开证据见 `qc/MANUAL_REVIEW_ATTESTATION_zh.md` 和
`qc/manual_review_attestation.json`。声明绑定四个冻结 labels SHA256；项目所有者
确认没有审核后剔除或标签修订，因此 locked 指标无需重新计算。公开材料不虚构
审核人姓名、签名、分歧数量或逐样本决定。

## 许可证与发布

项目代码与本项目产生的派生模型权重采用 Apache License 2.0，与
`PaddlePaddle/PaddleOCR-VL-1.5` 的官方许可证一致。第三方数据不随模型仓发布，
其来源、上游许可证据和未声明许可的边界见 `DATA_LICENSES_AND_ATTRIBUTION_zh.md`
及 `NOTICE`；项目许可证不会替代或扩大第三方数据条款。

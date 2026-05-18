# 单阶段真实加权 LoRA 微调方案

## 结论

当前 V2 不建议继续按 phase0/phase1/phase2/phase3 分段跑。推荐只跑一条线：

`PaddleOCR-VL-0.9B` -> `Single-stage Real-Weighted LoRA SFT` -> 导出 checkpoint -> 在 `ocsr_realworld_mixed_eval_v1p1` 上评测。

旧的多阶段入口已归档到：

```text
V2/archive/old_multiphase_training_20260512/
```

## 为什么这样做

1. 目标输出是 `canonical SMILES`，因此单阶段训练只使用 canonical-SMILES 样本。
2. 原始 EDU-CHEMC phase0 输出是 `ssml_normed/chemfig`，不直接并入本次单阶段训练，避免输出格式被带偏。
3. V2 原 phase3 中真实噪声样本偏少，因此对 `real_world` 和视觉鲁棒样本做确定性重复加权。
4. 训练仍保留 UOB/USPTO 的主分布覆盖，但下采样干净合成样本，避免模型过度偏向清晰渲染图。
5. 默认过滤 `ocsr_realworld_mixed_eval_v1p1` 中已出现的 canonical SMILES，避免同分子记忆影响评测可信度。

## 方法选择

最终选择 LoRA，而不是全参 SFT 或 RL：

- 全参 SFT 成本高、显存压力大，在当前 4090 单卡流程下不适合作为主线。
- 直接 RL/RLHF 对 OCSR 不划算，因为 canonical SMILES 是稀疏精确匹配奖励，错误样本的可解释 reward 难设计，短时间容易把训练复杂度拉高。
- LoRA rank 16 能在成本可控的情况下提升结构符号、长 SMILES、立体化学和真实噪声样本的适配能力。
- 本方案的创新点不是多阶段，而是格式对齐 + 真实样本重权重 + 合成样本限额 + checkpoint 级评估选择。

## 数据配比

生成脚本：

```bash
python V2/scripts/build_singleline_rw_sft_dataset.py --project-root .
```

输出训练集：

```text
V2/data/sft_materialized/train_singleline_rw_messages.jsonl
```

配比策略：

- `real_world`: repeat 5
- `molgrapher_synthetic`: repeat 2
- `uob`: repeat 1
- `uspto`: repeat 1
- `decimer`: repeat 2，如果输入训练集里存在该来源
- `uspto30k_clean`: cap 1500
- `uspto30k_abbreviated`: cap 1500
- `uspto30k_large`: cap 1500
- `ocsr_realworld_mixed_eval_v1p1` 中已出现的 canonical SMILES：过滤

当前已生成训练集审计结果：

```text
V2/reports/singleline_rw_dataset_audit.json
```

数据统计报告：

```text
V2/reports/singleline_rw_dataset_stats.json
```

关键结论：

- 总样本：`22807`
- 过滤评测集同 SMILES 原始样本：`397`
- 缺失图片：`0`
- 不可读图片：`0`
- 空输出：`0`
- 非 SMILES 输出：`0`
- RDKit 非法 SMILES：`0`
- 与 `ocsr_realworld_mixed_eval_v1p1` 的 ID 重叠：`0`
- 与 `ocsr_realworld_mixed_eval_v1p1` 的图片名重叠：`0`
- 与 `ocsr_realworld_mixed_eval_v1p1` 的 canonical SMILES 重叠：`0`
- SMILES 长度 p99：`265`，最大：`793`
- 图片面积 p95：`1048576`，最大：`4722138`

## 训练命令

```bash
bash V2/run_4090_lora_singleline_rw.sh
```

这个脚本会先重新生成 `train_singleline_rw_messages.jsonl`，然后启动训练。

核心配置：

- 配置文件：`V2/configs/ocsr_lora_singleline_rw_4090.yaml`
- LoRA rank：16
- 学习率：`1.0e-4`
- warmup：`0.03`
- max steps：`1600`
- effective batch：`4 * 8 = 32`
- 保存间隔：`200 steps`
- 训练内验证集：`ocsr_realworld_mixed_eval_v1p1` 转换后的 message jsonl

4090 说明：

- 当前配置按 24G 4090/4090D 设计，使用 `per_device_train_batch_size=4` 和 `gradient_accumulation_steps=8`。
- 这样保持有效 batch 为 `32`，与此前 `2 * 16` 一致，学习率不用重新放大。
- `max_seq_len=4096` 对当前训练集基本够用：按 PaddleOCR-VL 默认图像预处理粗估，只有约 `9/22807` 个超大图+超长 SMILES 样本可能超过 4096，且都来自 `uspto30k_large` 合成长分子；为了 4090 稳定性，第一轮先不升到 8192。
- 预计训练时长：4090 单卡约 `2.5-4 小时`；如果因 OOM 改成 `per_device_train_batch_size=2`、`gradient_accumulation_steps=16`，预计约 `3.5-5 小时`。
- 脚本默认使用 `models/PaddleOCR-VL-0.9B`，如后续要试 1.5 或其他本地模型，可临时指定 `MODEL_DIR=/path/to/model bash V2/run_4090_lora_singleline_rw.sh`。
- 如果出现 OOM，不新建配置文件，直接临时覆盖回保守设置：

```bash
CUDA_VISIBLE_DEVICES=0 paddleformers-cli train V2/configs/ocsr_lora_singleline_rw_4090.yaml model_name_or_path="$PWD/models/PaddleOCR-VL-0.9B" per_device_train_batch_size=2 gradient_accumulation_steps=16
```

训练日志：

```text
V2/outputs/singleline_rw_lora/train_singleline_rw.log
```

checkpoint：

```text
V2/outputs/singleline_rw_lora/checkpoint-*
```

如果第一轮训练后还想从某个 checkpoint 继续训练，可以用同一个脚本续训：

```bash
RESUME_FROM_CHECKPOINT=V2/outputs/singleline_rw_lora/checkpoint-800 bash V2/run_4090_lora_singleline_rw.sh
```

续训仍然使用同一份配置和同一个输出目录，不新增第二套方案。

## 报告写法

本方案可在技术报告中描述为：

> 我们采用单阶段真实加权 LoRA SFT。训练样本统一为 canonical SMILES 输出格式，在不引入多阶段训练成本的前提下，通过确定性样本重权重提升真实拍照、扫描退化、文档嵌入、复杂结构和视觉扰动样本的训练占比。EDU-CHEMC 原始 ssml_normed 标签不直接混入主训练，以避免标签空间冲突；其转换流程仅用于评测集构建与数据质量分析。

可以补充为“微调策略与创新”：

> 与标准 LoRA 相比，本方案引入了面向 OCSR 的数据配比策略：首先过滤所有非 canonical-SMILES 标签，保证输出空间一致；其次过滤评测集已出现的 canonical SMILES，避免同分子记忆影响评测可信度；随后按真实世界视觉复杂度对样本进行重权重，将拍照、扫描退化、文档嵌入、手写与多结构页面样本的训练权重提高；最后限制干净合成样本规模，降低模型对渲染风格的依赖。训练过程中每 200 step 保存 checkpoint，并使用混合真实评测集进行训练内验证和训练后 checkpoint 选择，从而以单次训练获得不同训练步数下的对比结果。

## 评测

训练完成后，用最新 checkpoint 跑：

```bash
bash V2/run_4090_eval_singleline_rw.sh
```

如果要评测这条线的所有 checkpoint：

```bash
bash V2/run_4090_eval_singleline_rw.sh --all-checkpoints
```

推荐最终按 `mixed_v1p1` 分数选 checkpoint；如果两个 checkpoint 分数接近，优先选 RDKit-valid 率高、短 SMILES/长 SMILES 分项更均衡的版本。

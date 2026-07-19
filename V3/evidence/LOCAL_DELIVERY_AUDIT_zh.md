# V3 本地最终交付审计

## 1. 云端包与关机

- 云端包：`v3_final_model_scripts_evidence_20260719_002042.tar.gz`
- 大小：`1,899,176,807` bytes
- SHA256：`aa3fd7f2e5c9f640ed5a9e66fbc2b8d54f537442f7bdbd27ce0d6ebd8a35e4e1`
- 本地下载、哈希和解压检查均通过。
- `/usr/bin/shutdown -h now` 于 `2026-07-19 00:32:12 +08:00` 被远端接受；随后 SSH 端口拒绝连接，符合实例已关闭状态。

## 2. 最终模型

- 目录：`V3/models/final_best_export/`
- 权重：`model-00001-of-00001.safetensors`，`1,917,255,968` bytes
- 权重 SHA256：`2a7ac278677ff56379e67933d6d81481991b755b93355fca5902cc36a7b1cc13`
- `config.json` SHA256：`7757d2584da82c862eee7f44c808a2f98697800ebff9070c94dc7f99edfa79ba`
- tokenizer、processor、generation config 和 PaddleOCR-VL remote code 均存在。

## 3. 最终决策与指标

| 项目 | 结果 | 决策 |
| --- | ---: | --- |
| checkpoint-1400 greedy development macro exact | 35.97% | final checkpoint |
| hard replay development macro exact | 35.24% | `-0.73pp`，拒绝 |
| beam4/return4 development macro exact | 42.07% | `+6.10pp`，采用 |
| chem-light rerank development macro exact | 39.55% | 相对 beam `-2.52pp`，拒绝 |
| locked wild strict exact / valid | 22.92% / 84.72% | 一次性结果 |
| locked scaffold-novel exact / valid | 13.43% / 75.37% | 一次性子集结果 |
| locked symbolic exact / nonempty | 0.00% / 100.00% | 独立 track |

`FINAL_RESULTS.json` SHA256：`6007dda2139caa1a56fcd24f50e15cc3f319d6eccfd754987f10f6be926adcb6`。

## 4. 代码与展示验证

- 本地 `unittest discover`：`29/29` 通过，日志为 `local_unittest_20260719.log`。
- 答辩 PPT：10 页，逐页渲染人工检查完成；10 个 layout JSON 的越界对象数为 0。
- PPT SHA256：`aa9af01ac2f08a09bf7f526ea4342957868bca1e57503c03804841fcd05921fb`。
- 自动报告生成器已修复二阶段 generation baseline/candidate 标签硬编码问题，并补回归断言。

## 5. 不能误写为完成的内容

- 项目所有者已确认 legacy/wild/symbolic 离线人工审核完成；公开证据为 owner attestation 与 frozen labels SHA256，不提供或虚构逐样本双盲记录。
- private photo 自采评测为 0；算法退化不能算真实采集。
- 训练样本级 license/source URL/structure ID 不完整。
- 项目 LICENSE 尚未由所有者确定。
- 公共 GitHub/Hugging Face 的最终上传与远端复验、第二台机器从零复现尚未完成；Demo 录屏按本轮范围取消。

这些缺口不影响本次 H800 机器实验已完成的事实，但会影响比赛的数据质量、开源和真实性评分。

# V3 提交与发布指南

本指南区分三类交付物：比赛提交包、GitHub 代码仓和 Hugging Face 模型仓。三者不能不加筛选地使用同一个目录。

## 1. 当前可直接使用的本地文件

- 比赛提交包：工作区根目录 `V3_submission_ready_20260719.tar.gz`
- 包校验：`V3_submission_ready_20260719.tar.gz.sha256`
- 最终模型：`V3/models/final_best_export/`
- 最终结果：`V3/evidence/FINAL_RESULTS.json` 与 `FINAL_RESULTS_zh.md`
- 答辩 HTML：`V3/outputs/PaddleOCR_VL_OCSR_V3_scientific_final.html`
- 答辩 PPT：`V3/outputs/PaddleOCR_VL_OCSR_V3_scientific_final.pptx`
- 提交检查：`V3/SUBMISSION_CHECKLIST_zh.md`

提交比赛前先重新计算 SHA256，并确认与同名 `.sha256` 一致。不要从旧的 `V3.tar`、`platform_migration_bundle` 或历史 V2-1 目录取模型。

## 2. GitHub 发布边界

建议公开：

- `README.md`、训练报告、数据卡、模型卡、复现指南和官方反馈响应；
- `scripts/`、`tests/`、`runbooks/`、`configs/`、`demo/`；
- 不含受限原图的统计证据、示例 manifest 和最终结果 JSON；
- `CONTRIBUTING.md`、项目所有者最终确认的 `LICENSE`、issue 模板。

暂不公开：

- 无样本级许可证/来源证明的训练数据和图像；
- 模型权重直接进入普通 Git history；
- SSH 密钥、密码、访问令牌、平台连接信息；
- 包含未授权论文原图或私人信息的日志/预测文件。

项目代码与派生权重已确定采用 Apache-2.0，第三方数据许可不由项目 LICENSE 扩张。正式源码发布使用已登录的 [GitHub 仓库](https://github.com/2658183739/-PaddleOCR-VL-1.5-OCSR)，保留历史提交并新增完整 `V3/`；模型权重不进入普通 Git history。

## 3. Hugging Face 模型仓

上传目录以 `models/final_best_export/` 为准，其中已经包含 safetensors、config、tokenizer、processor、generation config 和 PaddleOCR-VL remote code。发布前还需：

1. 把 `MODEL_CARD_zh.md` 整理为模型仓 `README.md`；
2. 在 YAML metadata 中填写 `license: apache-2.0`、base model、pipeline tag 和语言；
3. 保留 locked/development 角色说明，不把 42.07% development 写成 unseen test；
4. 写明最终 canonical 推理使用 beam4/return4，H800 80GB 使用单 worker；
5. 附模型权重 SHA256：`2a7ac278677ff56379e67933d6d81481991b755b93355fca5902cc36a7b1cc13`。

本轮已发布到 [Hugging Face 模型仓](https://huggingface.co/L2658183739/PaddleOCR-VL-1.5-OCSR)，V3 final revision 为 `e496110ec222c1a70ebca287990c07dae47a2daa`；V2-1 仍可由仓库历史追溯。远端 Xet 元数据已核对权重 SHA256；从第二台 GPU 机器 clean-download 后的加载与 20 条 beam smoke 仍需单独执行。

## 4. 比赛材料的一致性

提交页面只使用以下统一口径：

- development greedy macro exact：35.97%；
- development beam4/return4 macro exact：42.07%；
- locked wild strict exact/valid：22.92%/84.72%；
- locked scaffold-novel exact/valid：13.43%/75.37%；
- symbolic 为独立 track，exact 0%、nonempty 100%，不混入 canonical 主分数。

PPT、README、模型卡和训练报告已经按 `FINAL_RESULTS.json` 回填。若人工双审剔除错误标签，只能对已经冻结的预测重新统计，不能返回修改模型、prompt 或生成策略。

## 5. 发布后验收

GitHub 源码提交：`a68b434f2a905562929c545470192b4b11f1c66c`。Hugging Face 模型 revision：`e496110ec222c1a70ebca287990c07dae47a2daa`。页面级文件、许可证、模型卡和权重 SHA256 验收已完成，详见 `evidence/PUBLIC_RELEASE_VERIFICATION_zh.md`。

1. 在新目录下载 GitHub 代码和 HF 模型。
2. 按 `REPRODUCTION_GUIDE_zh.md` 安装环境。
3. 运行 `python -m unittest discover -s V3/tests -v`。
4. 对自有 20 张图片运行 beam4/return4 smoke。
5. 核对模型权重、结果 JSON 和提交包 SHA256。
6. 保存公开 URL、commit、HF revision 和下载复验日志。

当前仍未补齐的是逐样本训练数据许可、自采实拍、第二台机器从零复现和容器 digest。项目 LICENSE 与 owner-attested 人工审核已经完成；Demo 录屏按本轮范围取消。GitHub/HF 的公开发布与远端复验必须由实际 URL、commit/revision 和 hash 证明，不能由占位文本代替。

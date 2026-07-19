# V3 比赛提交检查清单

## A. 云端流水线完成后立即核对

- [x] `/root/autodl-fs/V3_results/evidence/all_pipeline_complete.txt` 存在。
- [x] `final_package_path.txt` 指向的 tar.gz 和同名 `.sha256` 均存在。
- [x] 本地 `Get-FileHash` 与云端一致：`aa3fd7f2e5c9f640ed5a9e66fbc2b8d54f537442f7bdbd27ce0d6ebd8a35e4e1`。
- [x] `models/final_best_export/` 包含 remote-code、processor、tokenizer、generation config 和 safetensors。
- [x] `evidence/FINAL_RESULTS.json` 与 `FINAL_RESULTS_zh.md` 已由同一 builder 重建。
- [x] final checkpoint、hard replay、generation policy 和 locked manifest 均有 JSON/hash 证据。
- [x] `evidence/training_artifacts/resume/` 包含胜出 checkpoint 的 LoRA、optimizer、scheduler、RNG 和 trainer state。

## B. 官方评估集材料

- [x] 评估图片、annotation、task description 和 evaluation script 已进入本地最终包。
- [x] 项目所有者确认 wild strict 301 条完成离线人工审核；完成声明与 frozen labels SHA256 绑定。
- [x] symbolic 460 条的人工审核范围和独立文字转写评价口径单独说明。
- [x] legacy development、wild strict 和 symbolic 的 owner-attested 完成状态已写入 QC 报告；`eval_manual_review.csv` 明确保留为工具模板，不冒充本次外部审核的逐样本记录。
- [ ] 如冲击 `>=1000` 真实实例档位，再补至少 699 张审核通过的真实 eval 图；同结构多视角按 `structure_id` 聚类。
- [ ] 算法退化不计作自行实拍，实拍记录包含设备、时间、角度、光照、采集人和授权。

## C. 训练数据与许可

- [ ] final 训练清单逐样本补齐 `license`、`source_url_or_doc`、`structure_id` 或可追踪的等价字段；本轮未补齐，不公开该清单。
- [x] 无法证明逐样本许可的历史训练原图和 JSONL 已从公共 release 隔离，只发布统计、代码与上游归属。
- [x] 数据构建报告说明过滤前后数量、七部分比例、重复/cap、真实/合成边界和泄漏检查。
- [x] 项目所有者确定项目代码与派生权重使用 Apache-2.0；`NOTICE` 与许可矩阵明确第三方数据不受项目许可证扩张。

## D. 代码、模型与复现发布

- [ ] GitHub 包含 README、训练/评测脚本、tests、runbooks、Demo、LICENSE、CONTRIBUTING 和 issue 模板。
- [ ] Hugging Face 包含 final merged model、model card、限制、结果、文件 hash 和最小推理示例。
- [ ] 大文件使用 LFS、release asset 或对象存储，不直接塞入普通 Git history。
- [ ] 在第二台 A100/H800 上从零完成环境安装、20-step smoke 和固定小样本复评。
- [ ] 保存 Dockerfile/镜像 digest，或明确记录基础镜像名称与 tag。

## E. 展示与提交页面

- [x] 18 页 HTML/PPT 覆盖任务、训练/评测配比、清洗防泄漏、LoRA、消融、checkpoint、失败后训练、decoder、locked test、许可、人审与复现；已逐页渲染检查。
- [x] Demo 录屏按本轮提交范围取消，不把未录制视频列为已交付证据。
- [x] 当前 README、模型卡、训练报告和 PPT 数字均来自 `FINAL_RESULTS.json`，未改写成更高值。
- [x] 文档明确标注 legacy development、locked wild、symbolic 和 private-photo 的不同角色。

## F. 绝不能做

- [x] 不根据 locked test 结果返回修改模型、prompt 或生成参数。
- [x] 不把两个 seed 写成统计显著结论。
- [x] 不把重复渲染或同结构多视角当成独立样本扩大 N。
- [x] 不把自动 QC 写成双人人工审核。
- [x] 不把算法退化写成真实采集。
- [x] 不在公开包、日志或 README 中保留 SSH 密码、访问令牌和私有密钥。

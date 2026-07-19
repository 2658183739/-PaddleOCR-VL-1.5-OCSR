# 发布状态

该目录是经过模型权重/训练图片排除和敏感信息扫描后发布的 GitHub 源码快照。公开仓库为 `https://github.com/2658183739/-PaddleOCR-VL-1.5-OCSR`。

已关闭：

1. 项目代码与派生权重采用 Apache-2.0，并提供 NOTICE 与逐来源许可矩阵。
2. GitHub/Hugging Face 使用现有公开仓库地址；V2-1 历史由提交记录保留。
3. legacy/wild/symbolic 人工审核由项目所有者确认完成，并绑定 frozen labels SHA256。
4. 未能证明逐样本许可的训练原图和 JSONL 已从公共源码候选隔离。
5. V3 首次公开源码提交为 `a68b434f2a905562929c545470192b4b11f1c66c`；HF 模型 revision 为 `e496110ec222c1a70ebca287990c07dae47a2daa`，远端权重 SHA256 与本地 final export 一致。

剩余限制是 private photo 为 0、第二台机器 clean-download/GPU smoke 与容器 digest 未完成；Demo 录屏按本轮范围取消。这些限制不应被写成已完成证据。发布使用现有 GitHub 历史的干净 clone，V2-1 父历史得到保留。

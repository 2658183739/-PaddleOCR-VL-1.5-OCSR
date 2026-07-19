# V3 公共发布验收记录

验收时间：2026-07-19 12:04 +08:00。

## 已完成

- GitHub：[`a68b434f2a905562929c545470192b4b11f1c66c`](https://github.com/2658183739/-PaddleOCR-VL-1.5-OCSR/commit/a68b434f2a905562929c545470192b4b11f1c66c)，`main`，保留父提交 `82ff294` 的 V2-1 历史。
- Hugging Face：[`e496110ec222c1a70ebca287990c07dae47a2daa`](https://huggingface.co/L2658183739/PaddleOCR-VL-1.5-OCSR/commit/e496110ec222c1a70ebca287990c07dae47a2daa)，`main`，页面显示 Apache-2.0 和 V3 模型卡。
- HF 远端 Xet 权重 SHA256：`2a7ac278677ff56379e67933d6d81481991b755b93355fca5902cc36a7b1cc13`，与本地 final export 一致。
- GitHub 页面显示 Apache-2.0、V3 代码/证据、人审声明、许可矩阵和最终 HTML/PPT；模型权重未进入 GitHub 普通 Git history。
- Hugging Face 页面显示 V3 的 `checkpoint-1400`、beam4/return4、wild strict `22.92%`、valid `84.72%`，旧 V2-1 卡片正文已替换。

## 材料哈希

| 文件 | SHA256 |
| --- | --- |
| `PaddleOCR_VL_OCSR_V3_scientific_final.html` | `bbc537f96ed8f3bf4bcf1687c2fe60175b6f627b6661aed11d1dd5060741e7d2` |
| `PaddleOCR_VL_OCSR_V3_scientific_final.pptx` | `9818447023d41d2eeda81725259f7e2abf43637bb693730d603e21d5de9df795` |
| `PaddleOCR_VL_OCSR_V3_scientific_final_montage.png` | `df88a9cae7aa87ec6216ae8237eaba980c22a2831fc83ff49345560cba9b5d37` |
| HF `SHA256SUMS` | `78fba1d3feb537e020c4e39efea8d91c2f9505003f30b157b5c376f374b7e8d8` |

## 未声称完成

- 尚未从第二台 A100/H800 做 clean-download、环境重建和固定 20 条 beam smoke。
- 尚未完成容器 digest 级复现、private-photo 面板和至少四 seed 的 confirmatory 复验。
- Demo 录屏按本轮范围取消。

因此，本记录证明“公开页面与发布对象一致”，不等同于“独立机器从零复现已经完成”。

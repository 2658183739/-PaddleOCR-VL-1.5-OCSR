# 邮件提交建议模板

收件人：

```text
ext_paddle_oss@baidu.com
paddleocr@baidu.com
cuicheng01@baidu.com
liujiaxuan01@baidu.com
```

## 1. 评估集提交

邮件标题：

```text
PaddleOCR衍生模型挑战赛-评估集提交-【GitHub ID】
```

正文建议包含：

1. 评估集名称
2. 数据总量
3. 子基准结构
4. 下载链接
5. 评测脚本位置
6. 数据说明文档位置

建议当前版本写法：

- 主 OCSR 基准：`canonical_smiles_main_v1`
- 真实世界增强基准：`ocsr_realworld_mixed_eval_v1p1`
- 说明：主分数建议优先基于 `canonical_smiles_main_v1` 报告，`mixed_v1p1` 作为真实世界补充基准解释

## 2. 训练数据构建报告

邮件标题：

```text
PaddleOCR衍生模型挑战赛-训练数据构建报告-【GitHub ID】
```

正文建议包含：

1. 报告文件链接
2. 数据来源概览
3. 标注规范
4. 质量控制方式
5. 关键脚本位置

## 3. 完整开源项目

邮件标题：

```text
PaddleOCR衍生模型挑战赛-完整开源项目-【GitHub ID】
```

正文建议包含：

1. GitHub 仓库链接
2. 说明训练数据未完全开源的边界
3. 训练、导出、评测、demo 入口

## 4. Hugging Face 模型

邮件标题可附在完整开源项目邮件正文中，也可以单独说明：

```text
PaddleOCR衍生模型挑战赛-Hugging Face模型-【GitHub ID】
```

正文建议包含：

1. 模型链接
2. 基座模型
3. 微调方式
4. 当前评测结果摘要
5. 已知局限

## 5. 当前最适合第一次试水的提交组合

建议先发：

1. 评估集提交
2. 训练数据构建报告
3. GitHub 开源项目
4. Hugging Face 模型

其中评估集最推荐先发：

- `canonical_smiles_main_v1`
- `ocsr_realworld_mixed_eval_v1p1`

而不是一上来把整个 `V2-1/data/eval/` 作为一个大 collection 混合提交。

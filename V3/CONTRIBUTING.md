# Contributing to PaddleOCR-VL OCSR V3

本项目欢迎对数据构建、OCSR 微调、生成评测、质量控制和复现文档的贡献。所有结果必须可回溯，不能用未记录的数据或人工操作替代证据。

## 开发环境

GPU 环境使用 `setup_h800_environment.sh`；Windows 本地的数据构建和测试使用项目内 RDKit 环境：

```powershell
& '.\.conda_rdkit\python.exe' V3\scripts\build_v3_datasets.py --project-root .
& '.\.conda_rdkit\python.exe' V3\scripts\verify_v3_workspace.py --project-root .
& '.\.conda_rdkit\python.exe' -m unittest discover -s V3\tests -v
```

提交代码前至少运行与改动范围相关的单元测试、`py_compile` 和小样本 smoke。修改训练、导出或推理路径时，还要验证 merged model 可以独立加载。

## 数据贡献要求

新增记录必须说明：

- `source`、`source_url_or_doc` 和 `license`；
- 图片 ID、`structure_id` 或等价的稳定上游 ID；
- 单分子 canonical SMILES 标签及其取得/清洗方式；
- `task_type`、`difficulty`、采集或变换来源；
- 若为实拍，记录设备、角度、光照、采集人、时间和授权；
- 若为算法增强，记录 `transform_parent` 和参数，不得写成真实采集。

缺少许可或来源证据的样本不得进入公共数据 release。含盐、多片段、dummy atom、反应式、R-group、chemfig 或 `ssml_normed` 的记录不得混入 canonical SMILES 主任务；需要保留时应进入独立 track。

## 划分和评测要求

- 普通 OCSR 按 canonical molecule/`structure_id` 隔离 train、development 和 test。
- 论文图按 `paper_group` 分组，同一论文不能跨 train/test。
- 同一结构的多视角照片必须进入同一 split。
- legacy core/region 只用于 development。
- locked test 只能在模型、prompt、checkpoint 和生成策略冻结后运行一次。
- 看到 locked test 结果后不得返回搜索超参；后续实验必须标记为 exploratory，并建立新的 confirmatory test。

比较模型时报告 RDKit canonical exact、valid SMILES 和必要的分层指标。重复分子、论文或多视角样本使用 cluster bootstrap；不要把相关图片当成独立重复扩大 N。

## 实验记录

每个训练或生成实验至少保存：

- 配置文件、基座模型 hash、训练数据 hash；
- seed、运行顺序、GPU/驱动/框架版本；
- stdout/stderr 日志和训练结果；
- checkpoint/export hash；
- 评测 manifest、预测、逐样本 details 和汇总报告；
- 失败实验及停止原因。

消融实验应尽量一次只改变一个可解释因素。多个数据因素同时变化时，使用完整 factorial 或明确承认混杂；只有两个 seed 时不得声称统计显著。

## 文档和术语

请区分：

- `records`：包含 repeat/cap 后的训练记录数；
- `unique_image_refs`：去重图片引用数；
- `unique_canonical_smiles`：去重 canonical 分子数；
- `structure_id`：自采/同结构多视角的聚类单位；
- `paper_group`：论文来源聚类单位；
- `oracle`：候选池上限，不是可提交模型分数。

所有“人工审核完成”“自行采集”“已发布”都必须有与相应声明匹配的可审计证据：人工审核至少提供项目所有者声明、范围、结果边界和冻结清单 hash；自行采集提供设备/授权/分组记录；发布提供公开 URL 与 commit/revision。不得虚构姓名、签名或逐样本决定，未完成内容应写成 limitation。

## Pull Request 检查

提交前确认：

1. 改动范围清楚，不包含模型权重、数据包或无关生成文件。
2. 测试通过，新增行为有相应测试或可复现 smoke 记录。
3. 没有 SSH 密码、token、私钥、个人路径或未授权数据。
4. README、数据卡、模型卡和证据文件的数字一致。
5. 未根据 locked test 结果修改模型或策略。

项目级许可证尚需项目所有者最终确定；贡献内容在许可证确定前不得被视为已经获得公共再分发授权。

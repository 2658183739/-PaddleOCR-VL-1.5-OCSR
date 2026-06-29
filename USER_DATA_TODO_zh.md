# 你需要补的数据清单

我已经把可训练的数据入口、图片规范化、manifest 校验、评测防泄漏、SFT 转换和 V2-2 合并脚本都搭好了。你后面主要负责拿到图片和对应 SMILES。

## 你要放数据的位置

```text
V2-1/data/incoming/weak_domain/
```

推荐按下面放原图：

```text
raw_images/private_handdrawn/
raw_images/real_world_photo_scan/
raw_images/edu_exam/
raw_images/patent_document/
```

## 图片要求

可以给 `.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff`，脚本会统一转成 RGB PNG。

最好满足：

- 一张图一个目标分子。
- 有一点白边，不要贴边裁切。
- 分子结构线条清晰可见。
- 可以有拍照阴影、扫描噪声、中文题干、选项字母、页码或标题，但目标分子必须明确。
- 图片不要太小，最短边最好大于 96 px。
- 图片不要巨大整页，除非目标分子非常明确；否则先 crop 到目标分子附近。

不要放：

- 没有明确目标分子的整页大图。
- 一张图多个候选分子但不知道识别哪一个。
- 只有文字/公式/LaTeX/chemfig，没有结构图。
- 标签不是 SMILES 的样本。
- 从评测集复制出来的图片或同一个分子的训练样本。

## SMILES 要求

必须是普通 SMILES/canonical SMILES，例如：

```text
CCO
COc1cc(N)ncn1
O=C(O)c1ccc(B(O)O)s1
```

不要填：

```text
\chemfig{...}
LaTeX
SSML
分子名称
中文描述
模型猜出来但没人核对的结果
```

如果你是自己生成/分配的手绘任务，标签用“分配给绘图者的原始 SMILES”，不要从图片反推。

## CSV 怎么填

编辑这个文件：

```text
V2-1/data/incoming/weak_domain/weak_domain_training_candidates.csv
```

字段示例：

```csv
id,image,smiles,source,difficulty,weak_domain,license,source_url_or_doc,collector,notes
private_handdrawn_000001,raw_images/private_handdrawn/private_handdrawn_000001.jpg,CCO,private_handdrawn,handwritten,decimer_handdrawn,private_internal,collection_batch_202605,张三,
```

`weak_domain` 推荐填：

```text
decimer_handdrawn
real_world_photo_scan
edu_exam
document_page_context
long_or_stereo
```

`difficulty` 推荐填：

```text
handwritten
photo
scan
degraded_scan
chinese_exam
document_embed
page_level
multi_grid
hard
```

## 你优先采什么

第一优先级：私人手绘

- 500-1500 张。
- 从 PubChem/ChEMBL/现有非评测 SMILES 里选分子。
- 让多人手画，手机拍照或扫描。
- 每张图对应已知 SMILES。

第二优先级：真实拍照/扫描

- 2000-4000 张。
- 把分子渲染到纸上或屏幕上，再手机拍。
- 做不同角度、阴影、模糊、低清、裁切、纸张背景。

第三优先级：中文考试/教学风格

- 1000-2500 张。
- 自己生成题干、选项、中文说明，目标分子 crop 清楚。
- 标签仍然是 SMILES。

第四优先级：公开 DECIMER / patent document

- 下载公开数据后按 manifest 填入。
- 许可不确定的数据不要公开，只作为私有训练材料。

## 放好后怎么处理

从项目根目录运行：

```bash
python V2-1/scripts/prepare_weak_domain_manifest.py \
  --input V2-1/data/incoming/weak_domain/weak_domain_training_candidates.csv \
  --output V2-1/data/manifests/weak_domain_training_candidates.jsonl \
  --image-output-root V2-1/data/incoming/weak_domain/normalized_images
```

再转成 SFT：

```bash
python V2-1/scripts/import_weak_domain_training_pool.py \
  --project-root . \
  --manifest V2-1/data/manifests/weak_domain_training_candidates.jsonl \
  --output V2-1/data/sft_materialized/train_weak_domain_pool_messages.jsonl \
  --assets-root V2-1/data/assets/weak_domain_pool
```

最后合并成 V2-2 训练集：

```bash
python V2-1/scripts/build_singleline_rw_v2_dataset.py --project-root .
```

输出：

```text
V2-1/data/sft_materialized/train_singleline_rw_v2_messages.jsonl
```

这个文件就是下一轮训练可以吃的 JSONL。

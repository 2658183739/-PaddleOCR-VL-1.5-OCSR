# 自采照片与手绘补充协议

## 规模分级

两天内的最低可信交付目标是 80 个不同 `structure_id`，每个结构 4 个视觉条件，共 320 张图。它主要用于补足真实退化类型和官方反馈，不等于已经达到官方评估集最高档。

官方高分倾向是至少 1000 个真实评测实例。当前 locked wild 有 301 张，因此若要靠 4 视角自采图补到 1000，至少还需保留 699 张自采 eval 图，即至少 175 个 eval-only 结构；考虑 10%-15% 质检淘汰，实际应准备约 195-200 个 eval 结构。若仍使用默认 30% eval 哈希切分，80 个结构只会贡献约 96 张评测图，总 locked 规模约 397，明显达不到最高档。

因此推荐把采集批次预先冻结为两类：

- `split=eval`：独立结构，只进入 locked private test；
- `split=train`：另一批独立结构，只用于训练增强。

CSV 未填写 `split` 时，导入器才按 `structure_id` 固定哈希切分约 70% train、30% eval。不能先看模型结果再改 split。

建议结构构成：标准 printed 30、长分子/密集环 15、stereo 15、缩写但可还原为标准分子 10、手绘 10。所有标签必须是单一、完整、RDKit 可解析的 canonical SMILES。

## 四种拍摄条件

| 条件码 | 要求 |
| --- | --- |
| `front_diffuse` | 正面 0-8 度、均匀室内光、目标完整 |
| `oblique_20_35` | 水平或俯仰 20-35 度，保留透视畸变 |
| `low_light_noise` | 较暗环境或较高 ISO，允许轻度噪声/模糊 |
| `glare_shadow_crop` | 局部反光或阴影，并有轻微边距裁切但不能切断结构 |

至少使用 2 台不同设备。打印纸、教材页、白板/手写纸应分别记录；从显示器翻拍可以作为一个 case，但不能把全部“实拍”都做成屏幕翻拍。

## 采集与标注步骤

1. 先冻结结构表：`structure_id + canonical_smiles + 渲染/手绘来源`。
2. 拍摄时记录设备、角度、光照、条件码、采集人和时间。
3. Reviewer 1 检查图像清晰度、单目标和标签对应。
4. Reviewer 2 独立复核标签与预先冻结的 `split`；争议样本不直接保留。
5. 两人均 pass 后填入 `qc/private_photo_collection.csv`，同一 `structure_id` 的所有视角必须使用相同 split。
6. 运行导入脚本；任何重复 ID、缺图、无同意记录、非法 SMILES、同结构多标签或 canonical train/eval 重叠都会失败。

```bash
python V3/scripts/import_private_photo_data.py \
  --csv V3/qc/private_photo_collection.csv \
  --project-root . \
  --eval-fraction 0.30
```

## 禁止项

- 不用水平/垂直翻转，因为会改变立体键含义。
- 不接受键线被裁断、原子标注不可读或目标区域不明确的图。
- 不把算法生成的亮度/旋转增强写成“自行实拍”。
- 不让同一结构的一张照片进训练、另一张照片进评测。
- 不补写不存在的审查人员或拍摄设备。

# V2-1 后续提分路线：先救输入，再谈训练

日期：2026-06-19

## 当前判断

V2-1 原模型在 UOB 这类干净结构图上已经能打，`chem_light` 候选重排还能把 UOB80 从 0.75 拉到 0.80。真正拖分的是 real_world/chinese_exam。

前面已经验证过两件事：

- 整页输入、高分辨率、TTA、beam 都没有把 realworld20 救起来。
- realworld20 的 candidate oracle 是 0，说明正确答案没进候选池。

所以现在不该直接 DPO。DPO 解决的是“候选里有正确答案但排错了”，不是“模型连目标分子都没看清”。当前第一步是把输入图裁对。

## 新增文件

- `V2-1/scripts/build_realworld_region_crop_probe.py`
- `V2-1/run_4090_realworld_region_crop_probe_v1.sh`

## 区域裁剪策略

`extra_chinese_exam_xxx` 图不是单分子图，而是一整页题面。标签对应左上第 1 题，原图里还包含右侧第 2 题、页眉、横线、答题线。模型直接看整页时，很容易抽错目标。

当前裁剪版本：

- `exam_q1_panel`：当前最好。
- `exam_q1_trim`：候选更多，但 selected 不如 panel。
- `exam_q1_trim_gray`：不如 panel。
- 相对裁剪框：`(0.13, 0.09, 0.385, 0.235)`
- `panel` 保留第 1 题结构图的小面板，不再把右边第 2 题、页眉和答题线喂给模型。
- `trim` 会进一步收紧到分子本体，预览图约 293x257。

## 远端运行

远端目录：

```bash
cd /root/autodl-tmp/data/platform_migration_bundle_20260531
```

启动单版本：

```bash
OUT=/root/autodl-fs/outputs_v2/realworld_region_crop_probe_v1
mkdir -p "$OUT"
VARIANTS=exam_q1_panel nohup bash V2-1/run_4090_realworld_region_crop_probe_v1.sh > "$OUT/nohup.log" 2>&1 &
```

查看进度：

```bash
OUT=/root/autodl-fs/outputs_v2/realworld_region_crop_probe_v1
wc -l "$OUT/exam_q1_trim/pred.jsonl"
tail -80 "$OUT/nohup.log"
nvidia-smi
```

跑完看汇总：

```bash
cat /root/autodl-fs/outputs_v2/realworld_region_crop_probe_v1/summary.json
```

## 已跑结果

realworld20：

| 输入/后处理 | canonical exact | oracle | mean Tanimoto | 备注 |
| --- | ---: | ---: | ---: | --- |
| 整页 highpix no-TTA | 0.00 | 0.00 | 0.10 | 之前的失败基线 |
| 普通裁白边 crop | 0.00 | 0.00 | 0.10 | 没解决多题干扰 |
| `exam_q1_trim` + `realworld_soft` | 0.20 | 0.25 | 0.503 | 从 0 救起来了 |
| `exam_q1_panel` + `chem_light` | 0.25 | 0.30 | 0.582 | 当前最好 |
| `exam_q1_trim_gray` + `realworld_soft` | 0.15 | 0.25 | 0.487 | 灰度增强不如原图 |
| `exam_q1_panel` beam8 + `realworld_soft` | 0.25 | 0.30 | 0.541 | 慢，没涨分 |

mixed60 投影面板：

| 方案 | canonical exact | UOB | USpto | real_world |
| --- | ---: | ---: | ---: | ---: |
| 原始 V2-1 + `chem_light` | 0.4333 | 0.8500 | 0.4500 | 0.0000 |
| UOB/USpto 沿用原结果，real_world 换 `exam_q1_panel` | 0.5167 | 0.8500 | 0.4500 | 0.2500 |

这一步是现在最实在的提分：不改权重，只改 real_world 的输入区域和候选选择，总面板从 43.33% 到 51.67%。

## 下一步怎么接 SFT

区域裁剪已经有效，但离 80%-90% 还差得远。短 SFT 可以做，但要按下面的顺序来：

1. 用裁剪后的 real_world/chinese_exam 图生成一批训练样本。
2. 从原始 V2-1 `V2-1/outputs/export` 接着训，不从 `fast90` 接。
3. 先跑 20-80 step smoke。
4. 固定评测顺序：realworld20 -> mixed60 -> UOB80。
5. 任何一步明显伤 UOB，就回滚。

注意：不要直接把评测集当训练集刷分。真正有价值的是按这个页面模板生成或收集同风格新样本，再用同一套裁剪脚本处理。

## DPO 什么时候上

DPO 不是现在的第一刀。满足这些条件再上：

- candidate oracle 比 selected exact 高出一截。
- preference pairs 至少几百条。
- preference pairs 里要有 real_world 或 USpto 的正例，不能只有 UOB。
- PaddleFormers 对当前 PaddleOCR-VL 训练入口确认支持 DPO，或者另写稳定的 DPO 入口。

现在 real_world 已经能产生少量 preference pairs，但数量还是太少。当前最多只有个位数，不足以开正式 DPO。先继续扩候选和扩数据，再谈 DPO。

## 当前主线

短期主线：

1. `V2-1/outputs/export`
2. `exam_q1_panel` 区域裁剪
3. `--save-candidates`
4. `chem_light` rerank
5. mixed60 里只替换 real_world 部分，UOB/USpto 沿用原始 V2-1 + `chem_light`
6. 新收集/合成同风格数据后，再做短 SFT
7. preference pairs 足够多后，再做 DPO

beam8 已经试过，没比 beam4 更好。继续堆 beam、TTA 或弱域全量评测，性价比不高。

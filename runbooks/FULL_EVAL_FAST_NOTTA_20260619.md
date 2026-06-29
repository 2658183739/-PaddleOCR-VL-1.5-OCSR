# 全量评测快版路线记录

日期：2026-06-19

## 当前采用的路线

主模型不换，仍然使用：

```text
V2-1/outputs/export
```

评测拆成两段：

1. `chinese_exam`：先裁左上第 1 题 panel，再 no-TTA 推理。
2. `general`：原图 no-TTA 推理，保留 5 个 prompt、beam4、return4。

输出目录：

```text
/root/autodl-fs/outputs_v2/full_eval_region_panel_v1_fast_notta/ocsr_realworld_mixed_eval_v1p1
```

## 为什么停掉 light-TTA 全量

`light TTA` 会跑 4 个图像版本：`orig`、`gray_auto`、`high_contrast`、`sharp_contrast`。这基本等于 4 倍推理时间。

离线用旧候选做过消融：

| 面板 | light + rerank | orig-only + rerank | 差异 |
| --- | ---: | ---: | ---: |
| mixed60 | 0.4333 | 0.4167 | 少 1/60 |
| UOB80 | 0.8000 | 0.7875 | 少 1/80 |

这个损失比 4 倍时间成本小，所以主线改为 no-TTA。被停掉的慢速目录没有删除：

```text
/root/autodl-fs/outputs_v2/full_eval_region_panel_v1/ocsr_realworld_mixed_eval_v1p1
```

里面保留了 `general` 的 14 条 partial，作为记录。

## 为什么不减少 prompt

用旧候选只看 `orig`，再按 prompt 子集离线评估：

| 面板 | p0 rerank | p2 rerank | p012 rerank | p01234 rerank |
| --- | ---: | ---: | ---: | ---: |
| mixed60 | 0.3667 | 0.3667 | 0.4000 | 0.4167 |
| UOB80 | 0.6875 | 0.7125 | 0.7750 | 0.7875 |

结论：TTA 可以砍，prompt 暂时不能砍。5 个 prompt 仍然保留。

## 当前监控命令

```bash
cd /root/autodl-tmp/data/platform_migration_bundle_20260531
OUT=/root/autodl-fs/outputs_v2/full_eval_region_panel_v1_fast_notta/ocsr_realworld_mixed_eval_v1p1

ps -eo pid,ppid,etime,pcpu,pmem,args | grep -E 'run_4090_eval_full_region_panel_v1|infer_ocsr_transformers' | grep -v grep
nvidia-smi
for f in "$OUT"/chinese_exam_panel/parts/*.jsonl "$OUT"/general/parts/*.jsonl; do [ -e "$f" ] && echo "$(wc -l < "$f") $f"; done
tail -80 "$OUT/nohup.log"
```

## 断点续跑

脚本会先合并已有 part，再生成 remaining：

```text
V2-1/scripts/eval_jsonl_resume.py
```

如果中断，直接重新运行同一个命令即可：

```bash
cd /root/autodl-tmp/data/platform_migration_bundle_20260531
OUT=/root/autodl-fs/outputs_v2/full_eval_region_panel_v1_fast_notta/ocsr_realworld_mixed_eval_v1p1
nohup env OUT_ROOT="$OUT" GENERAL_TTA=none bash V2-1/run_4090_eval_full_region_panel_v1.sh > "$OUT/nohup.log" 2>&1 &
```

## 完成后看结果

```bash
OUT=/root/autodl-fs/outputs_v2/full_eval_region_panel_v1_fast_notta/ocsr_realworld_mixed_eval_v1p1
cat "$OUT/summary.json"
cat "$OUT/merged/rerank_chem_light_eval_report.json"
```


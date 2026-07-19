# V3 最小复现指南

## 1. 磁盘布局

在 AutoDL/H800 上把代码、模型和高频读写数据放到 `/root/autodl-tmp/V3`，把日志、最终模型包和完成标记放到 `/root/autodl-fs/V3_results`。不要直接在文件存储上训练，避免低速随机 I/O。

## 2. 环境

```bash
cd /root/autodl-tmp
bash V3/setup_h800_environment.sh
python -m unittest discover -s V3/tests -v
python V3/scripts/verify_v3_workspace.py --project-root .
```

本次实际环境为单卡 NVIDIA H800 PCIe 80GB、Python 3.10、Paddle 3.3.1、PaddleFormers commit `e51f911c23b41283ef6c62f8aa4a7e99291bcd11`、Transformers 4.55.4 和 RDKit 2025.9.6。最终包中的 `evidence/runtime/` 保存真实 `pip freeze` 与 `nvidia-smi`。

## 3. 一键流水线

```bash
screen -dmS v3pipeline bash -lc '
  set -o pipefail
  cd /root/autodl-tmp
  exec env V3_OMP_NUM_THREADS=1 \
    V3_BACKUP_ROOT=/root/autodl-fs/V3_results \
    bash V3/run_h800_pipeline.sh \
    >> /root/autodl-fs/V3_results/logs/pipeline_master.log 2>&1
'
```

流水线按完成标记跳过已完成阶段。不要删除 checkpoint 后重跑；中断时先检查 `train_results.json`、`saved_signal_0` 和对应 evidence JSON。

## 4. 阶段说明

1. `run_h800_probes.sh`：8 个 factorial probe、warm-start、augmentation dose。
2. `run_h800_probe_eval.sh`：导出每个 LoRA 并在两个 development 面板生成评测。
3. `run_h800_final.sh`：1400-step final、7 个 checkpoint 评测、最佳 checkpoint、hard replay。
4. `run_h800_generation_ablation.sh`：greedy 与 beam4/return4。
5. `run_h800_locked_and_package.sh`：冻结 hash、一次性 locked test、报告和最终包。

## 5. 结果定位

- 最终模型：`V3/models/final_best_export/`
- 自动结果总表：`V3/evidence/FINAL_RESULTS_zh.md`
- 消融原始 JSON：`V3/evidence/probe_analysis.json`
- final checkpoint 选择：`V3/evidence/final_checkpoint_selection.json`
- hard replay：`V3/evidence/final_vs_hard_replay.json`
- 生成策略：`V3/evidence/generation_policy_selection.json`
- locked test：`V3/eval_runs_locked/<timestamp>/`
- 可恢复 checkpoint：`V3/evidence/training_artifacts/resume/`

## 6. 复现边界

完整训练复现需要有许可的图像资产。最终模型包只带约 100MB 的文本 manifest/annotation，不重复携带 14GB 图像。没有原始图像时仍可加载最终 merged 模型并对自有图片推理，但不能重建训练集或重复训练。

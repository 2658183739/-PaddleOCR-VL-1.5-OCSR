#!/usr/bin/env python3
"""Local, image-first review UI for V3 evaluation labels.

Run one process per reviewer so independent decisions are written to separate
CSV columns. The tool never changes labels or predictions; it only records the
human review decision and reason code.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path


DECISIONS = ("pending", "pass", "reject", "uncertain")
REASONS = (
    "ok",
    "label_mismatch",
    "multi_target",
    "bad_crop",
    "unreadable",
    "leakage",
    "source_unknown",
    "other",
)


class ReviewStore:
    def __init__(self, csv_path: Path, project_root: Path, reviewer: str):
        if reviewer not in {"1", "2", "adjudicator"}:
            raise ValueError("reviewer must be 1, 2, or adjudicator")
        self.csv_path = csv_path
        self.project_root = project_root
        self.reviewer = reviewer
        self.lock = threading.Lock()
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            self.fieldnames = list(reader.fieldnames or [])
            self.rows = list(reader)
        if "final_decision_reason" not in self.fieldnames:
            self.fieldnames.append("final_decision_reason")
        required = {
            "panel", "sample_id", "image", "label", "automated_status",
            "reviewer_1", "reviewer_1_decision", "reviewer_1_reason",
            "reviewer_2", "reviewer_2_decision", "reviewer_2_reason",
            "final_decision", "review_time",
        }
        missing = sorted(required.difference(self.fieldnames))
        if missing:
            raise ValueError(f"review CSV missing columns: {', '.join(missing)}")
        self._image_index: dict[str, Path] | None = None

    def _build_image_index(self) -> dict[str, Path]:
        if self._image_index is not None:
            return self._image_index
        roots = [self.project_root, self.project_root / "V3", self.project_root / "V3" / "data"]
        index: dict[str, Path] = {}
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file():
                    index.setdefault(path.name, path)
        self._image_index = index
        return index

    def image_path(self, row: dict[str, str]) -> str | None:
        raw = (row.get("image") or "").strip().replace("/", os.sep)
        panel_roots = {
            "core_767": self.project_root
            / "V3"
            / "data"
            / "eval"
            / "canonical_smiles_main_v1",
            "wild_strict_v3": self.project_root,
        }
        direct = [
            self.project_root / raw,
            self.project_root / "V3" / raw,
            self.project_root / "V3" / "data" / raw,
            panel_roots.get(row.get("panel", ""), self.project_root) / raw,
        ]
        for path in direct:
            if path.is_file():
                return str(path)
        name = Path(raw).name
        found = self._build_image_index().get(name)
        return str(found) if found else None

    def render(self, index: int):
        index = max(0, min(int(index), len(self.rows) - 1))
        row = self.rows[index]
        image = self.image_path(row)
        reviewer_decision = self._decision(row)
        reviewer_reason = self._reason(row)
        info = (
            f"{index + 1}/{len(self.rows)} | panel={row.get('panel', '')} | "
            f"sample_id={row.get('sample_id', '')} | source={row.get('source', '')} | "
            f"difficulty={row.get('difficulty', '')}\n"
            f"automated_status={row.get('automated_status', '')} | "
            f"image={'found' if image else 'MISSING'}"
        )
        progress = f"{index + 1}/{len(self.rows)}"
        return image, info, row.get("label", ""), reviewer_decision, reviewer_reason, progress

    def _decision(self, row: dict[str, str]) -> str:
        if self.reviewer == "adjudicator":
            return row.get("final_decision", "pending") or "pending"
        return row.get(f"reviewer_{self.reviewer}_decision", "pending") or "pending"

    def _reason(self, row: dict[str, str]) -> str:
        if self.reviewer == "adjudicator":
            return row.get("final_decision_reason", "") or ""
        return row.get(f"reviewer_{self.reviewer}_reason", "") or ""

    def save(self, index: int, decision: str, reason: str, reviewer_id: str) -> None:
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {DECISIONS}")
        if reason and reason not in REASONS:
            raise ValueError(f"reason must be one of {REASONS}")
        reviewer_id = reviewer_id.strip()
        if not reviewer_id:
            raise ValueError("reviewer id is required")
        with self.lock:
            row = self.rows[int(index)]
            now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            if self.reviewer == "adjudicator":
                row["final_decision"] = decision
                row["final_decision_reason"] = reason
            else:
                row[f"reviewer_{self.reviewer}"] = reviewer_id
                row[f"reviewer_{self.reviewer}_decision"] = decision
                row[f"reviewer_{self.reviewer}_reason"] = reason
            row["review_time"] = now
            self._atomic_write()

    def _atomic_write(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.csv_path.name}.", suffix=".tmp", dir=self.csv_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.csv_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)


def build_app(store: ReviewStore, reviewer_id: str):
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Install gradio to run the review UI: pip install gradio") from exc

    def render(index):
        return store.render(int(index))

    def move(index, delta):
        return max(0, min(int(index) + int(delta), len(store.rows) - 1))

    def save(index, decision, reason):
        store.save(int(index), decision, reason, reviewer_id)
        return store.render(int(index))

    with gr.Blocks(title=f"V3 QC review - {store.reviewer}") as app:
        gr.Markdown(
            "# V3 评测集逐图质检\n"
            "只记录人工决定，不修改标签。请独立审查后再保存。"
        )
        index = gr.State(0)
        with gr.Row():
            image = gr.Image(type="filepath", label="结构图", height=520, interactive=False)
            with gr.Column():
                info = gr.Textbox(label="样本信息", lines=5, interactive=False)
                label = gr.Textbox(label="预先冻结标签", lines=4, interactive=False)
                decision = gr.Dropdown(DECISIONS, value="pending", label="决定")
                reason = gr.Dropdown(REASONS, value="", allow_custom_value=True, label="原因码")
                reviewer = gr.Textbox(value=reviewer_id, label="审核人员编号", interactive=False)
                save_button = gr.Button("保存当前决定", variant="primary")
        with gr.Row():
            previous = gr.Button("上一条")
            next_button = gr.Button("下一条")
            progress = gr.Textbox(label="进度", interactive=False, scale=1)

        outputs = [image, info, label, decision, reason, progress]
        app.load(render, inputs=index, outputs=outputs)
        previous.click(lambda i: move(i, -1), inputs=index, outputs=index).then(
            render, inputs=index, outputs=outputs
        )
        next_button.click(lambda i: move(i, 1), inputs=index, outputs=index).then(
            render, inputs=index, outputs=outputs
        )
        save_button.click(save, inputs=[index, decision, reason], outputs=outputs)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("V3/qc/eval_manual_review.csv"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--reviewer", choices=("1", "2", "adjudicator"), required=True)
    parser.add_argument("--reviewer-id", required=True, help="Human reviewer ID; never leave this blank")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7861")))
    args = parser.parse_args()
    csv_path = args.csv if args.csv.is_absolute() else args.project_root / args.csv
    project_root = args.project_root.resolve()
    store = ReviewStore(csv_path.resolve(), project_root, args.reviewer)
    build_app(store, args.reviewer_id).queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, show_error=True
    )


if __name__ == "__main__":
    main()

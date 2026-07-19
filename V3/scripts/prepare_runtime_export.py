import argparse
from pathlib import Path

from eval_latest_checkpoints import sync_runtime_metadata


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--base-model-dir", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    runtime_dir = Path(args.runtime_dir).resolve()
    base_model_dir = Path(args.base_model_dir).resolve()

    if not (runtime_dir / "config.json").is_file():
        raise FileNotFoundError(f"Runtime export has no config.json: {runtime_dir}")
    if not any(runtime_dir.glob("model*.safetensors")):
        raise FileNotFoundError(f"Runtime export has no model safetensors: {runtime_dir}")

    copied = sync_runtime_metadata(runtime_dir=runtime_dir, base_model_dir=base_model_dir)
    names = ", ".join(path.name for path in copied) if copied else "none"
    print(f"[INFO] runtime export ready: {runtime_dir}")
    print(f"[INFO] copied missing metadata: {names}")


if __name__ == "__main__":
    main()

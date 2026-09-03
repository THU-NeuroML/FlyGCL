#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flygcl.common.config import WORKSPACE, load_config, resolve_data_paths
from flygcl.common.data import write_json
from flygcl.anchor.trainer import AnchorTrainer, copy_resolved_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the FlyGCL TL prompt/router anchor")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--auto-resume", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.num_workers is not None:
        config.setdefault("runtime", {})["num_workers"] = int(args.num_workers)
    output_dir = Path(args.output_dir) if args.output_dir else WORKSPACE / config.get("output_root", "outputs/default")
    output_dir = output_dir.resolve()
    copy_resolved_config(config, output_dir)

    final_path = output_dir / "final_results.json"
    if args.auto_resume and final_path.is_file():
        try:
            previous = json.loads(final_path.read_text(encoding="utf-8"))
            if previous.get("status") == "complete":
                print(f"[skip] completed run: {output_dir}")
                return 0
        except (OSError, json.JSONDecodeError):
            pass

    trainer = AnchorTrainer(config, resolve_data_paths(config), output_dir)
    print(f"[data] task order: {trainer.manifest['task_order']}")
    if args.manifest_only:
        print(json.dumps(trainer.manifest, indent=2, ensure_ascii=False))
        return 0
    resume = args.resume
    if args.auto_resume and resume is None:
        for candidate in sorted(output_dir.glob("task_*/checkpoint.pt"), reverse=True):
            try:
                state = __import__("torch").load(candidate, map_location="cpu", weights_only=False)
                if "completed_task" in state and "model" in state:
                    resume = str(candidate)
                    print(f"[resume] latest valid checkpoint: {candidate}")
                    break
            except Exception as error:
                print(f"[resume] ignoring invalid checkpoint {candidate}: {error}")
    result = trainer.run(resume)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

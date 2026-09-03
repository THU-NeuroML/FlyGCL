#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
FLYGCL = PACKAGE / "flygcl"
CONFIGS = FLYGCL / "configs"


def run(*arguments: object) -> None:
    command = [sys.executable, *map(str, arguments)]
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, cwd=PACKAGE, check=True)


def reproduce_skill(args: argparse.Namespace) -> None:
    root = FLYGCL / "runs/skill"
    tl_root, ego_root, rn_root = root / "tl", root / "ego", root / "rn"
    for seed in args.seeds:
        tl = tl_root / f"seed_{seed}"
        ego = ego_root / f"seed_{seed}"
        rn = rn_root / f"seed_{seed}"
        common = ("--seed", seed, "--num-workers", args.num_workers)
        run(
            "-m", "flygcl.anchor.train", "--config", CONFIGS / "skill_anchor.yaml",
            "--output-dir", tl / "base", *common, "--auto-resume",
        )
        run(
            "-m", "flygcl.skill.train_tl", "--config", CONFIGS / "skill_tl.json",
            "--base-run", tl / "base", "--output-dir", tl / "expert",
            *common, "--device", args.device, "--auto-resume",
        )
        run(
            "-m", "flygcl.skill.evaluate_tl", "--expert-run", tl / "expert",
            "--base-run", tl / "base", "--output-dir", tl / "fusion", "--seed", seed,
        )
        run(
            "-m", "flygcl.skill.train_ego", "--config", CONFIGS / "skill_ego.json",
            "--output-dir", ego / "train", *common, "--device", args.device,
            "--batch-size", 96, "--auto-resume",
        )
        run(
            "-m", "flygcl.skill.evaluate_ego", "--run", ego / "train",
            "--config", CONFIGS / "skill_ego.json", "--output-dir", ego / "evaluation_fixed",
            "--seed", seed,
        )
        run(
            "-m", "flygcl.skill.train_rn",
            "--config", CONFIGS / "skill_ego.json",
            "--ego-checkpoint", ego / "train/checkpoint.pt", "--ego-run", ego / "train",
            "--output-dir", rn / "train", *common, "--device", args.device,
            "--batch-size", 64, "--epochs", 12, 14, 9, 6,
            "--learning-rate", 0.00012, "--relation-weight", 0.0,
            "--pair-correction-weight", 0.0, "--hard-focus-weight", 0.0,
            "--ema-decay", 0.985, "--auto-resume",
        )
        run(
            "-m", "flygcl.skill.evaluate_rn",
            "--ego-run", ego / "train", "--rn-run", rn / "train",
            "--tl-base", tl / "base", "--tl-expert", tl / "expert",
            "--config", CONFIGS / "skill_ego.json",
            "--output-dir", rn / "rn_evaluation_fixed", "--seed", seed,
        )
    run(
        "-m", "flygcl.skill.aggregate_tl", "--output-root", tl_root,
        "--seeds", *args.seeds,
    )
    run(
        "-m", "flygcl.skill.aggregate_ego", "--root", ego_root,
        "--output-dir", ego_root / "multiseed", "--seeds", *args.seeds,
    )
    run(
        "-m", "flygcl.skill.aggregate_rn", "--head", "rn",
        "--root", rn_root, "--output-dir", rn_root / "multiseed",
        "--seeds", *args.seeds,
    )


def reproduce_anticipation(args: argparse.Namespace) -> None:
    root = FLYGCL / "runs/anticipation"
    for setting in ("ego_exo", "ego_only", "exo_only"):
        setting_root = root / setting
        for seed in args.seeds:
            run(
                "-m", "flygcl.anticipation.train",
                "--config", CONFIGS / "anticipation.yaml",
                "--feature-root", Path(args.feature_root).resolve(),
                "--annotation-root", Path(args.annotation_root).resolve(),
                "--output-dir", setting_root / f"seed_{seed}",
                "--seed", seed, "--input-setting", setting,
                "--device", args.device, "--num-workers", args.num_workers,
            )
        run(
            "-m", "flygcl.anticipation.aggregate",
            "--input-root", setting_root, "--seeds", *args.seeds,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FlyGCL ego-exo experiments")
    subparsers = parser.add_subparsers(dest="benchmark", required=True)
    skill = subparsers.add_parser("skill", help="Run continual skill assessment")
    anticipation = subparsers.add_parser("anticipation", help="Run continual action anticipation")
    for child in (skill, anticipation):
        child.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
        child.add_argument("--device", default="cuda")
        child.add_argument("--num-workers", type=int, default=4)
    anticipation.add_argument("--feature-root", required=True)
    anticipation.add_argument("--annotation-root", required=True)
    args = parser.parse_args()
    if args.benchmark == "skill":
        reproduce_skill(args)
    else:
        reproduce_anticipation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

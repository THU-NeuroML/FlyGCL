#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flygcl.anticipation.data import (
    AnticipationDataset,
    assign_class_sessions,
    attach_sessions,
    audit_feature_coverage,
    audit_split_overlap,
    build_session_stream,
    choose_blurry_classes,
    decontaminate_training_split,
    load_valid_ids,
    read_official_csv,
)
from flygcl.anticipation.model import FlyGCLAnticipation


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_top5_recall(logits: torch.Tensor, labels: torch.Tensor) -> float:
    logits = logits.clone()
    labels = labels.clone()
    k = min(5, logits.shape[1])
    prediction = torch.zeros_like(labels)
    prediction.scatter_(1, logits.topk(k, dim=1).indices, 1.0)
    positives = labels.sum(0)
    true_positives = (prediction * labels).sum(0)
    valid = positives > 0
    return float((true_positives[valid] / positives[valid]).mean()) if valid.any() else 0.0


def limit_records(records, maximum: int, seed: int):
    if maximum <= 0 or len(records) <= maximum:
        return records
    generator = random.Random(seed)
    indices = list(range(len(records)))
    generator.shuffle(indices)
    return [records[index] for index in sorted(indices[:maximum])]


def evaluate_records(
    model,
    records,
    feature_root: Path,
    classes: int,
    config: Dict,
    device: torch.device,
    model_session: int,
    prior_correction: torch.Tensor,
    use_view_expert: bool,
) -> float:
    """Compute one macro Top-5 recall over a single, merged record pool."""
    if not records:
        return 0.0
    loader = DataLoader(
        AnticipationDataset(
            records,
            feature_root,
            classes,
            int(config["data"]["segments"]),
        ),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["runtime"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for batch in loader:
            result = model(
                batch["feature"].to(device),
                batch["view"].to(device),
                model_session,
                use_view_expert=use_view_expert,
            )
            logits = (
                result["logits"]
                + float(config["evaluation"]["prototype_weight"])
                * result["prototype_logits"]
                + prior_correction[None]
            )
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"])
    return macro_top5_recall(torch.cat(all_logits), torch.cat(all_labels))


def train_target(config: Dict, target: str, output: Path, feature_root: Path, annotation_root: Path):
    seed = int(config["seed"])
    valid_ids = load_valid_ids(annotation_root / f"valid_{target}s.txt")
    classes = len(valid_ids)
    views = ("ego", "exo")
    input_setting = str(config.get("training", {}).get("input_setting", "ego_exo"))
    setting_views = {
        "ego_exo": ("ego", "exo"),
        "ego_only": ("ego",),
        "exo_only": ("exo",),
    }
    if input_setting not in setting_views:
        raise ValueError(
            f"training.input_setting must be one of {sorted(setting_views)}, got {input_setting}"
        )
    train_views = setting_views[input_setting]
    splits = {}
    for split in ("train", "val", "test"):
        for view in views:
            path = annotation_root / f"{view}_{split}_valid.csv"
            splits[split, view] = read_official_csv(path, target, view, split, valid_ids)
    removed_train_duplicates = {}
    for view in views:
        splits["train", view], removed = decontaminate_training_split(
            splits["train", view], splits["val", view], splits["test", view]
        )
        removed_train_duplicates[view] = removed
    class_sessions = assign_class_sessions(
        splits["train", "ego"] + splits["train", "exo"], classes, 4
    )
    blurry_classes = choose_blurry_classes(
        classes, float(config["stream"]["r_D"]), seed
    )
    for key in list(splits):
        splits[key] = attach_sessions(splits[key], class_sessions)
    eval_split = str(config["evaluation"].get("split", "val"))
    if eval_split not in ("val", "test"):
        raise ValueError("evaluation.split must be val or test")
    coverage = audit_feature_coverage(
        feature_root,
        sum((splits["train", view] for view in train_views), [])
        + splits[eval_split, "ego"]
        + splits[eval_split, "exo"],
    )
    if coverage["features_missing"]:
        raise FileNotFoundError(
            f"Anticipation CLIP feature coverage is incomplete at {feature_root}: {coverage}. "
            "Set ANTICIPATION_FEATURE_ROOT to the official 5FPS CLIP feature directory."
        )

    device = torch.device(config["runtime"]["device"] if torch.cuda.is_available() else "cpu")
    sample_path = next(feature_root.glob("*.pt"))
    try:
        sample = torch.load(sample_path, map_location="cpu", weights_only=True)
    except TypeError:
        sample = torch.load(sample_path, map_location="cpu")
    if isinstance(sample, dict):
        sample = sample.get("features", sample.get("feature"))
    input_dim = int(torch.as_tensor(sample).shape[-1])
    model = FlyGCLAnticipation(
        input_dim, int(config["model"]["hidden_dim"]), classes, 4, float(config["model"]["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"])
    )
    replay_records: List = []
    teacher = None
    matrix = {view: [] for view in views}
    global_curves = {view: [] for view in views}
    counts = torch.zeros(classes, device=device)
    manifest = {
        "target": target,
        "input_setting": input_setting,
        "train_views": list(train_views),
        "evaluation_views": list(views),
        "classes": classes,
        "class_sessions": class_sessions,
        "feature_coverage": coverage,
        "source": "official balanced_full_annotation CSV (not leaked derived lists)",
        "decontamination": {
            "policy": "remove projected exact duplicates from train; preserve official val/test",
            "removed_from_train": removed_train_duplicates,
        },
        "split_overlap_audit": {
            view: audit_split_overlap(
                {
                    split: splits[split, view]
                    for split in ("train", "val", "test")
                }
            )
            for view in views
        },
        "protocol": {"sessions": 4, "r_D": config["stream"]["r_D"], "r_B": config["stream"]["r_B"]},
        "protocol_version": "flygcl_anticipation_v3_global_macro_top5",
        "reporting": {
            "primary": "global macro Top-5 over one merged pool of all seen-session records",
            "A_last": "last point of global_curve",
            "A_auc": "arithmetic mean of global_curve",
            "task_matrix": "diagnostic only; never averaged for primary metrics",
        },
        "disjoint_classes": sorted(set(range(classes)) - blurry_classes),
        "blurry_classes": sorted(blurry_classes),
    }
    (output / target).mkdir(parents=True, exist_ok=True)
    (output / target / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for view, audit in manifest["split_overlap_audit"].items():
        if any(item["exact_sample_overlap"] for item in audit.values()):
            raise RuntimeError(f"Exact anticipation sample leakage detected for {view}: {audit}")

    for session in range(4):
        if session > 0:
            with torch.no_grad():
                initialized = torch.stack(
                    [model.prompts[index].detach() for index in range(session)]
                ).mean(0)
                model.prompts[session].copy_(initialized)
        current = []
        for view_id, view in enumerate(train_views):
            records = build_session_stream(
                splits["train", view],
                session,
                seed,
                float(config["stream"]["r_D"]),
                float(config["stream"]["r_B"]),
                blurry_classes,
            )
            records = limit_records(records, int(config["runtime"]["max_train_samples_per_view"]), seed + session * 31 + view_id)
            current.extend(records)
        train_records = current + replay_records
        train_dataset = AnticipationDataset(train_records, feature_root, classes, int(config["data"]["segments"]))
        loader = DataLoader(
            train_dataset,
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            num_workers=int(config["runtime"]["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        for record in current:
            for label in record.labels:
                counts[label] += 1
        positive_weight = (counts.sum() / counts.clamp_min(1)).clamp(1, 20)
        model.train()
        for _ in range(int(config["training"]["epochs_per_session"])):
            for batch in loader:
                feature = batch["feature"].to(device)
                labels = batch["label"].to(device)
                view = batch["view"].to(device)
                result = model(feature, view, session)
                logits = result["logits"]
                bce = F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=positive_weight
                )
                probability = torch.sigmoid(logits)
                asymmetric = -(
                    labels * torch.log(probability.clamp_min(1e-6))
                    + (1 - labels)
                    * probability.pow(float(config["training"]["negative_focal_gamma"]))
                    * torch.log((1 - probability).clamp_min(1e-6))
                ).mean()
                loss = bce + float(config["training"]["asymmetric_weight"]) * asymmetric
                if teacher is not None:
                    with torch.no_grad():
                        old = teacher(feature, view, max(session - 1, 0))["logits"]
                    loss = loss + float(config["training"]["distill_weight"]) * F.mse_loss(
                        logits, old
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                with torch.no_grad():
                    model.update_prototypes(result["embedding"].detach(), labels)

        # Class-balanced replay records are selected without validation labels.
        replay_limit = int(config["training"]["replay_per_session"])
        replay_records.extend(limit_records(current, replay_limit, seed + 1000 + session))
        teacher = copy.deepcopy(model).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

        model.eval()
        seen = counts > 0
        prior = counts.clamp_min(1) / counts.clamp_min(1).sum()
        prior_correction = torch.zeros_like(prior)
        prior_correction[seen] = -float(config["evaluation"]["prior_tau"]) * prior[seen].log()
        task_payload = {
            "session": session + 1,
            "evaluation_protocol": "global_macro_top5_seen_sessions",
            "views": {},
            "global_macro_top5": {},
            "global_record_count": {},
        }
        for view in views:
            row = []
            if bool(config["evaluation"].get("task_matrix_diagnostic", True)):
                for eval_session in range(session + 1):
                    records = [r for r in splits[eval_split, view] if r.home_session == eval_session]
                    records = limit_records(records, int(config["runtime"]["max_eval_samples_per_view"]), seed + 2000 + eval_session)
                    row.append(
                        evaluate_records(
                            model,
                            records,
                            feature_root,
                            classes,
                            config,
                            device,
                            session,
                            prior_correction,
                            view in train_views,
                        )
                    )
            matrix[view].append(row)
            task_payload["views"][view] = row

            # Primary paper metric: merge all seen sessions before computing
            # class-wise recall. This avoids giving small tasks equal weight.
            global_records = [
                record
                for record in splits[eval_split, view]
                if record.home_session <= session
            ]
            global_records = limit_records(
                global_records,
                int(config["runtime"]["max_eval_samples_per_view"]),
                seed + 3000 + session,
            )
            global_score = evaluate_records(
                model,
                global_records,
                feature_root,
                classes,
                config,
                device,
                session,
                prior_correction,
                view in train_views,
            )
            global_curves[view].append(global_score)
            task_payload["global_macro_top5"][view] = global_score
            task_payload["global_record_count"][view] = len(global_records)
        checkpoint = {
            "session": session + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "class_sessions": class_sessions,
            "matrices": matrix,
            "global_curves": global_curves,
            "config": config,
        }
        torch.save(checkpoint, output / target / f"task_{session + 1:02d}.pt")
        (output / target / f"task_{session + 1:02d}.json").write_text(json.dumps(task_payload, indent=2), encoding="utf-8")

    metrics = {}
    for view in views:
        metrics[view] = {
            "A_last": 100 * global_curves[view][-1],
            "A_auc": 100 * sum(global_curves[view]) / len(global_curves[view]),
            "global_curve": global_curves[view],
            "task_accuracy_matrix_diagnostic": matrix[view],
        }
    return metrics


def load_completed_target(output: Path, target: str):
    """Recover metrics from a completed four-session target without retraining."""
    target_root = output / target
    checkpoint = target_root / "task_04.pt"
    task_files = [target_root / f"task_{session:02d}.json" for session in range(1, 5)]
    if not checkpoint.is_file() or not all(path.is_file() for path in task_files):
        return None
    matrices = {view: [] for view in ("ego", "exo")}
    global_curves = {view: [] for view in ("ego", "exo")}
    for path in task_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("session", -1)) != len(matrices["ego"]) + 1:
            return None
        for view in matrices:
            matrices[view].append([float(value) for value in payload["views"][view]])
            global_payload = payload.get("global_macro_top5", {})
            if view not in global_payload:
                # A v2 task-equal report is not compatible with the corrected
                # primary metric. Keep its files untouched and retrain only
                # when the caller explicitly reuses that old output path.
                return None
            global_curves[view].append(float(global_payload[view]))
    metrics = {}
    for view, matrix in matrices.items():
        metrics[view] = {
            "A_last": 100 * global_curves[view][-1],
            "A_auc": 100 * sum(global_curves[view]) / len(global_curves[view]),
            "global_curve": global_curves[view],
            "task_accuracy_matrix_diagnostic": matrix,
        }
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="FlyGCL EgoExoLearn continual anticipation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--annotation-root", help="Override data.annotation_root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", nargs="+", choices=["verb", "noun"], default=["verb", "noun"])
    parser.add_argument("--seed", type=int, help="Override config seed for a reproducible multi-seed run")
    parser.add_argument("--device", help="Override runtime.device")
    parser.add_argument("--num-workers", type=int, help="Override runtime.num_workers")
    parser.add_argument(
        "--input-setting",
        choices=["ego_exo", "ego_only", "exo_only"],
        help="Override the training-view setting without changing model hyperparameters",
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device is not None:
        config.setdefault("runtime", {})["device"] = args.device
    if args.num_workers is not None:
        config.setdefault("runtime", {})["num_workers"] = args.num_workers
    if args.input_setting is not None:
        config.setdefault("training", {})["input_setting"] = args.input_setting
    seed_all(int(config["seed"]))
    feature_root = Path(args.feature_root).resolve()
    if not feature_root.is_dir():
        raise FileNotFoundError(f"Official anticipation feature directory not found: {feature_root}")
    output = Path(args.output_dir).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError(f"Output must remain inside {ROOT}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    annotation_root = Path(args.annotation_root or config["data"]["annotation_root"])
    if not annotation_root.is_absolute():
        annotation_root = (PROJECT_ROOT / annotation_root).resolve()
    results = {}
    for target in args.targets:
        completed = load_completed_target(output, target)
        if completed is not None:
            print(f"[resume] reuse completed anticipation target: {target}")
            results[target] = completed
        else:
            results[target] = train_target(
                config, target, output, feature_root, annotation_root
            )
    cells_last = [results[target][view]["A_last"] for target in args.targets for view in ("ego", "exo")]
    cells_auc = [results[target][view]["A_auc"] for target in args.targets for view in ("ego", "exo")]
    payload = {
        "status": "complete",
        "method": "flygcl_cross_view_replay_prototype",
        "input_setting": str(config.get("training", {}).get("input_setting", "ego_exo")),
        "evaluation_protocol": "global_macro_top5_seen_sessions",
        "results": results,
        "Avg_A_last": sum(cells_last) / len(cells_last),
        "Avg_A_auc": sum(cells_auc) / len(cells_auc),
    }
    (output / "final_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if {"verb", "noun"}.issubset(results):
        columns = ["Metric", "Ego-V", "Ego-N", "Exo-V", "Exo-N", "Avg"]
        with (output / "table_s8_results.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for metric in ("A_last", "A_auc"):
                values = {
                    "Ego-V": results["verb"]["ego"][metric],
                    "Ego-N": results["noun"]["ego"][metric],
                    "Exo-V": results["verb"]["exo"][metric],
                    "Exo-N": results["noun"]["exo"][metric],
                }
                writer.writerow({"Metric": metric, **values, "Avg": sum(values.values()) / 4})
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

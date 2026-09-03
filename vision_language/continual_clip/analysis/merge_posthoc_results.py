import argparse
import math
import os
from typing import Dict, List, Tuple

import pandas as pd

from .io_utils import ensure_dir, safe_read_csv, safe_to_csv, save_json, warn_once


INPUT_FILES = [
    "alignment_posthoc.csv",
    "feature_posthoc.csv",
    "hard_negative_summary.csv",
    "attention_stage_summary.csv",
    "attention_method_comparison.csv",
    "seqlora_kv_layer_analysis.csv",
]

WIDE_FIELDS = [
    "method",
    "seed",
    "final_step",
    "final_pos_cos",
    "final_margin",
    "final_max_neg_cos",
    "delta_pos_cos",
    "delta_margin",
    "delta_max_neg_cos",
    "final_frozen_pos_cos",
    "final_frozen_margin",
    "final_frozen_max_neg_cos",
    "final_extra_margin_loss",
    "final_extra_pos_cos_change",
    "final_extra_max_neg_change",
    "final_gap_norm",
    "final_gap_direction_drift",
    "final_feature_drift",
    "final_feature_drift_std",
    "final_prototype_drift",
    "final_prototype_to_feature_ratio",
    "early_attention_drift",
    "mid_attention_drift",
    "high_attention_drift",
    "mean_attention_drift",
    "early_attention_distance_shift",
    "mid_attention_distance_shift",
    "high_attention_distance_shift",
    "final_old_acc",
    "final_new_acc",
]

LONG_FIELDS = [
    "method",
    "seed",
    "step_idx",
    "group_name",
    "pos_cos_mean",
    "margin_mean",
    "max_neg_cos_mean",
    "delta_pos_cos",
    "delta_margin",
    "delta_max_neg_cos",
    "frozen_pos_cos",
    "frozen_margin",
    "frozen_max_neg_cos",
    "extra_margin_loss",
    "extra_pos_cos_change",
    "extra_max_neg_change",
    "gap_norm",
    "gap_direction_drift",
    "feature_drift_mean",
    "prototype_drift_mean",
    "early_attention_drift",
    "mid_attention_drift",
    "high_attention_drift",
    "old_acc",
    "new_acc",
]


def _parse_method_dir(values: List[str]) -> List[Tuple[str, str]]:
    return list(_parse_method_mapping(values, "--method_dir").items())


def _parse_method_mapping(values: List[str], option_name: str) -> Dict[str, str]:
    out = {}
    if not values:
        return out
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option_name} must be method=/path format, got: {value}")
        method, path = value.split("=", 1)
        method = method.strip()
        path = path.strip()
        if not method or not path:
            raise ValueError(f"{option_name} must be method=/path format, got: {value}")
        out[method] = path
    return out


def _read_method_dir(
    method: str,
    posthoc_dir: str,
    summary_paths: Dict[str, str],
    analysis_dirs: Dict[str, str],
) -> Tuple[Dict[str, pd.DataFrame], List[str], List[str], Dict]:
    data = {}
    found, missing = [], []
    for name in INPUT_FILES:
        path = os.path.join(posthoc_dir, name)
        if os.path.exists(path):
            found.append(name)
        else:
            missing.append(name)
        data[name] = safe_read_csv(path)
    summary_path = _find_summary_path(method, posthoc_dir, summary_paths, analysis_dirs)
    summary_df = safe_read_csv(summary_path) if summary_path else pd.DataFrame()
    data["summary.csv"] = summary_df
    if summary_path:
        found.append(_display_path(summary_path, posthoc_dir))
    else:
        missing.append("summary.csv")
        warn_once(f"{method}: summary.csv not found; old/new acc fields will be NaN")
    summary_info = {
        "summary_source": summary_path,
        "summary_found": bool(summary_path and not summary_df.empty),
        "summary_num_rows": int(len(summary_df)) if summary_path else 0,
        "summary_columns": summary_df.columns.tolist() if summary_path else [],
    }
    return data, found, missing, summary_info


def _display_path(path: str, base_dir: str) -> str:
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return path


def _find_summary_path(
    method: str,
    posthoc_dir: str,
    summary_paths: Dict[str, str],
    analysis_dirs: Dict[str, str],
):
    candidates = []
    if method in summary_paths:
        candidates.append(summary_paths[method])
    if method in analysis_dirs:
        candidates.append(os.path.join(analysis_dirs[method], "summary.csv"))
    candidates.extend([
        os.path.join(posthoc_dir, "summary.csv"),
        os.path.join(os.path.dirname(posthoc_dir), "summary.csv"),
    ])
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def _group_or_first(df: pd.DataFrame, method: str, filename: str, preferred: str = "all") -> pd.DataFrame:
    if df.empty or "group_name" not in df.columns:
        return df
    groups = [g for g in df["group_name"].dropna().unique().tolist()]
    if preferred in groups:
        return df[df["group_name"] == preferred].copy()
    if groups:
        warn_once(f"{method}/{filename}: group 'all' missing; use first group '{groups[0]}'")
        return df[df["group_name"] == groups[0]].copy()
    return df


def _final_row(df: pd.DataFrame, method: str, filename: str, group_filter: bool = True) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    work = _group_or_first(df, method, filename) if group_filter else df.copy()
    if work.empty:
        return pd.Series(dtype=object)
    if "step_idx" in work.columns:
        work = work.copy()
        work["step_idx"] = pd.to_numeric(work["step_idx"], errors="coerce")
        max_step = work["step_idx"].max()
        work = work[work["step_idx"] == max_step]
    elif "final_step" in work.columns:
        work = work.copy()
        work["final_step"] = pd.to_numeric(work["final_step"], errors="coerce")
        max_step = work["final_step"].max()
        work = work[work["final_step"] == max_step]
    return work.iloc[0] if not work.empty else pd.Series(dtype=object)


def _value(row, name, default=math.nan):
    if row is None or len(row) == 0:
        return default
    return row[name] if name in row.index else default


def _normalize_summary(summary: pd.DataFrame, method: str) -> pd.DataFrame:
    out = pd.DataFrame(columns=["step_idx", "old_acc", "new_acc"])
    if summary.empty:
        return out
    if "step_idx" not in summary.columns:
        warn_once(f"{method}/summary.csv missing step_idx; old/new acc cannot be merged by step")
        return out
    old_col = _first_existing_column(summary, ["old_acc", "old_acc_if_available", "old_exposed_acc"])
    new_col = _first_existing_column(summary, ["new_acc", "new_acc_if_available", "new_exposed_acc"])
    if old_col is None:
        warn_once(f"{method}/summary.csv missing old_acc or old_acc_if_available; old_acc will be NaN")
    if new_col is None:
        warn_once(f"{method}/summary.csv missing new_acc or new_acc_if_available; new_acc will be NaN")

    out["step_idx"] = pd.to_numeric(summary["step_idx"], errors="coerce")
    out["old_acc"] = pd.to_numeric(summary[old_col], errors="coerce") if old_col else math.nan
    out["new_acc"] = pd.to_numeric(summary[new_col], errors="coerce") if new_col else math.nan
    out = out.dropna(subset=["step_idx"])
    if out.empty:
        return out
    return out.groupby("step_idx", as_index=False, sort=True).last()


def _first_existing_column(df: pd.DataFrame, names: List[str]):
    for name in names:
        if name in df.columns:
            return name
    return None


def _build_attention_from_stage(attention_stage: pd.DataFrame, method: str) -> Dict[str, float]:
    if attention_stage.empty:
        return {}
    df = _group_or_first(attention_stage, method, "attention_stage_summary.csv")
    if df.empty or "step_idx" not in df.columns:
        return {}
    df = df.copy()
    df["step_idx"] = pd.to_numeric(df["step_idx"], errors="coerce")
    final_step = df["step_idx"].max()
    final = df[df["step_idx"] == final_step]
    out = {"final_step": final_step, "mean_drift": final.get("attention_drift_js_mean", pd.Series(dtype=float)).mean()}
    for stage in ["early", "mid", "high"]:
        stage_df = final[final["stage"] == stage] if "stage" in final.columns else pd.DataFrame()
        out[f"{stage}_drift"] = (
            stage_df["attention_drift_js_mean"].mean()
            if not stage_df.empty and "attention_drift_js_mean" in stage_df.columns
            else math.nan
        )
        out[f"{stage}_distance_shift"] = (
            stage_df["attention_distance_shift"].mean()
            if not stage_df.empty and "attention_distance_shift" in stage_df.columns
            else math.nan
        )
    return out


def _final_old_new_acc(summary: pd.DataFrame, final_step, method: str) -> Tuple[float, float]:
    if summary.empty:
        return math.nan, math.nan
    work = summary.copy()
    if not pd.isna(final_step):
        final_step_num = pd.to_numeric(pd.Series([final_step]), errors="coerce").iloc[0]
        matched = work[work["step_idx"] == final_step_num]
        if not matched.empty:
            row = matched.iloc[-1]
            return _value(row, "old_acc"), _value(row, "new_acc")
        warn_once(f"{method}/summary.csv missing final_step={final_step_num}; use summary max step for final old/new acc")
    max_step = work["step_idx"].max()
    matched = work[work["step_idx"] == max_step]
    if not matched.empty:
        row = matched.iloc[-1]
        return _value(row, "old_acc"), _value(row, "new_acc")
    return math.nan, math.nan


def _wide_row(method: str, seed: int, data: Dict[str, pd.DataFrame]) -> Dict:
    alignment = _final_row(data["alignment_posthoc.csv"], method, "alignment_posthoc.csv")
    feature = _final_row(data["feature_posthoc.csv"], method, "feature_posthoc.csv")
    att_comp = data["attention_method_comparison.csv"]
    if not att_comp.empty:
        att = _final_row(att_comp, method, "attention_method_comparison.csv", group_filter=False)
        att_values = {
            "final_step": _value(att, "final_step"),
            "early_drift": _value(att, "early_drift"),
            "mid_drift": _value(att, "mid_drift"),
            "high_drift": _value(att, "high_drift"),
            "mean_drift": _value(att, "mean_drift"),
            "early_distance_shift": _value(att, "early_distance_shift"),
            "mid_distance_shift": _value(att, "mid_distance_shift"),
            "high_distance_shift": _value(att, "high_distance_shift"),
        }
    else:
        att_values = _build_attention_from_stage(data["attention_stage_summary.csv"], method)

    final_step = _value(alignment, "step_idx", _value(feature, "step_idx", att_values.get("final_step", math.nan)))
    summary = _normalize_summary(data.get("summary.csv", pd.DataFrame()), method)
    old_acc, new_acc = _final_old_new_acc(summary, final_step, method)
    return {
        "method": method,
        "seed": seed,
        "final_step": final_step,
        "final_pos_cos": _value(alignment, "pos_cos_mean"),
        "final_margin": _value(alignment, "margin_mean"),
        "final_max_neg_cos": _value(alignment, "max_neg_cos_mean"),
        "delta_pos_cos": _value(alignment, "delta_pos_cos"),
        "delta_margin": _value(alignment, "delta_margin"),
        "delta_max_neg_cos": _value(alignment, "delta_max_neg_cos"),
        "final_frozen_pos_cos": _value(alignment, "frozen_pos_cos"),
        "final_frozen_margin": _value(alignment, "frozen_margin"),
        "final_frozen_max_neg_cos": _value(alignment, "frozen_max_neg_cos"),
        "final_extra_margin_loss": _value(alignment, "extra_margin_loss"),
        "final_extra_pos_cos_change": _value(alignment, "extra_pos_cos_change"),
        "final_extra_max_neg_change": _value(alignment, "extra_max_neg_change"),
        "final_gap_norm": _value(alignment, "gap_norm"),
        "final_gap_direction_drift": _value(alignment, "gap_direction_drift"),
        "final_feature_drift": _value(feature, "feature_drift_mean"),
        "final_feature_drift_std": _value(feature, "feature_drift_std"),
        "final_prototype_drift": _value(feature, "prototype_drift_mean"),
        "final_prototype_to_feature_ratio": _value(feature, "prototype_to_feature_ratio"),
        "early_attention_drift": att_values.get("early_drift", math.nan),
        "mid_attention_drift": att_values.get("mid_drift", math.nan),
        "high_attention_drift": att_values.get("high_drift", math.nan),
        "mean_attention_drift": att_values.get("mean_drift", math.nan),
        "early_attention_distance_shift": att_values.get("early_distance_shift", math.nan),
        "mid_attention_distance_shift": att_values.get("mid_distance_shift", math.nan),
        "high_attention_distance_shift": att_values.get("high_distance_shift", math.nan),
        "final_old_acc": old_acc,
        "final_new_acc": new_acc,
    }


def _attention_stage_wide(attention_stage: pd.DataFrame, method: str) -> pd.DataFrame:
    if attention_stage.empty:
        return pd.DataFrame(columns=["step_idx", "group_name", "early_attention_drift", "mid_attention_drift", "high_attention_drift"])
    df = _group_or_first(attention_stage, method, "attention_stage_summary.csv")
    if df.empty:
        return pd.DataFrame(columns=["step_idx", "group_name", "early_attention_drift", "mid_attention_drift", "high_attention_drift"])
    for col in ["step_idx", "group_name", "stage", "attention_drift_js_mean"]:
        if col not in df.columns:
            warn_once(f"attention_stage_summary.csv missing {col}; attention long fields may be NaN")
            df[col] = math.nan
    pivot = df.pivot_table(
        index=["step_idx", "group_name"],
        columns="stage",
        values="attention_drift_js_mean",
        aggfunc="mean",
    ).reset_index()
    return pivot.rename(columns={
        "early": "early_attention_drift",
        "mid": "mid_attention_drift",
        "high": "high_attention_drift",
    })


def _long_rows(method: str, seed: int, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    align = _group_or_first(data["alignment_posthoc.csv"], method, "alignment_posthoc.csv")
    feat = _group_or_first(data["feature_posthoc.csv"], method, "feature_posthoc.csv")
    att = _attention_stage_wide(data["attention_stage_summary.csv"], method)
    summary = _normalize_summary(data.get("summary.csv", pd.DataFrame()), method)

    for df in [align, feat, att, summary]:
        if not df.empty and "step_idx" in df.columns:
            df["step_idx"] = pd.to_numeric(df["step_idx"], errors="coerce")

    base = None
    if not align.empty:
        keep = [
            "step_idx",
            "group_name",
            "pos_cos_mean",
            "margin_mean",
            "max_neg_cos_mean",
            "delta_pos_cos",
            "delta_margin",
            "delta_max_neg_cos",
            "frozen_pos_cos",
            "frozen_margin",
            "frozen_max_neg_cos",
            "extra_margin_loss",
            "extra_pos_cos_change",
            "extra_max_neg_change",
            "gap_norm",
            "gap_direction_drift",
        ]
        base = align[[c for c in keep if c in align.columns]].copy()
    if base is None or base.empty:
        if not feat.empty and {"step_idx", "group_name"}.issubset(feat.columns):
            base = feat[["step_idx", "group_name"]].copy()
        elif not att.empty and {"step_idx", "group_name"}.issubset(att.columns):
            base = att[["step_idx", "group_name"]].copy()
        elif not summary.empty and "step_idx" in summary.columns:
            base = summary[["step_idx"]].copy()
            base["group_name"] = "all"
        else:
            return pd.DataFrame(columns=LONG_FIELDS)

    if "group_name" not in base.columns:
        base["group_name"] = "all"

    if not feat.empty:
        fkeep = ["step_idx", "group_name", "feature_drift_mean", "prototype_drift_mean"]
        feat_m = feat[[c for c in fkeep if c in feat.columns]].copy()
        base = base.merge(feat_m, on=["step_idx", "group_name"], how="outer")

    if not att.empty:
        base = base.merge(att, on=["step_idx", "group_name"], how="outer")

    if not summary.empty:
        base = base.merge(summary, on="step_idx", how="left")

    base.insert(0, "seed", seed)
    base.insert(0, "method", method)
    for field in LONG_FIELDS:
        if field not in base.columns:
            base[field] = math.nan
    return base[LONG_FIELDS].sort_values(["method", "step_idx", "group_name"])


def merge_posthoc_results(
    method_dirs: List[Tuple[str, str]],
    output_dir: str,
    seed: int,
    summary_paths: Dict[str, str] = None,
    analysis_dirs: Dict[str, str] = None,
) -> Dict:
    summary_paths = summary_paths or {}
    analysis_dirs = analysis_dirs or {}
    ensure_dir(output_dir)
    wide_rows = []
    long_frames = []
    report = {"methods": {}, "output_files": []}

    for method, posthoc_dir in method_dirs:
        data, found, missing, summary_info = _read_method_dir(method, posthoc_dir, summary_paths, analysis_dirs)
        wide_rows.append(_wide_row(method, seed, data))
        long_df = _long_rows(method, seed, data)
        if not long_df.empty:
            long_frames.append(long_df)
        steps = set()
        for df in data.values():
            if not df.empty and "step_idx" in df.columns:
                steps.update(pd.to_numeric(df["step_idx"], errors="coerce").dropna().astype(int).tolist())
        final_step = max(steps) if steps else math.nan
        report["methods"][method] = {
            "posthoc_dir": posthoc_dir,
            "found_files": found,
            "missing_files": missing,
            "num_steps": len(steps),
            "final_step": final_step,
            **summary_info,
        }

    wide = pd.DataFrame(wide_rows, columns=WIDE_FIELDS)
    long = pd.concat(long_frames, ignore_index=True) if long_frames else pd.DataFrame(columns=LONG_FIELDS)
    long = long[LONG_FIELDS] if not long.empty else long

    wide_path = os.path.join(output_dir, f"method_comparison_seed{seed}.csv")
    long_path = os.path.join(output_dir, "method_comparison_long.csv")
    report_path = os.path.join(output_dir, "analysis_quality_report.json")
    safe_to_csv(wide, wide_path)
    safe_to_csv(long, long_path)
    report["output_files"] = [wide_path, long_path, report_path]
    save_json(report_path, report)
    return report


def build_argparser():
    parser = argparse.ArgumentParser(description="Merge CLIP-GCL post-hoc analysis outputs across methods.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method_dir", action="append", required=True, help="Repeated method=/path/to/posthoc_dir")
    parser.add_argument("--method_summary", action="append", default=[], help="Repeated method=/path/to/summary.csv")
    parser.add_argument("--method_analysis_dir", action="append", default=[], help="Repeated method=/path/to/original_analysis_dir")
    return parser


def main(argv=None):
    parser = build_argparser()
    args = parser.parse_args(argv)
    method_dirs = _parse_method_dir(args.method_dir)
    summary_paths = _parse_method_mapping(args.method_summary, "--method_summary")
    analysis_dirs = _parse_method_mapping(args.method_analysis_dir, "--method_analysis_dir")
    report = merge_posthoc_results(method_dirs, args.output_dir, args.seed, summary_paths, analysis_dirs)
    for path in report["output_files"]:
        print(path)


if __name__ == "__main__":
    main()

import argparse
import math
import os
from typing import Dict

import pandas as pd

from .classwise_analysis import empty_classwise_analysis
from .hard_negative_analysis import build_hard_negative_summary
from .io_utils import ensure_dir, safe_read_csv, safe_to_csv, warn_once
from .prototype_geometry import empty_prototype_geometry


CSV_INPUTS = {
    "summary": "summary.csv",
    "clip_alignment": "clip_alignment.csv",
    "feature_drift": "feature_drift.csv",
    "text_feature_drift": "text_feature_drift.csv",
    "attention_stats": "attention_stats.csv",
    "lora_stats": "lora_stats.csv",
    "grad_conflict": "grad_conflict.csv",
}


def _read_inputs(analysis_dir: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for key, name in CSV_INPUTS.items():
        path = os.path.join(analysis_dir, name)
        out[key] = safe_read_csv(path)
    return out


def _with_method_seed(df: pd.DataFrame, method_name: str, seed) -> pd.DataFrame:
    out = df.copy()
    if "method" not in out.columns:
        out.insert(0, "method", method_name)
    if "seed" not in out.columns:
        out.insert(1, "seed", seed)
    return out


def build_alignment_posthoc(clip_alignment: pd.DataFrame, method_name: str, seed, frozen_clip_path=None):
    fields = [
        "method",
        "seed",
        "step_idx",
        "group_name",
        "pos_cos_mean",
        "margin_mean",
        "max_neg_cos_mean",
        "gap_norm",
        "gap_direction_drift",
        "delta_pos_cos",
        "delta_margin",
        "delta_max_neg_cos",
        "frozen_pos_cos",
        "frozen_margin",
        "frozen_max_neg_cos",
        "extra_margin_loss",
        "extra_pos_cos_change",
        "extra_max_neg_change",
    ]
    if clip_alignment.empty:
        return pd.DataFrame(columns=fields)
    df = _with_method_seed(clip_alignment.copy(), method_name, seed)
    for col in ["pos_cos_mean", "margin_mean", "gap_norm", "gap_direction_drift"]:
        if col not in df.columns:
            df[col] = math.nan
    df["max_neg_cos_mean"] = df["pos_cos_mean"] - df["margin_mean"]
    keys = ["group_name"]
    base = df.sort_values("step_idx").groupby(keys, dropna=False).first().reset_index()
    base = base[keys + ["pos_cos_mean", "margin_mean", "max_neg_cos_mean"]].rename(columns={
        "pos_cos_mean": "base_pos_cos",
        "margin_mean": "base_margin",
        "max_neg_cos_mean": "base_max_neg",
    })
    df = df.merge(base, on=keys, how="left")
    df["delta_pos_cos"] = df["pos_cos_mean"] - df["base_pos_cos"]
    df["delta_margin"] = df["margin_mean"] - df["base_margin"]
    df["delta_max_neg_cos"] = df["max_neg_cos_mean"] - df["base_max_neg"]
    df["extra_margin_loss"] = math.nan
    df["extra_pos_cos_change"] = math.nan
    df["extra_max_neg_change"] = math.nan
    df["frozen_pos_cos"] = math.nan
    df["frozen_margin"] = math.nan
    df["frozen_max_neg_cos"] = math.nan

    if frozen_clip_path:
        frozen = safe_read_csv(frozen_clip_path)
        if not frozen.empty:
            f = frozen.copy()
            if "frozen_pos_cos" not in f.columns and "pos_cos_mean" in f.columns:
                f["frozen_pos_cos"] = f["pos_cos_mean"]
            if "frozen_margin" not in f.columns and "margin_mean" in f.columns:
                f["frozen_margin"] = f["margin_mean"]
            if "frozen_max_neg_cos" not in f.columns:
                if "max_neg_cos_mean" in f.columns:
                    f["frozen_max_neg_cos"] = f["max_neg_cos_mean"]
                elif {"frozen_pos_cos", "frozen_margin"}.issubset(f.columns):
                    f["frozen_max_neg_cos"] = f["frozen_pos_cos"] - f["frozen_margin"]
            keep = ["step_idx", "group_name", "frozen_pos_cos", "frozen_margin", "frozen_max_neg_cos"]
            f = f[[c for c in keep if c in f.columns]]
            df = df.drop(columns=["frozen_pos_cos", "frozen_margin", "frozen_max_neg_cos"], errors="ignore")
            df = df.merge(f, on=["step_idx", "group_name"], how="left")
            df["extra_margin_loss"] = df["margin_mean"] - df.get("frozen_margin", math.nan)
            df["extra_pos_cos_change"] = df["pos_cos_mean"] - df.get("frozen_pos_cos", math.nan)
            df["extra_max_neg_change"] = df["max_neg_cos_mean"] - df.get("frozen_max_neg_cos", math.nan)
    for col in ["frozen_pos_cos", "frozen_margin", "frozen_max_neg_cos"]:
        if col not in df.columns:
            df[col] = math.nan
    return df[fields]


def build_feature_posthoc(feature_drift: pd.DataFrame, method_name: str, seed):
    fields = [
        "method",
        "seed",
        "step_idx",
        "group_name",
        "feature_drift_mean",
        "feature_drift_std",
        "prototype_drift_mean",
        "delta_feature_drift",
        "delta_prototype_drift",
        "prototype_to_feature_ratio",
    ]
    if feature_drift.empty:
        return pd.DataFrame(columns=fields)
    df = _with_method_seed(feature_drift.copy(), method_name, seed)
    for col in ["feature_drift_mean", "feature_drift_std", "prototype_drift_mean"]:
        if col not in df.columns:
            df[col] = math.nan
    keys = ["group_name"]
    base = df.sort_values("step_idx").groupby(keys, dropna=False).first().reset_index()
    base = base[keys + ["feature_drift_mean", "prototype_drift_mean"]].rename(columns={
        "feature_drift_mean": "base_feature_drift",
        "prototype_drift_mean": "base_prototype_drift",
    })
    df = df.merge(base, on=keys, how="left")
    df["delta_feature_drift"] = df["feature_drift_mean"] - df["base_feature_drift"]
    df["delta_prototype_drift"] = df["prototype_drift_mean"] - df["base_prototype_drift"]
    df["prototype_to_feature_ratio"] = df["prototype_drift_mean"] / df["feature_drift_mean"]
    return df[fields]


def _stage_for_layer(layer_idx: int, max_layer: int) -> str:
    if max_layer == 11:
        if layer_idx <= 3:
            return "early"
        if layer_idx <= 7:
            return "mid"
        return "high"
    total = max_layer + 1
    cut1 = max(1, total // 3)
    cut2 = max(cut1 + 1, (2 * total) // 3)
    if layer_idx < cut1:
        return "early"
    if layer_idx < cut2:
        return "mid"
    return "high"


def build_attention_stage_summary(attention_stats: pd.DataFrame, method_name: str, seed):
    fields = [
        "method",
        "seed",
        "step_idx",
        "group_name",
        "stage",
        "attention_drift_js_mean",
        "attention_entropy_mean",
        "topk_overlap_mean",
        "attention_distance_mean",
        "attention_distance_shift",
    ]
    if attention_stats.empty or "layer_idx" not in attention_stats.columns:
        return pd.DataFrame(columns=fields)
    df = _with_method_seed(attention_stats.copy(), method_name, seed)
    for col in ["attention_drift_js", "attention_entropy", "topk_overlap", "attention_distance"]:
        if col not in df.columns:
            df[col] = math.nan
    df["layer_idx"] = pd.to_numeric(df["layer_idx"], errors="coerce")
    max_layer = int(df["layer_idx"].max()) if not df["layer_idx"].dropna().empty else 0
    df["stage"] = df["layer_idx"].fillna(0).astype(int).map(lambda idx: _stage_for_layer(idx, max_layer))
    agg = df.groupby(["method", "seed", "step_idx", "group_name", "stage"], dropna=False).agg(
        attention_drift_js_mean=("attention_drift_js", "mean"),
        attention_entropy_mean=("attention_entropy", "mean"),
        topk_overlap_mean=("topk_overlap", "mean"),
        attention_distance_mean=("attention_distance", "mean"),
    ).reset_index()
    base = agg.sort_values("step_idx").groupby(["group_name", "stage"], dropna=False).first().reset_index()
    base = base[["group_name", "stage", "attention_distance_mean"]].rename(columns={
        "attention_distance_mean": "base_attention_distance",
    })
    agg = agg.merge(base, on=["group_name", "stage"], how="left")
    agg["attention_distance_shift"] = agg["attention_distance_mean"] - agg["base_attention_distance"]
    return agg[fields]


def build_attention_method_comparison(stage_summary: pd.DataFrame, method_name: str, seed):
    fields = [
        "method",
        "seed",
        "final_step",
        "early_drift",
        "mid_drift",
        "high_drift",
        "mean_drift",
        "early_distance_shift",
        "mid_distance_shift",
        "high_distance_shift",
    ]
    if stage_summary.empty:
        return pd.DataFrame(columns=fields)
    final_step = stage_summary["step_idx"].max()
    final = stage_summary[stage_summary["step_idx"] == final_step]
    row = {
        "method": method_name,
        "seed": seed,
        "final_step": final_step,
        "mean_drift": final["attention_drift_js_mean"].mean(),
    }
    for stage in ["early", "mid", "high"]:
        stage_df = final[final["stage"] == stage]
        row[f"{stage}_drift"] = stage_df["attention_drift_js_mean"].mean() if not stage_df.empty else math.nan
        row[f"{stage}_distance_shift"] = stage_df["attention_distance_shift"].mean() if not stage_df.empty else math.nan
    return pd.DataFrame([row], columns=fields)


def _normalize_matrix_type(value) -> str:
    text = str(value).lower()
    if text in {"k", "key", "lora_k", "k_proj"} or "k" == text[-1:]:
        return "k"
    if text in {"v", "value", "lora_v", "v_proj"} or "v" == text[-1:]:
        return "v"
    if text in {"q", "query", "lora_q", "q_proj"} or "q" == text[-1:]:
        return "q"
    if "key" in text or ".k" in text or "k_proj" in text:
        return "k"
    if "value" in text or ".v" in text or "v_proj" in text:
        return "v"
    if "query" in text or ".q" in text or "q_proj" in text:
        return "q"
    return "unknown"


def build_seqlora_kv_layer_analysis(
    lora_stats: pd.DataFrame,
    attention_stats: pd.DataFrame,
    feature_drift: pd.DataFrame,
    clip_alignment: pd.DataFrame,
    method_name: str,
    seed,
):
    fields = [
        "method",
        "seed",
        "step_idx",
        "layer_idx",
        "k_norm",
        "v_norm",
        "k_effective_rank",
        "v_effective_rank",
        "k_delta_cos_prev",
        "v_delta_cos_prev",
        "attention_drift_js",
        "feature_drift_mean",
        "margin_mean",
    ]
    if lora_stats.empty:
        warn_once("lora_stats.csv missing; skip seqlora_kv_layer_analysis")
        return pd.DataFrame(columns=fields)
    df = _with_method_seed(lora_stats.copy(), method_name, seed)
    if "layer_idx" not in df.columns:
        warn_once("lora_stats.csv has no layer_idx; skip seqlora_kv_layer_analysis")
        return pd.DataFrame(columns=fields)
    matrix_source = "matrix_type" if "matrix_type" in df.columns else "module_name"
    df["matrix_type_norm"] = df[matrix_source].map(_normalize_matrix_type)
    df = df[df["matrix_type_norm"].isin(["k", "v"])]
    if df.empty:
        warn_once("lora_stats.csv contains no K/V matrices; skip seqlora_kv_layer_analysis")
        return pd.DataFrame(columns=fields)
    pivot = df.pivot_table(
        index=["method", "seed", "step_idx", "layer_idx"],
        columns="matrix_type_norm",
        values=["delta_fro_norm", "effective_rank", "delta_cos_prev"],
        aggfunc="mean",
    ).reset_index()
    pivot.columns = [
        "_".join(str(v) for v in col if v != "").rstrip("_") if isinstance(col, tuple) else str(col)
        for col in pivot.columns
    ]
    out = pivot.rename(columns={
        "delta_fro_norm_k": "k_norm",
        "delta_fro_norm_v": "v_norm",
        "effective_rank_k": "k_effective_rank",
        "effective_rank_v": "v_effective_rank",
        "delta_cos_prev_k": "k_delta_cos_prev",
        "delta_cos_prev_v": "v_delta_cos_prev",
    })
    if not attention_stats.empty and {"step_idx", "layer_idx", "attention_drift_js"}.issubset(attention_stats.columns):
        att = attention_stats.groupby(["step_idx", "layer_idx"], dropna=False)["attention_drift_js"].mean().reset_index()
        out = out.merge(att, on=["step_idx", "layer_idx"], how="left")
    else:
        out["attention_drift_js"] = math.nan
    if not feature_drift.empty and "feature_drift_mean" in feature_drift.columns:
        feat = feature_drift.groupby("step_idx", dropna=False)["feature_drift_mean"].mean().reset_index()
        out = out.merge(feat, on="step_idx", how="left")
    else:
        out["feature_drift_mean"] = math.nan
    if not clip_alignment.empty and "margin_mean" in clip_alignment.columns:
        margin = clip_alignment.groupby("step_idx", dropna=False)["margin_mean"].mean().reset_index()
        out = out.merge(margin, on="step_idx", how="left")
    else:
        out["margin_mean"] = math.nan
    for field in fields:
        if field not in out.columns:
            out[field] = math.nan
    return out[fields]


def run_posthoc_analysis(args) -> Dict[str, str]:
    ensure_dir(args.output_dir)
    data = _read_inputs(args.analysis_dir)
    outputs = {}

    if args.enable_alignment_posthoc:
        df = build_alignment_posthoc(
            data["clip_alignment"],
            args.method_name,
            args.seed,
            frozen_clip_path=args.frozen_clip_reference_path,
        )
        path = os.path.join(args.output_dir, "alignment_posthoc.csv")
        safe_to_csv(df, path)
        outputs["alignment_posthoc"] = path

    if args.enable_feature_posthoc:
        df = build_feature_posthoc(data["feature_drift"], args.method_name, args.seed)
        path = os.path.join(args.output_dir, "feature_posthoc.csv")
        safe_to_csv(df, path)
        outputs["feature_posthoc"] = path
        text_df = build_feature_posthoc(data["text_feature_drift"], args.method_name, args.seed)
        text_path = os.path.join(args.output_dir, "text_feature_posthoc.csv")
        safe_to_csv(text_df, text_path)
        outputs["text_feature_posthoc"] = text_path

    if args.enable_hard_negative_analysis:
        df = build_hard_negative_summary(data["clip_alignment"], args.method_name, args.seed)
        path = os.path.join(args.output_dir, "hard_negative_summary.csv")
        safe_to_csv(df, path)
        outputs["hard_negative_summary"] = path

    if args.enable_prototype_geometry_analysis:
        df = empty_prototype_geometry(args.method_name, args.seed)
        path = os.path.join(args.output_dir, "prototype_geometry.csv")
        safe_to_csv(df, path)
        outputs["prototype_geometry"] = path

    if args.enable_attention_stage_summary:
        df = build_attention_stage_summary(data["attention_stats"], args.method_name, args.seed)
        path = os.path.join(args.output_dir, "attention_stage_summary.csv")
        safe_to_csv(df, path)
        outputs["attention_stage_summary"] = path
        comp = build_attention_method_comparison(df, args.method_name, args.seed)
        comp_path = os.path.join(args.output_dir, "attention_method_comparison.csv")
        safe_to_csv(comp, comp_path)
        outputs["attention_method_comparison"] = comp_path

    if args.enable_kv_layer_posthoc:
        if "seq" not in str(args.method_name).lower() and "lora" not in str(args.method_name).lower():
            warn_once("method is not SeqLoRA/LoRA; skip seqlora_kv_layer_analysis")
        else:
            df = build_seqlora_kv_layer_analysis(
                data["lora_stats"],
                data["attention_stats"],
                data["feature_drift"],
                data["clip_alignment"],
                args.method_name,
                args.seed,
            )
            if not df.empty:
                path = os.path.join(args.output_dir, "seqlora_kv_layer_analysis.csv")
                safe_to_csv(df, path)
                outputs["seqlora_kv_layer_analysis"] = path

    if args.enable_classwise_analysis:
        df = empty_classwise_analysis()
        path = os.path.join(args.output_dir, "classwise_analysis.csv")
        safe_to_csv(df, path)
        outputs["classwise_analysis"] = path

    if args.enable_frozen_clip_reference:
        warn_once("frozen CLIP reference requires explicit experiment execution; no-op in post-hoc")
    if args.enable_layer_rollback_analysis:
        warn_once("layer rollback requires explicit experiment execution; no-op in post-hoc")
    if args.enable_svd_direction_ablation:
        warn_once("SVD direction ablation requires explicit experiment execution; no-op in post-hoc")

    return outputs


def build_argparser():
    parser = argparse.ArgumentParser(description="Post-hoc analysis for CLIP-GCL diagnostic CSVs.")
    parser.add_argument("--run_posthoc_analysis", action="store_true", default=False)
    parser.add_argument("--analysis_dir", "--posthoc_analysis_dir", dest="analysis_dir", required=True)
    parser.add_argument("--output_dir", "--posthoc_output_dir", dest="output_dir", required=True)
    parser.add_argument("--method_name", "--posthoc_method_name", dest="method_name", default="unknown")
    parser.add_argument("--seed", "--posthoc_seed", dest="seed", type=int, default=-1)
    parser.add_argument("--frozen_clip_reference_path", default="")

    parser.add_argument("--enable_groupwise_analysis", action="store_true", default=False)
    parser.add_argument("--enable_classwise_analysis", action="store_true", default=False)
    parser.add_argument("--enable_hard_negative_analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_prototype_geometry_analysis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_attention_stage_summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_alignment_posthoc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_feature_posthoc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_kv_layer_posthoc", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--enable_frozen_clip_reference", action="store_true", default=False)
    parser.add_argument("--enable_layer_rollback_analysis", action="store_true", default=False)
    parser.add_argument("--enable_svd_direction_ablation", action="store_true", default=False)
    return parser


def main(argv=None):
    parser = build_argparser()
    args = parser.parse_args(argv)
    outputs = run_posthoc_analysis(args)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

import os
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .attention_stats import compute_attention_stats
from .clip_alignment_stats import compute_clip_alignment_stats
from .feature_stats import compute_feature_drift_stats, compute_text_feature_drift_stats
from .grad_conflict_stats import compute_lora_gradient_conflict
from .io_utils import append_rows_csv, append_csv, ensure_dir, extract_batch, limit_batches, save_json, warn_once
from .lora_stats import collect_lora_stats
from .schema_metadata import write_analysis_schema_metadata


SUMMARY_FIELDS = [
    "step_idx",
    "task_idx",
    "num_samples",
    "old_acc_if_available",
    "new_acc_if_available",
    "mean_old_pos_cos",
    "mean_new_pos_cos",
    "mean_old_margin",
    "mean_new_margin",
    "modality_gap_norm",
    "modality_gap_drift",
    "mean_feature_drift",
    "mean_text_feature_drift",
    "mean_attention_drift",
    "mean_lora_norm",
    "mean_lora_effective_rank",
    "mean_grad_cosine",
]

LORA_FIELDS = [
    "step_idx",
    "task_idx",
    "layer_name",
    "encoder_side",
    "layer_idx",
    "block_idx",
    "module_name",
    "matrix_type",
    "delta_fro_norm",
    "delta_spectral_norm",
    "effective_rank",
    "norm_ratio",
    "delta_cos_prev",
]

CLIP_ALIGNMENT_FIELDS = [
    "step_idx",
    "task_idx",
    "encoder_side",
    "group_name",
    "pos_cos_mean",
    "pos_cos_std",
    "margin_mean",
    "margin_std",
    "gap_norm",
    "gap_direction_drift",
    "max_neg_cos_mean",
    "max_neg_cos_std",
    "hard_neg_is_new_ratio",
]

FEATURE_FIELDS = [
    "step_idx",
    "task_idx",
    "group_name",
    "layer_idx_or_final",
    "feature_drift_mean",
    "feature_drift_std",
    "prototype_drift_mean",
]

TEXT_FEATURE_FIELDS = [
    "step_idx",
    "task_idx",
    "encoder_side",
    "group_name",
    "layer_idx_or_final",
    "feature_drift_mean",
    "feature_drift_std",
    "prototype_drift_mean",
    "num_text_classes",
]

ATTENTION_FIELDS = [
    "step_idx",
    "task_idx",
    "encoder_side",
    "modality",
    "group_name",
    "layer_idx",
    "head_idx_or_mean",
    "attention_entropy",
    "attention_drift_js",
    "topk_overlap",
    "attention_distance",
]

HARD_NEGATIVE_FIELDS = [
    "step_idx",
    "task_idx",
    "group_name",
    "num_samples",
    "pos_cos_mean",
    "max_neg_cos_mean",
    "margin_mean",
    "hard_neg_is_new_ratio",
    "hard_neg_is_old_ratio",
]

GRAD_FIELDS = [
    "step_idx",
    "task_idx",
    "layer_name",
    "matrix_type",
    "grad_cos_old_current",
    "old_grad_norm",
    "current_grad_norm",
]


class SeqLoRAAnalyzer:
    def __init__(self, args, model, device, output_dir, class_names=None, text_features=None):
        self.args = args
        self.model = model
        self.device = device
        self.output_dir = ensure_dir(output_dir)
        self.raw_dir = ensure_dir(os.path.join(self.output_dir, "raw"))
        self.class_names = list(class_names) if class_names is not None else None
        self.text_features = text_features
        self.prev_lora_delta = None
        self.reference_features = None
        self.reference_prototypes = None
        self.reference_text_features = None
        self.reference_gap_vec = None
        self.reference_attention = None
        self._write_schema_metadata()

    def _write_schema_metadata(self):
        write_analysis_schema_metadata(
            self.output_dir,
            output_files={
                "summary.csv": SUMMARY_FIELDS,
                "lora_stats.csv": LORA_FIELDS,
                "clip_alignment.csv": CLIP_ALIGNMENT_FIELDS,
                "feature_drift.csv": FEATURE_FIELDS,
                "text_feature_drift.csv": TEXT_FEATURE_FIELDS,
                "attention_stats.csv": ATTENTION_FIELDS,
                "hard_negative.csv": HARD_NEGATIVE_FIELDS,
                "grad_conflict.csv": GRAD_FIELDS,
            },
            extra={
                "analysis_component": "SeqLoRAAnalyzer",
                "effective_context_drift": "pending",
                "not_available_reason": "value_vectors_not_recorded",
            },
        )

    def run(
        self,
        step_idx,
        task_idx,
        dataloader,
        old_dataloader=None,
        current_dataloader=None,
        metrics=None,
        seen_class_ids=None,
        current_class_ids=None,
    ):
        interval = int(getattr(self.args, "analysis_interval", 1))
        if interval > 1 and (int(step_idx) + 1) % interval != 0:
            return

        was_training = self.model.training
        summary = {key: float("nan") for key in SUMMARY_FIELDS}
        summary.update({
            "step_idx": int(step_idx),
            "task_idx": int(task_idx),
        })
        metrics = metrics or {}
        summary["old_acc_if_available"] = metrics.get("old_acc_if_available", float("nan"))
        summary["new_acc_if_available"] = metrics.get("new_acc_if_available", float("nan"))

        tag = f"step_{int(step_idx):03d}_task_{int(task_idx):03d}"
        try:
            self.model.eval()
            if bool(getattr(self.args, "analysis_record_lora", True)):
                self._run_lora(tag, summary, step_idx, task_idx)
            if bool(getattr(self.args, "analysis_record_clip_alignment", True)):
                self._run_clip_alignment(tag, summary, step_idx, task_idx, dataloader, seen_class_ids, current_class_ids)
            if bool(getattr(self.args, "analysis_record_feature_drift", True)):
                self._run_feature_drift(tag, summary, step_idx, task_idx, dataloader, seen_class_ids, current_class_ids)
                self._run_text_feature_drift(tag, summary, step_idx, task_idx, seen_class_ids, current_class_ids)
            if bool(getattr(self.args, "analysis_record_attention", True)):
                self._run_attention(tag, summary, step_idx, task_idx, dataloader, seen_class_ids, current_class_ids)
            if bool(getattr(self.args, "analysis_record_grad_conflict", False)):
                self._run_grad_conflict(tag, summary, step_idx, task_idx, old_dataloader, current_dataloader)
        finally:
            self.model.train(was_training)

        append_csv(os.path.join(self.output_dir, "summary.csv"), summary, fieldnames=SUMMARY_FIELDS)

    def _max_batches(self) -> int:
        return int(getattr(self.args, "analysis_max_batches", 4))

    def _enable_groupwise(self) -> bool:
        return bool(
            getattr(
                self.args,
                "enable_groupwise_analysis",
                getattr(self.args, "analysis_groupwise", False),
            )
        )

    def _with_step_task(self, rows, step_idx, task_idx):
        out = []
        for row in rows:
            merged = {"step_idx": int(step_idx), "task_idx": int(task_idx)}
            merged.update(row)
            out.append(merged)
        return out

    def _label_to_text_index(self):
        names = None
        if hasattr(self.model, "all_class_names"):
            names = list(getattr(self.model, "all_class_names"))
        elif hasattr(self.model, "current_class_names"):
            names = list(getattr(self.model, "current_class_names"))
        if names is None or self.class_names is None:
            return None
        global_by_name = {name: idx for idx, name in enumerate(self.class_names)}
        mapping = {}
        for text_idx, name in enumerate(names):
            if name in global_by_name:
                mapping[int(global_by_name[name])] = int(text_idx)
        return mapping or None

    def _current_text_features(self):
        if self.text_features is not None:
            return F.normalize(self.text_features.detach().float(), dim=-1)
        core = getattr(self.model, "model", getattr(self.model, "clip_model", self.model))
        with torch.no_grad():
            if hasattr(self.model, "_text_features") and getattr(self.model, "_text_features") is not None:
                text = getattr(self.model, "_text_features")
            elif hasattr(self.model, "all_text_tokens") and hasattr(core, "encode_text"):
                text = core.encode_text(getattr(self.model, "all_text_tokens").to(self.device))
            elif hasattr(self.model, "_encode_text_features"):
                text, _ = self.model._encode_text_features(train=False)
            elif hasattr(self.model, "cur_text_features"):
                text = self.model.cur_text_features()
            elif hasattr(self.model, "text_tokens") and hasattr(core, "encode_text"):
                text = core.encode_text(getattr(self.model, "text_tokens").to(self.device))
            else:
                return None
        return F.normalize(text.detach().float(), dim=-1)

    def _current_text_tokens(self):
        core = getattr(self.model, "model", getattr(self.model, "clip_model", self.model))
        for attr in ("all_text_tokens", "text_tokens"):
            if hasattr(self.model, attr):
                tokens = getattr(self.model, attr)
                if torch.is_tensor(tokens):
                    return tokens.to(self.device)
            if hasattr(core, attr):
                tokens = getattr(core, attr)
                if torch.is_tensor(tokens):
                    return tokens.to(self.device)
        if self.class_names is None:
            return None
        try:
            import clip

            template = str(getattr(self.args, "prompt_template", "a photo of a {}."))
            return clip.tokenize([template.format(name) for name in self.class_names]).to(self.device)
        except Exception as exc:
            warn_once(f"text token construction failed: {exc}")
            return None

    def _current_text_class_ids(self):
        names = None
        if hasattr(self.model, "all_class_names"):
            names = list(getattr(self.model, "all_class_names"))
        elif hasattr(self.model, "current_class_names"):
            names = list(getattr(self.model, "current_class_names"))
        if names is None:
            text_features = self._current_text_features()
            if text_features is None:
                return None
            return list(range(int(text_features.shape[0])))
        if self.class_names is None:
            return list(range(len(names)))
        global_by_name = {name: idx for idx, name in enumerate(self.class_names)}
        return [int(global_by_name.get(name, idx)) for idx, name in enumerate(names)]

    def _first_batch(self, dataloader):
        if dataloader is None:
            return None
        for _, batch in limit_batches(dataloader, 1):
            return batch
        return None

    def _run_lora(self, tag, summary, step_idx, task_idx):
        try:
            rows, current_delta = collect_lora_stats(self.model, prev_delta_dict=self.prev_lora_delta)
            self.prev_lora_delta = current_delta
            rows = self._with_step_task(rows, step_idx, task_idx)
            save_json(os.path.join(self.raw_dir, f"{tag}_lora_stats.json"), rows)
            append_rows_csv(os.path.join(self.output_dir, "lora_stats.csv"), rows, fieldnames=LORA_FIELDS)
            if rows:
                norms = torch.tensor([float(r["delta_fro_norm"]) for r in rows], dtype=torch.float32)
                ranks = torch.tensor([float(r["effective_rank"]) for r in rows], dtype=torch.float32)
                summary["mean_lora_norm"] = float(norms.mean().item())
                summary["mean_lora_effective_rank"] = float(ranks.mean().item())
            else:
                warn_once("no LoRA layers found; lora_stats skipped")
        except Exception as exc:
            warn_once(f"lora_stats failed: {exc}")

    def _run_clip_alignment(self, tag, summary, step_idx, task_idx, dataloader, seen_class_ids=None, current_class_ids=None):
        try:
            text_features = self._current_text_features()
            if text_features is None:
                warn_once("text features unavailable; clip_alignment skipped")
                return
            rows, gap_vec, clip_summary, hard_rows = compute_clip_alignment_stats(
                self.model,
                dataloader,
                text_features=text_features,
                device=self.device,
                max_batches=self._max_batches(),
                label_to_text_index=self._label_to_text_index(),
                reference_gap_vec=self.reference_gap_vec,
                enable_groupwise=self._enable_groupwise(),
                seen_class_ids=seen_class_ids,
                current_class_ids=current_class_ids,
                return_hard_negative=True,
            )
            if self.reference_gap_vec is None:
                self.reference_gap_vec = gap_vec
            rows = self._with_step_task(rows, step_idx, task_idx)
            hard_rows = self._with_step_task(hard_rows, step_idx, task_idx)
            save_json(os.path.join(self.raw_dir, f"{tag}_clip_alignment.json"), rows)
            append_rows_csv(os.path.join(self.output_dir, "clip_alignment.csv"), rows, fieldnames=CLIP_ALIGNMENT_FIELDS)
            save_json(os.path.join(self.raw_dir, f"{tag}_hard_negative.json"), hard_rows)
            append_rows_csv(os.path.join(self.output_dir, "hard_negative.csv"), hard_rows, fieldnames=HARD_NEGATIVE_FIELDS)
            summary.update({k: v for k, v in clip_summary.items() if k in summary})
            summary["num_samples"] = max(summary.get("num_samples", 0) if not isinstance(summary.get("num_samples"), float) else 0, int(clip_summary.get("num_alignment_samples", 0)))
        except Exception as exc:
            warn_once(f"clip_alignment failed: {exc}")

    def _run_feature_drift(self, tag, summary, step_idx, task_idx, dataloader, seen_class_ids=None, current_class_ids=None):
        try:
            rows, current_features, current_prototypes, feature_summary = compute_feature_drift_stats(
                self.model,
                dataloader,
                reference_features=self.reference_features,
                reference_prototypes=self.reference_prototypes,
                device=self.device,
                max_batches=self._max_batches(),
                enable_groupwise=self._enable_groupwise(),
                seen_class_ids=seen_class_ids,
                current_class_ids=current_class_ids,
                fill_missing_reference_with_current=self._enable_groupwise(),
            )
            if self.reference_features is None:
                self.reference_features = dict(current_features)
            else:
                for sample_id, feat in current_features.items():
                    self.reference_features.setdefault(sample_id, feat)
            if self.reference_prototypes is None:
                self.reference_prototypes = dict(current_prototypes)
            else:
                for class_id, proto in current_prototypes.items():
                    self.reference_prototypes.setdefault(class_id, proto)
            rows = self._with_step_task(rows, step_idx, task_idx)
            save_json(os.path.join(self.raw_dir, f"{tag}_feature_stats.json"), rows)
            append_rows_csv(os.path.join(self.output_dir, "feature_drift.csv"), rows, fieldnames=FEATURE_FIELDS)
            summary["mean_feature_drift"] = feature_summary.get("mean_feature_drift", float("nan"))
            summary["num_samples"] = max(summary.get("num_samples", 0) if not isinstance(summary.get("num_samples"), float) else 0, int(feature_summary.get("num_samples", 0)))
        except Exception as exc:
            warn_once(f"feature_drift failed: {exc}")

    def _run_text_feature_drift(self, tag, summary, step_idx, task_idx, seen_class_ids=None, current_class_ids=None):
        try:
            text_features = self._current_text_features()
            text_class_ids = self._current_text_class_ids()
            if text_features is None or text_class_ids is None:
                warn_once("text features unavailable; text_feature_drift skipped")
                return
            rows, current_text_features, text_summary = compute_text_feature_drift_stats(
                text_features,
                text_class_ids,
                reference_text_features=self.reference_text_features,
                enable_groupwise=self._enable_groupwise(),
                seen_class_ids=seen_class_ids,
                current_class_ids=current_class_ids,
                text_tokens=self._current_text_tokens(),
            )
            if self.reference_text_features is None:
                self.reference_text_features = dict(current_text_features)
            else:
                for class_id, feat in current_text_features.items():
                    self.reference_text_features.setdefault(class_id, feat)
            rows = self._with_step_task(rows, step_idx, task_idx)
            save_json(os.path.join(self.raw_dir, f"{tag}_text_feature_stats.json"), rows)
            append_rows_csv(os.path.join(self.output_dir, "text_feature_drift.csv"), rows, fieldnames=TEXT_FEATURE_FIELDS)
            summary["mean_text_feature_drift"] = text_summary.get("mean_text_feature_drift", float("nan"))
            summary["num_samples"] = max(
                summary.get("num_samples", 0) if not isinstance(summary.get("num_samples"), float) else 0,
                int(text_summary.get("num_text_classes", 0)),
            )
        except Exception as exc:
            warn_once(f"text_feature_drift failed: {exc}")

    def _run_attention(self, tag, summary, step_idx, task_idx, dataloader, seen_class_ids=None, current_class_ids=None):
        try:
            rows, current_attention, attention_summary = compute_attention_stats(
                self.model,
                dataloader,
                reference_attention=self.reference_attention,
                device=self.device,
                max_batches=self._max_batches(),
                enable_groupwise=self._enable_groupwise(),
                seen_class_ids=seen_class_ids,
                current_class_ids=current_class_ids,
            )
            if self.reference_attention is None:
                self.reference_attention = current_attention
            else:
                self.reference_attention.update({
                    key: value
                    for key, value in current_attention.items()
                    if isinstance(key, tuple) and len(key) >= 3 and str(key[0]) == "__class__" and key not in self.reference_attention
                })
            rows = self._with_step_task(rows, step_idx, task_idx)
            save_json(os.path.join(self.raw_dir, f"{tag}_attention_stats.json"), rows)
            append_rows_csv(os.path.join(self.output_dir, "attention_stats.csv"), rows, fieldnames=ATTENTION_FIELDS)
            summary["mean_attention_drift"] = attention_summary.get("mean_attention_drift", float("nan"))
        except Exception as exc:
            warn_once(f"attention_stats skipped/failed: {exc}")

    def _run_grad_conflict(self, tag, summary, step_idx, task_idx, old_dataloader, current_dataloader):
        try:
            old_batch = self._first_batch(old_dataloader)
            current_batch = self._first_batch(current_dataloader)
            rows = compute_lora_gradient_conflict(
                self.model,
                old_batch=old_batch,
                current_batch=current_batch,
                device=self.device,
                label_to_logit_index=self._label_to_text_index(),
            )
            rows = self._with_step_task(rows, step_idx, task_idx)
            save_json(os.path.join(self.raw_dir, f"{tag}_grad_conflict.json"), rows)
            append_rows_csv(os.path.join(self.output_dir, "grad_conflict.csv"), rows, fieldnames=GRAD_FIELDS)
            if rows:
                summary["mean_grad_cosine"] = float(rows[0].get("grad_cos_old_current", float("nan")))
        except Exception as exc:
            warn_once(f"grad_conflict skipped/failed: {exc}")

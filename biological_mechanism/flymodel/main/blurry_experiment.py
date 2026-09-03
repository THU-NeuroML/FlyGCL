"""Inherited-expert training on conserved boundary-blurred streams."""
from __future__ import annotations
import random
from typing import Any
import numpy as np
import torch
from torch.nn import functional as F
from .audit import tensor_sha256
from .blurry_metrics import exposure_metrics
from .config import Config, READOUTS
from .experiment import inherit_expert, integrated_logits, online_route
from .model import ExpertBank, FixedEncoder

CONDITIONS = ("single_head", "shared_el", "inherited_moe_mid", "inherited_moe_el")


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def evaluate_blurry(encoder: FixedEncoder, bank: ExpertBank, dataset: dict[str, Any], routed: bool, artifact: dict[str, torch.Tensor], device: torch.device, position: int, stage: int | None, active: int, exposure_counts: np.ndarray, cfg: Config) -> dict[str, Any]:
    encoder.eval(); bank.eval()
    predictions = {name: [] for name in READOUTS}; routes = []
    before = (tensor_sha256(artifact["sums"]), tensor_sha256(artifact["counts"]))
    for start in range(0, len(dataset["test_samples"]), cfg.evaluation_batch_size):
        x = torch.from_numpy(np.array(dataset["test_samples"][start:start + cfg.evaluation_batch_size], dtype=np.float32, copy=True)).to(device)
        kc = encoder(x)
        if routed and active:
            route = online_route(kc, artifact["sums"], artifact["counts"], active)
        elif routed:
            route = torch.full((len(x),), -1, dtype=torch.long, device=device)
        else:
            route = torch.zeros(len(x), dtype=torch.long, device=device)
        if routed and active:
            merged = {name: torch.empty((len(x), cfg.n_classes), device=device) for name in READOUTS}
            for expert in range(active):
                mask = route == expert
                if mask.any():
                    values = integrated_logits(bank.expert_logits(kc[mask], expert))
                    for name in READOUTS: merged[name][mask] = values[name]
            for name in READOUTS: predictions[name].append(merged[name].argmax(1).cpu())
        elif routed:
            values = [integrated_logits(bank.expert_logits(kc, expert)) for expert in range(bank.n_experts)]
            for name in READOUTS: predictions[name].append(torch.stack([item[name] for item in values], 1).mean(1).argmax(1).cpu())
        else:
            values = integrated_logits(bank.expert_logits(kc, 0))
            for name in READOUTS: predictions[name].append(values[name].argmax(1).cpu())
        routes.append(route.cpu())
    if before != (tensor_sha256(artifact["sums"]), tensor_sha256(artifact["counts"])):
        raise RuntimeError("evaluation mutated router")

    predictions = {name: torch.cat(parts) for name, parts in predictions.items()}
    route = torch.cat(routes)
    labels = torch.as_tensor(np.asarray(dataset["test_labels"], dtype=np.int64))
    regions = torch.as_tensor(np.asarray(dataset["test_regions"], dtype=np.int64))
    class_groups = torch.as_tensor(np.asarray(dataset["class_groups"], dtype=np.int64))
    class_counts = torch.bincount(labels, minlength=cfg.n_classes)
    routed_mask = route >= 0
    route_accuracy = float((route[routed_mask] == regions[routed_mask]).float().mean()) if routed and active else None
    readouts = {}
    for name, prediction in predictions.items():
        class_correct = torch.bincount(labels[prediction == labels], minlength=cfg.n_classes)
        region_accuracy = [float((prediction[regions == region] == labels[regions == region]).float().mean()) for region in range(cfg.n_regions)]
        detail = {"overall_accuracy": float((prediction == labels).float().mean()), "region_accuracy": region_accuracy, "class_correct": class_correct.tolist(), "class_test_counts": class_counts.tolist()}
        if routed and active:
            wrong = routed_mask & (route != regions)
            detail["wrong_route_classification_accuracy"] = float((prediction[wrong] == labels[wrong]).float().mean()) if wrong.any() else None
            detail["outside_selected_expert_prediction_rate"] = float((class_groups[prediction[routed_mask]] != route[routed_mask]).float().mean())
        else:
            detail["wrong_route_classification_accuracy"] = None; detail["outside_selected_expert_prediction_rate"] = None
        readouts[name] = detail
    primary = readouts["softmax_mean"]
    return {
        "position": position, "stage": stage, "exposure_counts": np.asarray(exposure_counts, dtype=np.int64).tolist(),
        "overall_accuracy": primary["overall_accuracy"], "region_accuracy": primary["region_accuracy"],
        "class_correct": primary["class_correct"], "class_test_counts": primary["class_test_counts"], "readouts": readouts,
        "routing": {"home_region_accuracy": route_accuracy, "route_counts": torch.bincount(route[routed_mask], minlength=cfg.n_regions).tolist() if routed and active else None, "active_prototypes": active if routed else None},
        "audit": {"sample_count": len(labels), "read_only": True, "online_state_before": before},
    }


def train_blurry(dataset: dict[str, Any], condition: str, eta: float, gamma: float, seed: int, cfg: Config, device_name: str = "cuda") -> dict[str, Any]:
    if condition not in CONDITIONS or eta != 1e-3 or gamma != 10:
        raise ValueError("invalid fixed boundary-blur protocol")
    set_seed(seed); device = torch.device(device_name)
    routed = condition.startswith("inherited_"); temporal = condition.endswith("_el")
    n_experts = cfg.n_regions if routed else 1; n_timescales = 3 if temporal else 1
    rates = [gamma * eta, eta, eta / gamma] if temporal else [eta]
    encoder = FixedEncoder(dataset["orn_pn"], dataset["pn_kc"], cfg, device_name)
    bank = ExpertBank(seed, n_experts, n_timescales, cfg, device_name)
    groups = [{"params": bank.head(expert, scale).parameters(), "lr": rate} for expert in range(n_experts) for scale, rate in enumerate(rates)]
    optimizer = torch.optim.Adam(groups)
    initial_hashes = [[tensor_sha256(bank.head(expert, scale).weight) for scale in range(n_timescales)] for expert in range(n_experts)]
    stream = np.asarray(dataset["stream_indices"], dtype=np.int64); labels = np.asarray(dataset["train_labels"], dtype=np.int64)
    boundaries = dataset["blur_metadata"]["stage_boundaries"]
    targets = set(range(2_000, len(stream) + 1, 2_000))
    artifact = {"sums": torch.zeros((cfg.n_regions, cfg.n_kc), dtype=torch.float64, device=device), "counts": torch.zeros(cfg.n_regions, dtype=torch.long, device=device)}
    exposure_counts = np.zeros(cfg.n_classes, dtype=np.int64)
    records = [evaluate_blurry(encoder, bank, dataset, routed, artifact, device, 0, None, 0, exposure_counts, cfg)]
    position = 0; initialized_stage = -1; running: list[float] = []; losses = []; inheritance = []; stage_snapshots = []
    noise = torch.Generator(device=device).manual_seed(seed + 150_000)
    while position < len(stream):
        event = min((value for value in (*targets, *boundaries) if value > position), default=len(stream))
        end = min(position + cfg.batch_size, event)
        stage = next(index for index, boundary in enumerate(boundaries) if position < boundary)
        if stage != initialized_stage:
            if routed and stage > 0: inheritance.append(inherit_expert(bank, stage, optimizer))
            initialized_stage = stage
        indices = stream[position:end]
        x = torch.from_numpy(np.array(dataset["train_samples"][indices], dtype=np.float32, copy=True)).to(device)
        y = torch.as_tensor(labels[indices], dtype=torch.long, device=device)
        expert = stage if routed else 0
        encoder.train(); bank.train(); optimizer.zero_grad(set_to_none=True)
        kc = encoder(x, cfg.train_noise_sigma, noise); logits = bank.expert_logits(kc, expert)
        loss = torch.stack([F.cross_entropy(logits[:, scale], y) for scale in range(n_timescales)]).mean(); loss.backward()
        for other in range(n_experts):
            if other != expert and any(bank.head(other, scale).weight.grad is not None for scale in range(n_timescales)):
                raise RuntimeError("inactive expert received gradient")
        optimizer.step(); running.append(float(loss.detach()))
        if routed:
            artifact["sums"][stage] += kc.detach().double().sum(0); artifact["counts"][stage] += len(kc)
        exposure_counts += np.bincount(labels[indices], minlength=cfg.n_classes)
        position = end
        if position in targets:
            current = next(index for index, boundary in enumerate(boundaries) if position <= boundary)
            records.append(evaluate_blurry(encoder, bank, dataset, routed, artifact, device, position, current, current + 1 if routed else 0, exposure_counts, cfg))
            losses.append({"position": position, "stage": current, "mean_loss": float(np.mean(running))}); running = []
        if position in boundaries:
            snapshot = [[tensor_sha256(bank.head(expert_index, scale).weight) for scale in range(n_timescales)] for expert_index in range(n_experts)]
            if routed and stage_snapshots:
                for old_expert in range(stage):
                    if snapshot[old_expert] != stage_snapshots[-1]["hashes"][old_expert]: raise RuntimeError("previous expert changed after its stage")
            stage_snapshots.append({"stage": stage, "hashes": snapshot, "optimizer_state_experts": [all(bool(optimizer.state.get(bank.head(expert_index, scale).weight)) for scale in range(n_timescales)) for expert_index in range(n_experts)]})
    if len(records) != 26 or records[-1]["position"] != 50_000 or exposure_counts.sum() != 50_000:
        raise RuntimeError("boundary-blur evaluation schedule mismatch")
    summaries = {}
    for name in READOUTS:
        selected = [{**record, "class_correct": record["readouts"][name]["class_correct"], "class_test_counts": record["readouts"][name]["class_test_counts"]} for record in records]
        summaries[name] = exposure_metrics(selected, len(stream), cfg.n_classes)
    return {
        "schema_version": 1, "experiment": "olfactory_boundary_blur", "status": "complete", "condition": condition, "seed": seed,
        "training": {"eta": eta, "gamma": gamma, "rates": rates, "device": device_name, "optimizer": "Adam", "primary_integration": "softmax_mean", "recorded_integrations": list(READOUTS)},
        "stream": {"length": len(stream), **dataset["blur_metadata"]},
        "inheritance": {"enabled": routed, "source": "arithmetic mean of all previously trained experts", "parameter_values_only": True, "optimizer_state_copied": False, "events": inheritance},
        "evaluation": {"expected_count": 26, "records": records}, "readout_summaries": summaries, "summary": summaries["softmax_mean"],
        "loss_trajectory": losses, "initial_head_hashes": initial_hashes, "stage_parameter_snapshots": stage_snapshots,
        "routing_diagnostics": {"prototype_counts": artifact["counts"].tolist()},
        "audits": {"stream_conservation": True, "evaluation_read_only": True, "online_causal": True, "inactive_gradient_none": True, "old_experts_immutable": routed, "inheritance_mean_exact": routed},
    }

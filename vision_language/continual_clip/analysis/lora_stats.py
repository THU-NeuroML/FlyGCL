import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


LORA_A_SUFFIXES = ("lora_A", "lora_a", "lora_A.default", "lora_a.default", "A")
LORA_B_SUFFIXES = ("lora_B", "lora_b", "lora_B.default", "lora_b.default", "B")


def _parse_block_idx(name: str) -> int:
    patterns = [
        r"lora_bank\.experts\.\d+\.(\d+)",
        r"blocks_by_expert\.\d+\.(\d+)",
        r"(?:resblocks|blocks|layers)\.(\d+)",
        r"(?:visual\.transformer\.resblocks|transformer\.resblocks)\.(\d+)",
        r"block[_\.]?(\d+)",
        r"layer[_\.]?(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return -1


def _matrix_type(name: str) -> str:
    lower = name.lower()
    if re.search(r"(^|[._])k($|_proj|ey|[._])|k_proj|key|k_proj_weight", lower):
        return "k"
    if re.search(r"(^|[._])v($|_proj|alue|[._])|v_proj|value|v_proj_weight", lower):
        return "v"
    if re.search(r"(^|[._])q($|_proj|uery|[._])|q_proj|query|q_proj_weight", lower):
        return "q"
    if "in_proj" in lower:
        return "qkv"
    if "out_proj" in lower:
        return "other"
    return "unknown"


def _encoder_side(name: str) -> str:
    lower = name.lower()
    if lower.startswith("visual.") or ".visual." in lower:
        return "visual"
    if lower.startswith("model.visual.") or lower.startswith("clip_model.visual."):
        return "visual"
    if "lora_bank" in lower or "blocks_by_expert" in lower:
        return "visual"
    return "text"


def _strip_lora_suffix(param_name: str, suffixes) -> Optional[str]:
    for suffix in suffixes:
        if param_name == suffix:
            return ""
        tail = f"_{suffix}"
        if param_name.endswith(tail):
            return param_name[: -len(tail)]
        dot_tail = f".{suffix}"
        if param_name.endswith(dot_tail):
            return param_name[: -len(dot_tail)]
    return None


def _candidate_base_weight(module, prefix: str):
    candidates = []
    clean = prefix.rstrip("._")
    if clean:
        candidates.append(clean)
    if clean.endswith("_weight"):
        candidates.append(clean)
    elif clean:
        candidates.append(f"{clean}_weight")
    candidates.append("weight")
    for attr in candidates:
        if hasattr(module, attr):
            weight = getattr(module, attr)
            if torch.is_tensor(weight):
                return weight
    return None


def _safe_svdvals(matrix: torch.Tensor) -> torch.Tensor:
    matrix = matrix.detach().float()
    try:
        return torch.linalg.svdvals(matrix)
    except Exception:
        return torch.empty(0)


def _effective_rank(singular_values: torch.Tensor) -> float:
    if singular_values.numel() == 0:
        return float("nan")
    s = singular_values.float()
    total = torch.sum(s)
    if float(total.item()) <= 0.0:
        return 0.0
    p = s / total
    entropy = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy).item())


def _pair_lora_params(module) -> Dict[str, Dict[str, torch.Tensor]]:
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    for pname, param in module.named_parameters(recurse=False):
        a_prefix = _strip_lora_suffix(pname, LORA_A_SUFFIXES)
        if a_prefix is not None:
            pairs.setdefault(a_prefix, {})["A"] = param
            continue
        b_prefix = _strip_lora_suffix(pname, LORA_B_SUFFIXES)
        if b_prefix is not None:
            pairs.setdefault(b_prefix, {})["B"] = param
    for cname, child in module.named_children():
        if not hasattr(child, "weight") or not torch.is_tensor(getattr(child, "weight")):
            continue
        lower = cname.lower()
        if lower.endswith("_a") or lower.endswith(".a") or lower in {"a", "lora_a"}:
            prefix = cname.rsplit("_", 1)[0] if "_" in cname else ""
            pairs.setdefault(prefix, {})["A"] = child.weight
        elif lower.endswith("_b") or lower.endswith(".b") or lower in {"b", "lora_b"}:
            prefix = cname.rsplit("_", 1)[0] if "_" in cname else ""
            pairs.setdefault(prefix, {})["B"] = child.weight
    return pairs


def collect_lora_stats(model, prev_delta_dict: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[List[Dict], Dict[str, torch.Tensor]]:
    stats: List[Dict] = []
    current_delta_dict: Dict[str, torch.Tensor] = {}
    prev_delta_dict = prev_delta_dict or {}

    with torch.no_grad():
        for module_name, module in model.named_modules():
            pairs = _pair_lora_params(module)
            if not pairs:
                continue
            scaling = float(getattr(module, "scaling", 1.0))
            block_idx = _parse_block_idx(module_name)
            for prefix, pair in pairs.items():
                if "A" not in pair or "B" not in pair:
                    continue
                A = pair["A"]
                B = pair["B"]
                if A is None or B is None or A.ndim != 2 or B.ndim != 2:
                    continue
                layer_name = module_name if prefix == "" else f"{module_name}.{prefix}"
                try:
                    delta = scaling * (B.detach().float() @ A.detach().float())
                except Exception:
                    continue
                delta_cpu = delta.detach().cpu()
                current_delta_dict[layer_name] = delta_cpu
                svals = _safe_svdvals(delta)
                fro_norm = float(torch.linalg.norm(delta.detach().float(), ord="fro").item())
                spectral_norm = float(svals.max().item()) if svals.numel() else float("nan")
                eff_rank = _effective_rank(svals)

                base_weight = _candidate_base_weight(module, prefix)
                if base_weight is not None and tuple(base_weight.shape) == tuple(delta.shape):
                    denom = float(torch.linalg.norm(base_weight.detach().float(), ord="fro").item())
                    norm_ratio = fro_norm / denom if denom > 0 else float("nan")
                else:
                    norm_ratio = float("nan")

                prev = prev_delta_dict.get(layer_name)
                if prev is not None and tuple(prev.shape) == tuple(delta_cpu.shape):
                    delta_cos_prev = float(F.cosine_similarity(delta_cpu.flatten(), prev.flatten(), dim=0).item())
                else:
                    delta_cos_prev = float("nan")

                stats.append({
                    "layer_name": layer_name,
                    "module_name": module_name,
                    "encoder_side": _encoder_side(layer_name),
                    "layer_idx": block_idx,
                    "block_idx": block_idx,
                    "matrix_type": _matrix_type(layer_name),
                    "delta_fro_norm": fro_norm,
                    "delta_spectral_norm": spectral_norm,
                    "effective_rank": eff_rank,
                    "norm_ratio": norm_ratio,
                    "delta_cos_prev": delta_cos_prev,
                })

    return stats, current_delta_dict

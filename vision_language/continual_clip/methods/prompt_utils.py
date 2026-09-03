def parse_prompt_modalities(cfg):
    raw = getattr(cfg, "prompt_modalities", getattr(cfg, "prompt_modality", "vision"))
    if raw is None:
        raw = "vision"
    if isinstance(raw, (list, tuple, set)):
        parts = [str(x).strip().lower() for x in raw]
    else:
        text = str(raw).strip().lower().replace(",", "+").replace("|", "+")
        if text in {"both", "all", "vision_text", "text_vision"}:
            text = "vision+text"
        parts = [p.strip() for p in text.split("+") if p.strip()]
    modalities = set()
    for part in parts:
        if part in {"v", "visual", "image"}:
            modalities.add("vision")
        elif part in {"t", "txt", "language"}:
            modalities.add("text")
        elif part in {"vision", "text"}:
            modalities.add(part)
    if not modalities:
        modalities.add("vision")
    return modalities


def resolve_prompt_layers(cfg, attr_name, num_layers, default_layers=None):
    inject_all = bool(getattr(cfg, "prompt_inject_all_layers", True))
    if inject_all:
        return list(range(int(num_layers)))
    raw = getattr(cfg, attr_name, None)
    if raw is None:
        raw = default_layers if default_layers is not None else list(range(int(num_layers)))
    return [int(x) for x in list(raw)]

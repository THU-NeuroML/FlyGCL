__all__ = ["SeqLoRAAnalyzer"]


def __getattr__(name):
    if name == "SeqLoRAAnalyzer":
        from .seq_lora_analyzer import SeqLoRAAnalyzer

        return SeqLoRAAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

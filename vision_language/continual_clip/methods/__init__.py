from .lora_method import LoRAMethod
from .coda_method import CODAMethod
from .proof_method import PROOFMethod
from .official_prompt_method import L2POfficialMethod, DualPromptOfficialMethod
from .lwf_method import LwFMethod
from .ewc_method import EWCMethod
from .clip_ft_method import CLIPFullFineTuneMethod
from .fly_method import FlyMethod
from .fly_clip_method import FlyCLIPMethod
from .fly_clip_text_linear_ema_method import FlyCLIPTextLinearEMAMethod
from .fly_clip_text_feature_ema_method import (
    FlyCLIPMethod as FlyCLIPTextFeatureEMAMethod,
)
from .fly_clip_text_feature_linear_ema_method import (
    FlyCLIPTextLinearEMAMethod as FlyCLIPTextFeatureLinearEMAMethod,
)
from .fly_gaploss_method import FlyGapLossMethod
from .fly_vga_method import FlyVGAMethod
from .misa_method import MISAMethod, MISAL2PMethod

METHOD_REGISTRY = {
    "lora": LoRAMethod,
    "l2p": L2POfficialMethod,
    "l2p_official": L2POfficialMethod,
    "dualprompt": DualPromptOfficialMethod,
    "dualprompt_official": DualPromptOfficialMethod,
    "misa": MISAMethod,
    "misa_l2p": MISAL2PMethod,
    "coda": CODAMethod,
    "proof": PROOFMethod,
    "lwf": LwFMethod,
    "clip_ft": CLIPFullFineTuneMethod,
    "full_ft": CLIPFullFineTuneMethod,
    "finetune": CLIPFullFineTuneMethod,
    # Canonical name for the current online/compressed CLIP-LoRA EWC variant.
    "online_ewc": EWCMethod,
    # Online EWC restricted to attention K/V rows via clip_ft_trainable_scope.
    "ewc_kv": EWCMethod,
    # Backward-compatible alias.
    "ewc": EWCMethod,
    "fly": FlyMethod,
    "fly_clip": FlyCLIPMethod,
    "fly_clip_text_linear_ema": FlyCLIPTextLinearEMAMethod,
    "fly_clip_text_feature_ema": FlyCLIPTextFeatureEMAMethod,
    "fly_clip_text_feature_linear_ema": FlyCLIPTextFeatureLinearEMAMethod,
    "fly_gaploss": FlyGapLossMethod,
    "fly_vga": FlyVGAMethod,
}

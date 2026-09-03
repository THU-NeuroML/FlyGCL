from __future__ import annotations
from dataclasses import dataclass, field
import copy
import draccus
from typing import Any, List, Optional, Union, Dict

import draccus.parsers

from peft.config import PeftConfig
from peft.utils.peft_types import PeftType

ModuleSelector = Union[str, "re.Pattern[str]"]

@dataclass
class FuncAdapterConfig:
    """
    This is the sub-configuration class to store the configuration of a [`CLAREModel`].

    Args:
        hidden_dim (`int`):
            The dimension of the hidden feature of the bottleneck adapter.
        use_lora (`bool`):
            whether to use lora on functional adapter or not
        lora_rank (`int`):
            Lora attention dimension (the "rank").
        lora_alpha (`int`):
            The alpha parameter for Lora scaling.
    """
    hidden_dim: int = field(default=0)
    use_lora: bool = field(default=False)
    lora_rank: int = field(default=32)
    lora_alpha: int = field(default=32)
    zero_init_output: bool = field(default=False)


@dataclass
class ResidualHeadEMAConfig:
    ratios: List[float] = field(default_factory=list)
    ensemble_method: str = field(default="mean")


@dataclass
class ResidualEnsembleGateConfig:
    hidden_dim: int = field(default=128)
    temperature: float = field(default=1.0)
    online_logit_bias: float = field(default=0.0)
    detach_residual_inputs: bool = field(default=True)



@dataclass
class DiscriminatorConfig(draccus.ChoiceRegistry):
    """
    This is the sub-configuration class to store the configuration of a [`CLAREModel`].
    
    Args:
        
        max_batches_tracked (`int`):
            How many batches will be tracked to calculate the statistic.
    """
    feature_dim: int = None
    batch_first: bool = True
    feature_fusion: bool = False
    num_tokens: int = None
    fused_feature_dim: int = None
    max_batches_tracked: int = 2000
    use_momentum: bool = True
    momentum: float = 0.1

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "autoencoder"

@dataclass
class CLAREModuleConfig:
    """Configuration for a specific target module pattern.
    
    Args:
        pattern (`str`):
            Regex pattern to match module names
        feature_dim (`int`):
            The dimension of the input feature
        out_feature_dim (`Optional[int]`):
            The dimension of the output feature. If None, defaults to feature_dim
        discriminator_cfg (`Optional[DiscriminatorConfig]`):
            Configuration for the discriminator
        batch_first (`bool`):
            Whether the input tensor has batch dimension first (B, T, D) vs (T, B, D)
        use_trainable_copy (`bool`):
            Whether to copy the module from base model as adapter
        add_zero_init_conv_layer (`bool`):
            Whether to add a zero-initialized conv layer
        func_adapter_cfg (`Optional[FuncAdapterConfig]`):
            Configuration for the functional adapter
    """
    pattern: str
    feature_dim: int
    out_feature_dim: Optional[int] = None
    discriminator_cfg: Optional[DiscriminatorConfig] = None
    batch_first: bool = True
    use_trainable_copy: bool = False
    add_zero_init_conv_layer: bool = False
    func_adapter_cfg: Optional[FuncAdapterConfig] = None
    routing_group: Optional[str] = None
    routing_role: str = "independent"
    residual_head_ema_cfg: Optional[ResidualHeadEMAConfig] = None
    residual_ensemble_gate_cfg: Optional[ResidualEnsembleGateConfig] = None

@dataclass
class CLAREConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`CLAREModel`].

    Args:
        target_modules (`Union[List[CLAREModuleConfig], Dict[str, CLAREModuleConfig], str]`):
            The configuration for different module patterns. Can be specified in several ways:
            - A string (legacy mode): Single regex pattern with default config
            - A list of CLAREModuleConfig: Each config specifies its own pattern and settings
            - A dict mapping regex patterns to CLAREModuleConfig objects
            Legacy behavior is preserved when passing a single string or using default values.
        feature_dim (`Optional[int]`):
            Default dimension of the input feature. Used if not specified in module config.
            Given an input of shape (B, T, D), D is feature_dim
        out_feature_dim (`Optional[int]`):
            Default dimension of the output feature. Used if not specified in module config.
        discriminator_cfg (`Optional[DiscriminatorConfig]`):
            Default discriminator configuration. Used if not specified in module config.
        use_trainable_copy (`bool`):
            Default setting for whether to copy the module from base model as adapter.
        func_adapter_cfg (`Optional[FuncAdapterConfig]`):
            Default adapter configuration. Used if not specified in module config.
    """
    target_modules: Union[List[CLAREModuleConfig], Dict[str, CLAREModuleConfig], str] = \
        field(default=r"(?P<layer_name>.+)\.(?P<layer_id>\d+)(?:\.[^.]+)*\.mlp")
    feature_dim: Optional[int] = None
    out_feature_dim: Optional[int] = None
    # Default values for module configs when not specified
    batch_first: bool = True  # Default batch_first for new module configs
    discriminator_cfg: Optional[DiscriminatorConfig] = None
    use_trainable_copy: bool = False  # Default use_trainable_copy for new module configs
    add_zero_init_conv_layer: bool = False
    func_adapter_cfg: Optional[FuncAdapterConfig] = None
    num_learned_task: int = 0
    structure: Dict = field(default_factory=dict)
    routing_mode: str = "discriminator"
    rp_dim: int = 10000
    rp_ridge: float = 1e4
    rp_max_classes: int = 64
    collect_rp_stats_during_train: bool = True
    ws_subspace_k: int = 32
    ws_eps: float = 1e-6
    ws_gamma: float = 1.0
    ws_max_classes: int = 64
    collect_wsr_stats_during_train: bool = True
    l2p_pool_size: int = 10
    l2p_top_k: int = 1
    l2p_key_loss_weight: float = 0.1
    l2p_key_init: str = "normal"
    dual_num_general_adapters: int = 1
    dual_prototype_batches: int = 200
    dual_general_train: bool = True
    dual_prototype_momentum: Optional[float] = None
    dual_max_tasks: int = 64
    
    # Internal state to store processed module configs
    _module_configs: Dict[str, CLAREModuleConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        # Assign to a valid PEFT type so load/save works. It does not alter the tuner name.
        self.peft_type = PeftType.CLARE

        # Process default configurations first
        self.out_feature_dim = self.out_feature_dim or self.feature_dim
        if self.routing_mode not in {
            "discriminator",
            "rp_gate",
            "ws_router",
            "whitened_subspace",
            "l2p_adapter",
            "dualprompt_adapter",
        }:
            raise ValueError(
                f"Unsupported routing_mode: {self.routing_mode}. "
                "Expected one of {'discriminator', 'rp_gate', 'ws_router', 'whitened_subspace', "
                "'l2p_adapter', 'dualprompt_adapter'}."
            )
        if self.rp_max_classes <= 0:
            raise ValueError(f"rp_max_classes must be positive, got {self.rp_max_classes}")
        if self.ws_subspace_k <= 0:
            raise ValueError(f"ws_subspace_k must be positive, got {self.ws_subspace_k}")
        if self.ws_eps <= 0:
            raise ValueError(f"ws_eps must be positive, got {self.ws_eps}")
        if self.ws_gamma <= 0:
            raise ValueError(f"ws_gamma must be positive, got {self.ws_gamma}")
        if self.ws_max_classes <= 0:
            raise ValueError(f"ws_max_classes must be positive, got {self.ws_max_classes}")
        if isinstance(self.func_adapter_cfg, dict):
            self.func_adapter_cfg = FuncAdapterConfig(**self.func_adapter_cfg)
        if isinstance(self.discriminator_cfg, dict):
            discriminator_cfg = copy.deepcopy(self.discriminator_cfg)
            discriminator_type = discriminator_cfg.pop("type", DiscriminatorConfig.default_choice_name())
            self.discriminator_cfg = DiscriminatorConfig.get_choice_class(discriminator_type)(**discriminator_cfg)

        # Process module configurations using processed defaults
        self._process_module_configs()

    @staticmethod
    def _validate_module_config(config: CLAREModuleConfig) -> None:
        if config.routing_role not in {"independent", "router_owner", "router_follower"}:
            raise ValueError(
                f"Unsupported routing_role: {config.routing_role}. "
                "Expected one of {'independent', 'router_owner', 'router_follower'}."
            )
        if config.routing_role != "independent" and config.routing_group is None:
            raise ValueError(
                f"routing_role={config.routing_role} requires a non-empty routing_group for pattern {config.pattern}."
            )
        if config.residual_head_ema_cfg is None:
            if config.residual_ensemble_gate_cfg is not None:
                raise ValueError(
                    "residual_ensemble_gate_cfg requires residual_head_ema_cfg to be set "
                    f"for pattern {config.pattern}."
                )
            return

        ema_cfg = config.residual_head_ema_cfg
        if ema_cfg.ensemble_method not in {"mean"}:
            raise ValueError(
                f"Unsupported residual_head_ema_cfg.ensemble_method: {ema_cfg.ensemble_method}. "
                "Expected 'mean'."
            )
        if not ema_cfg.ratios:
            raise ValueError("residual_head_ema_cfg requires non-empty ratios.")
        for ratio in ema_cfg.ratios:
            if not (0.0 < ratio < 1.0):
                raise ValueError(f"residual_head_ema_cfg.ratios must lie in (0, 1), got {ratio}")
        if config.routing_role != "router_follower":
            raise ValueError(
                "residual_head_ema_cfg is only supported on router_follower modules "
                f"(got routing_role={config.routing_role} for pattern {config.pattern})."
            )
        if config.use_trainable_copy:
            raise ValueError(
                f"residual_head_ema_cfg requires FuncAdapter-based modules, but {config.pattern} uses trainable copy."
            )
        if config.func_adapter_cfg is None:
            raise ValueError(
                f"residual_head_ema_cfg requires func_adapter_cfg to be set for pattern {config.pattern}."
            )
        if config.func_adapter_cfg.use_lora:
            raise ValueError(
                f"residual_head_ema_cfg does not support LoRA functional adapters for pattern {config.pattern}."
            )
        gate_cfg = config.residual_ensemble_gate_cfg
        if gate_cfg is None:
            return
        if gate_cfg.hidden_dim <= 0:
            raise ValueError(
                f"residual_ensemble_gate_cfg.hidden_dim must be positive for pattern {config.pattern}, "
                f"got {gate_cfg.hidden_dim}."
            )
        if gate_cfg.temperature <= 0:
            raise ValueError(
                f"residual_ensemble_gate_cfg.temperature must be positive for pattern {config.pattern}, "
                f"got {gate_cfg.temperature}."
            )

    def _process_module_configs(self) -> None:
        """Process and validate module configurations while maintaining backward compatibility."""
        if isinstance(self.target_modules, str):
            # Legacy mode - single pattern with default config
            self._module_configs[self.target_modules] = CLAREModuleConfig(
                pattern=self.target_modules,
                feature_dim=self.feature_dim,
                out_feature_dim=self.out_feature_dim,
                discriminator_cfg=self.discriminator_cfg,
                batch_first=self.batch_first,
                use_trainable_copy=self.use_trainable_copy,
                add_zero_init_conv_layer=self.add_zero_init_conv_layer,
                func_adapter_cfg=self.func_adapter_cfg
            )
        elif isinstance(self.target_modules, list):
            # List of module configs
            for config in self.target_modules:
                if isinstance(config, dict):
                    config = CLAREModuleConfig(**config)
                # Apply default values if not specified
                if config.out_feature_dim is None:
                    config.out_feature_dim = config.feature_dim
                if config.batch_first is None:
                    config.batch_first = self.batch_first
                if config.use_trainable_copy is None:
                    config.use_trainable_copy = self.use_trainable_copy
                if config.func_adapter_cfg and isinstance(config.func_adapter_cfg, dict):
                    config.func_adapter_cfg = FuncAdapterConfig(**config.func_adapter_cfg)
                if config.residual_head_ema_cfg and isinstance(config.residual_head_ema_cfg, dict):
                    config.residual_head_ema_cfg = ResidualHeadEMAConfig(**config.residual_head_ema_cfg)
                if config.residual_ensemble_gate_cfg and isinstance(config.residual_ensemble_gate_cfg, dict):
                    config.residual_ensemble_gate_cfg = ResidualEnsembleGateConfig(**config.residual_ensemble_gate_cfg)
                if config.discriminator_cfg and isinstance(config.discriminator_cfg, dict):
                    disc_cfg = copy.deepcopy(config.discriminator_cfg)
                    disc_type = disc_cfg.pop("type", DiscriminatorConfig.default_choice_name())
                    config.discriminator_cfg = DiscriminatorConfig.get_choice_class(disc_type)(**disc_cfg)
                self._validate_module_config(config)
                self._module_configs[config.pattern] = config
        elif isinstance(self.target_modules, dict):
            # Dict mapping patterns to configs
            for pattern, config in self.target_modules.items():
                if isinstance(config, dict):
                    config = CLAREModuleConfig(pattern=pattern, **config)
                # Apply default values if not specified
                if config.out_feature_dim is None:
                    config.out_feature_dim = config.feature_dim
                if config.batch_first is None:
                    config.batch_first = self.batch_first
                if config.use_trainable_copy is None:
                    config.use_trainable_copy = self.use_trainable_copy
                if config.func_adapter_cfg and isinstance(config.func_adapter_cfg, dict):
                    config.func_adapter_cfg = FuncAdapterConfig(**config.func_adapter_cfg)
                if config.residual_head_ema_cfg and isinstance(config.residual_head_ema_cfg, dict):
                    config.residual_head_ema_cfg = ResidualHeadEMAConfig(**config.residual_head_ema_cfg)
                if config.residual_ensemble_gate_cfg and isinstance(config.residual_ensemble_gate_cfg, dict):
                    config.residual_ensemble_gate_cfg = ResidualEnsembleGateConfig(**config.residual_ensemble_gate_cfg)
                if config.discriminator_cfg and isinstance(config.discriminator_cfg, dict):
                    disc_cfg = copy.deepcopy(config.discriminator_cfg)
                    disc_type = disc_cfg.pop("type", DiscriminatorConfig.default_choice_name())
                    config.discriminator_cfg = DiscriminatorConfig.get_choice_class(disc_type)(**disc_cfg)
                self._validate_module_config(config)
                self._module_configs[pattern] = config
                
    def get_module_config(self, module_name: str) -> Optional[CLAREModuleConfig]:
        """
        Get the configuration for a given module name by matching against patterns.
        Uses PEFT's check_target_module_exists for consistent pattern matching behavior.
        """
        from peft.tuners.tuners_utils import check_target_module_exists

        for pattern, config in self._module_configs.items():
            # Create temporary config to use check_target_module_exists
            temp_config = type('TempConfig', (), {'target_modules': pattern})()
            if check_target_module_exists(temp_config, module_name):
                return config
        return None

    @classmethod
    def check_kwargs(cls, **kwargs):
        kwargs.pop("adapter_ema_enabled", None)
        kwargs.pop("adapter_ema_ratios", None)
        kwargs.pop("adapter_ema_ensemble_method", None)
        return super().check_kwargs(**kwargs)


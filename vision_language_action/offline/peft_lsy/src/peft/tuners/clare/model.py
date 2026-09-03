# peft/tuners/clare/model.py

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Union, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file
from tqdm import tqdm

from peft.tuners.tuners_utils import BaseTuner, check_target_module_exists, onload_layer
from peft.utils import _get_submodules

from .config import CLAREConfig
from .layer import CLARELayer

def extract_layer(current_key: str, key_pattern: str) -> Optional[Tuple[str, int]]:
    """
    Returns (layer_name, layer_id) if `key_pattern` matches `current_key`,
    else returns None.

    key_pattern should contain:
      - (?P<layer_name>...)   -> e.g. (layers) or (encoders|decoders)
      - (?P<layer_id>\\d+)    -> the numeric id
    """
    m = re.search(key_pattern, current_key)
    if not m:
        return None
    if "layer_name" in m.re.groupindex and "layer_id" in m.re.groupindex:
        layer_name = m.group("layer_name")
        if bool(m.group("layer_id")):
            layer_id = int(m.group("layer_id"))
        else:
            layer_id = 0
    else:
        layer_name = m.group(0)
        layer_id = 0
    return layer_name, layer_id


class CLAREModel(BaseTuner):
    """
    PEFT-compatible tuner that injects OurAdapterLayer into target modules.
    """
    prefix = "clare_"
    _clare_layers: List[CLARELayer] = []
    router_safe_weights_name = "router_state.safetensors"
    router_weights_name = "router_state.bin"
    legacy_router_safe_weights_name = "rp_head.safetensors"
    legacy_router_weights_name = "rp_head.bin"

    def __init__(
        self,
        model,
        peft_config: Union[CLAREConfig, dict[str, CLAREConfig]],
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        self._clare_layers = []
        self._shared_routing_state: Dict[str, list[int]] = {}
        super().__init__(
            model,
            peft_config,
            adapter_name,
            low_cpu_mem_usage=low_cpu_mem_usage,
            state_dict=state_dict,
        )

    @staticmethod
    def _check_target_module_exists(peft_config: CLAREConfig, key: str) -> bool:
        # Check if any pattern in module_configs matches the key
        module_config = peft_config.get_module_config(key)
        return module_config is not None
    
    def _create_and_replace(
        self,
        peft_config: CLAREConfig,
        adapter_name: str,
        target: nn.Module,
        target_name: str,
        parent: nn.Module,
        current_key: str,
        *,
        parameter_name: Optional[str] = None,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")
            
        # Get the specific config for this module
        module_config = peft_config.get_module_config(current_key)
        if not module_config:
            raise ValueError(f"No configuration found for module {current_key}")
            
        # Extract layer info using the module's pattern
        layer_name, layer_id = extract_layer(current_key, module_config.pattern)

        # normal situation
        device_map = self.model.hf_device_map if hasattr(self.model, "hf_device_map") else None
        new_module = self._create_new_module(peft_config, adapter_name, target, layer_name, layer_id, target_name, original_key=current_key, device_map=device_map)
        new_module.set_shared_routing_state(self._shared_routing_state)
        self._replace_module(parent, target_name, new_module, target)

        self._clare_layers.append(new_module)



    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.base_layer

        meta = torch.device("meta")
        # dispatch to correct device
        for name, module in new_module.named_modules():
            if isinstance(module, CLARELayer):
                if hasattr(child, "qweight"):
                    weight = child.qweight
                elif hasattr(child, "W_q"):
                    weight = child.W_q
                elif hasattr(child, "weight"):
                    weight = child.weight
                elif getattr(child, "in_proj_weight", None) is not None:  # MHA
                    weight = child.in_proj_weight
                else:
                    weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    module.to(device=weight.device, dtype=weight.dtype)

    @staticmethod
    def _create_new_module(peft_config, adapter_name, target, layer_name, layer_id, base_layer_name, original_key, **kwargs):
        key = f"{layer_name}.{layer_id}"
        current_key = f"{key}.{base_layer_name}"

        if key not in peft_config.structure:
            peft_config.structure[key] = [0, 0]

        num_adapters, num_discriminators = peft_config.structure[key]
        
        # Get module-specific config
        module_config = peft_config.get_module_config(original_key)
        if not module_config:
            raise ValueError(f"No configuration found for module {original_key}")
            
        # Create CLARELayer with module-specific config
        new_module = CLARELayer(
            base_layer=target,
            peft_config=peft_config,
            module_config=module_config,  # Use module-specific config
            adapter_name=adapter_name,
            layer_name=layer_name, 
            layer_id=layer_id,
            base_layer_name=base_layer_name,
            num_adapters=num_adapters,
            num_discriminators=num_discriminators,
        )

        return new_module

    def disable_adapter_layers(self):
        pass

    def enable_adapter_layers(self):
        pass

    def set_adapter(self, adapter_name: str | list[str]):
        self.active_adapter = adapter_name

    def forward(self, *args: Any, **kwargs: Any):
        self.clear_shared_routing_state()
        return self.model.forward(*args, **kwargs)

    def _unload_and_optionally_merge(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: Optional[list[str]] = None,
    ):
        key_list = [key for key, _ in self.model.named_modules() if self.prefix not in key]
        desc = "Unloading " + ("and merging " if merge else "") + "model"
        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            with onload_layer(target):
                if hasattr(target, "unload_and_optionally_merge_module"):
                    # if layers have special unloading method, like MultiheadAttention, use that
                    unloaded_module = target.unload_and_optionally_merge_module(
                        merge=merge, safe_merge=safe_merge, adapter_names=adapter_names
                    )
                    self._replace_module(parent, target_name, unloaded_module, target)
                elif hasattr(target, "base_layer"):
                    if merge:
                        target.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    self._replace_module(parent, target_name, target.get_base_layer(), target)

        return self.model

    def unload(self) -> torch.nn.Module:
        """
        Gets back the base model by removing all the lora modules without merging. This gives back the original base
        model.
        """
        return self._unload_and_optionally_merge(merge=False)
        

    @property
    def adapter_layers(self):
        return self._clare_layers

    def uses_router(self) -> bool:
        return any(layer.routing_mode in {"rp_gate", "ws_router"} for layer in self._clare_layers)

    def uses_rp_gate(self) -> bool:
        return any(layer.routing_mode == "rp_gate" for layer in self._clare_layers)

    def update_residual_head_ema(self) -> None:
        for layer in self._clare_layers:
            layer.update_residual_head_ema()

    def clear_shared_routing_state(self) -> None:
        self._shared_routing_state.clear()

    def finalize_router_state(self) -> None:
        for layer in self._clare_layers:
            layer.update_router()

    def router_state_key_substrings(self) -> tuple[str, ...]:
        return (".rp_head.", ".ws_head.")

    def router_checkpoint_filenames(self) -> tuple[str, ...]:
        return (
            self.router_safe_weights_name,
            self.router_weights_name,
            self.legacy_router_safe_weights_name,
            self.legacy_router_weights_name,
        )

    def get_router_state_dict(self) -> Dict[str, torch.Tensor]:
        if not self.uses_router():
            return {}
        self.finalize_router_state()
        key_substrings = self.router_state_key_substrings()
        return {
            key: value.detach().cpu()
            for key, value in self.model.state_dict().items()
            if any(substring in key for substring in key_substrings)
        }

    def save_router_state_pretrained(self, save_directory: str, safe_serialization: bool = True) -> str | None:
        router_state = self.get_router_state_dict()
        if not router_state:
            return None

        file_name = self.router_safe_weights_name if safe_serialization else self.router_weights_name
        output_path = os.path.join(save_directory, file_name)
        if safe_serialization:
            safe_save_file(router_state, output_path, metadata={"format": "pt"})
        else:
            torch.save(router_state, output_path)
        return output_path

    def load_router_state_dict(self, router_state: Dict[str, torch.Tensor]) -> None:
        load_result = self.model.load_state_dict(router_state, strict=False)
        key_substrings = self.router_state_key_substrings()
        missing_router_keys = [
            key for key in load_result.missing_keys if any(substring in key for substring in key_substrings)
        ]
        unexpected_router_keys = [
            key for key in load_result.unexpected_keys if any(substring in key for substring in key_substrings)
        ]
        if missing_router_keys or unexpected_router_keys:
            raise ValueError(
                "Failed to restore router state cleanly. "
                f"Missing router keys: {missing_router_keys}. "
                f"Unexpected router keys: {unexpected_router_keys}."
            )

    def load_router_state_pretrained(self, load_directory: str) -> str | None:
        if not self.uses_router():
            return None

        router_state = None
        source_path = None
        for filename in self.router_checkpoint_filenames():
            candidate_path = os.path.join(load_directory, filename)
            if not os.path.exists(candidate_path):
                continue
            if filename.endswith(".safetensors"):
                router_state = safe_load_file(candidate_path)
            else:
                router_state = torch.load(candidate_path, map_location="cpu")
            source_path = candidate_path
            break

        if router_state is None or source_path is None:
            raise FileNotFoundError(
                f"Missing router checkpoint in {load_directory}. Expected one of "
                f"{', '.join(self.router_checkpoint_filenames())}."
            )

        self.load_router_state_dict(router_state)

        return source_path

    # Add this static method: return the config unchanged
    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        """
        Prepare and return the adapter config.
        For OurAdapter, we simply return the config unchanged.
        """
        return peft_config

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        """
        Mark only the adapter parameters as trainable.
        Freeze everything else (already done in __init__).
        This satisfies the BaseTuner abstract method contract.
        """

        # no need to enable gradient here
        # only enbale gradient when adding new adapters
        for n, p in model.named_parameters():
            p.requires_grad = False

        # # Explicitly enable gradients on adapter params
        # for layer in self._clare_layers:
        #     for n, p in layer.named_parameters():
        #         if "base_layer" not in n:
        #             p.requires_grad = True

    # ------- Convenience controls --------
    def forward(self, *args, **kwargs):
        self.clear_shared_routing_state()
        return self.model(*args, **kwargs)


    
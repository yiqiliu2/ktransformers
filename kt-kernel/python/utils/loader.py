"""
Weight loaders for different formats.

This module provides loaders for:
- SafeTensor format (for AMX quantized weights)
- GGUF format (for Llamafile quantized weights)
"""

from __future__ import annotations

import json
import os
import re
import numpy as np
import torch
from enum import IntEnum
from safetensors import safe_open
from gguf.gguf_reader import GGUFReader


class GGMLQuantizationType(IntEnum):
    """GGML quantization type enumeration"""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30


def translate_name_to_gguf(name):
    """
    Translate PyTorch tensor name to GGUF format
    """
    name = name.replace("lm_head.", "output.")
    name = name.replace("model.embed_tokens.", "token_embd.")
    name = name.replace("model.norm.", "output_norm.")
    name = name.replace("model.layers.", "blk.")
    name = name.replace(".input_layernorm", ".attn_norm")
    name = name.replace(".mlp.down_proj", ".ffn_down")
    name = name.replace(".mlp.gate_proj", ".ffn_gate")
    name = name.replace(".mlp.up_proj", ".ffn_up")
    name = name.replace(".post_attention_layernorm", ".ffn_norm")
    name = name.replace(".self_attn.q_proj", ".attn_q")
    name = name.replace(".self_attn.k_proj", ".attn_k")
    name = name.replace(".self_attn.v_proj", ".attn_v")
    name = name.replace(".self_attn.o_proj", ".attn_output")
    name = name.replace(".self_attn.qkv_proj", ".attn_qkv")
    name = name.replace(".self_attn.kv_a_proj_with_mqa", ".attn_kv_a_mqa")
    name = name.replace(".self_attn.kv_a_layernorm", ".attn_kv_a_norm")
    name = name.replace(".self_attn.kv_b_proj", ".attn_kv_b")
    name = name.replace(".self_attn.q_a_proj", ".attn_q_a")
    name = name.replace(".self_attn.q_a_layernorm", ".attn_q_a_norm")
    name = name.replace(".self_attn.q_b_proj", ".attn_q_b")
    name = name.replace(".self_attn.q_norm", ".attn_q_norm")
    name = name.replace(".self_attn.k_norm", ".attn_k_norm")
    name = name.replace(".shared_expert.", ".shared_experts.")
    name = name.replace(".shared_expert_", ".shared_experts_")
    name = name.replace(".gate_up_proj.", ".up_proj")
    name = name.replace(".mlp.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".mlp.gate.e_score_correction_bias", ".exp_probs_b.bias")
    name = name.replace(".mlp.gate", ".ffn_gate_inp")
    name = name.replace(".mlp.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".mlp.shared_experts.up_proj", ".ffn_up_shexp")
    name = name.replace(".mlp.shared_experts_gate", ".ffn_gate_inp_shexp")
    name = name.replace(".mlp.experts", "")
    name = name.replace(".mlp.experts.ffn_down_exps", ".ffn_down_exps")
    name = name.replace(".mlp.experts.ffn_gate_exps", ".ffn_gate_exps")
    name = name.replace(".mlp.experts.ffn_up_exps", ".ffn_up_exps")
    name = name.replace(".block_sparse_moe.gate.", ".ffn_gate_inp.")
    name = name.replace(".block_sparse_moe.experts", "")
    name = name.replace(".feed_forward.experts", "")
    name = name.replace(".feed_forward.router", ".ffn_gate_inp")
    name = name.replace(".feed_forward.shared_experts.down_proj", ".ffn_down_shexp")
    name = name.replace(".feed_forward.shared_experts.gate_proj", ".ffn_gate_shexp")
    name = name.replace(".feed_forward.shared_experts.up_proj", ".ffn_up_shexp")
    return name


class SafeTensorLoader:
    """
    SafeTensor format loader for AMX quantized weights.

    Supports loading tensors from .safetensors files with NUMA-sharded expert weights.
    """

    tensor_file_map: dict
    tensor_type_map: dict
    file_handle_map: dict
    tensor_device_map: dict

    def __init__(self, file_path: str):
        self.__load_tensor_file_map(file_path)

    def __load_tensor_file_map(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Path not found: {file_path}")
        if os.path.isfile(file_path):
            folder_path = os.path.dirname(file_path)
        else:
            folder_path = file_path
        self.file_handle_map = {}
        self.tensor_file_map = {}
        self.tensor_type_map = {}
        self.tensor_device_map = {}

        found_safetensor = False
        for root, _, files in os.walk(folder_path):
            files = sorted(files)
            for file in files:
                if file.endswith(".safetensors"):
                    found_safetensor = True
                    file_path = os.path.join(root, file)
                    if file not in self.file_handle_map:
                        try:
                            handle = safe_open(file_path, framework="pt")
                            self.file_handle_map[file] = handle
                        except Exception as e:
                            print(f"Error opening Safetensor file {file_path}: {e}")
                            continue

                    f = self.file_handle_map.get(file)
                    if f is None:
                        continue
                    try:
                        for key in f.keys():
                            self.tensor_file_map[key] = file
                    except Exception as e:
                        print(f"Error reading Safetensor file {file_path}: {e}")

        if not found_safetensor:
            raise FileNotFoundError(f"No Safetensor files found in {folder_path}")

    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        return tensor.to(device)

    def close_all_handles(self):
        """Close all file handles and clear the handle map.

        Note: safetensors.safe_open doesn't have a close() method,
        so we just clear the references and let garbage collection handle cleanup.
        """
        # safetensors.safe_open doesn't have close(), just clear references
        self.file_handle_map.clear()

    def load_experts(self, base_key: str, device: str = "cpu"):
        """
        Load expert weights from SafeTensor files.

        Expected format:
        - blk.{layer_index}.ffn_[up, down, gate]_exps.{expert_id}.numa.{numa_id}.weight
        - blk.{layer_index}.ffn_[up, down, gate]_exps.{expert_id}.numa.{numa_id}.scale

        Args:
            base_key: Base key like "blk.{layer_index}"
            device: Target device for tensors

        Returns:
            Dictionary with keys: up, gate, down, up_scale, gate_scale, down_scale
            Each value is a list of lists: [numa_id][expert_id] -> numpy array
        """
        up_base_key = f"{base_key}.ffn_up_exps"
        gate_base_key = f"{base_key}.ffn_gate_exps"
        down_base_key = f"{base_key}.ffn_down_exps"
        max_numa_id = -1
        max_experts_count = -1
        while self.has_tensor(f"{up_base_key}.{max_experts_count+1}.numa.{0}.weight"):
            max_experts_count += 1
        if max_experts_count == 0:
            raise ValueError(f"No experts found for key {base_key}")
        while self.has_tensor(f"{up_base_key}.{0}.numa.{max_numa_id+1}.weight"):
            max_numa_id += 1
        # Initialize empty lists to store tensors for each projection type
        up_weights = [[] for _ in range(max_numa_id + 1)]
        gate_weights = [[] for _ in range(max_numa_id + 1)]
        down_weights = [[] for _ in range(max_numa_id + 1)]
        up_scales = [[] for _ in range(max_numa_id + 1)]
        gate_scales = [[] for _ in range(max_numa_id + 1)]
        down_scales = [[] for _ in range(max_numa_id + 1)]
        # Check if backward weights exist
        up_bwd_base_key = f"{base_key}.ffn_up_bwd_exps"
        gate_bwd_base_key = f"{base_key}.ffn_gate_bwd_exps"
        down_bwd_base_key = f"{base_key}.ffn_down_bwd_exps"
        has_bwd = self.has_tensor(f"{gate_bwd_base_key}.{0}.numa.{0}.weight")

        if has_bwd:
            up_bwd_weights = [[] for _ in range(max_numa_id + 1)]
            gate_bwd_weights = [[] for _ in range(max_numa_id + 1)]
            down_bwd_weights = [[] for _ in range(max_numa_id + 1)]
            up_bwd_scales = [[] for _ in range(max_numa_id + 1)]
            gate_bwd_scales = [[] for _ in range(max_numa_id + 1)]
            down_bwd_scales = [[] for _ in range(max_numa_id + 1)]

        for numa_id in range(max_numa_id + 1):
            for expert_id in range(max_experts_count + 1):
                up_key = f"{up_base_key}.{expert_id}.numa.{numa_id}.weight"
                gate_key = f"{gate_base_key}.{expert_id}.numa.{numa_id}.weight"
                down_key = f"{down_base_key}.{expert_id}.numa.{numa_id}.weight"
                up_scale_key = f"{up_base_key}.{expert_id}.numa.{numa_id}.scale"
                gate_scale_key = f"{gate_base_key}.{expert_id}.numa.{numa_id}.scale"
                down_scale_key = f"{down_base_key}.{expert_id}.numa.{numa_id}.scale"
                # make sure contiguous
                up_tensor = self.load_tensor(up_key, device).numpy()
                gate_tensor = self.load_tensor(gate_key, device).numpy()
                down_tensor = self.load_tensor(down_key, device).numpy()
                up_scale_tensor = self.load_tensor(up_scale_key, device).numpy()
                gate_scale_tensor = self.load_tensor(gate_scale_key, device).numpy()
                down_scale_tensor = self.load_tensor(down_scale_key, device).numpy()

                up_weights[numa_id].append(up_tensor)
                gate_weights[numa_id].append(gate_tensor)
                down_weights[numa_id].append(down_tensor)
                up_scales[numa_id].append(up_scale_tensor)
                gate_scales[numa_id].append(gate_scale_tensor)
                down_scales[numa_id].append(down_scale_tensor)

                # Load backward weights if available
                if has_bwd:
                    gate_bwd_weights[numa_id].append(
                        self.load_tensor(f"{gate_bwd_base_key}.{expert_id}.numa.{numa_id}.weight", device).numpy()
                    )
                    up_bwd_weights[numa_id].append(
                        self.load_tensor(f"{up_bwd_base_key}.{expert_id}.numa.{numa_id}.weight", device).numpy()
                    )
                    down_bwd_weights[numa_id].append(
                        self.load_tensor(f"{down_bwd_base_key}.{expert_id}.numa.{numa_id}.weight", device).numpy()
                    )
                    gate_bwd_scales[numa_id].append(
                        self.load_tensor(f"{gate_bwd_base_key}.{expert_id}.numa.{numa_id}.scale", device).numpy()
                    )
                    up_bwd_scales[numa_id].append(
                        self.load_tensor(f"{up_bwd_base_key}.{expert_id}.numa.{numa_id}.scale", device).numpy()
                    )
                    down_bwd_scales[numa_id].append(
                        self.load_tensor(f"{down_bwd_base_key}.{expert_id}.numa.{numa_id}.scale", device).numpy()
                    )

        result = {
            "up": up_weights,
            "gate": gate_weights,
            "down": down_weights,
            "up_scale": up_scales,
            "gate_scale": gate_scales,
            "down_scale": down_scales,
        }
        if has_bwd:
            result["gate_bwd"] = gate_bwd_weights
            result["up_bwd"] = up_bwd_weights
            result["down_bwd"] = down_bwd_weights
            result["gate_bwd_scale"] = gate_bwd_scales
            result["up_bwd_scale"] = up_bwd_scales
            result["down_bwd_scale"] = down_bwd_scales
        return result

    def has_tensor(self, name: str):
        return name in self.tensor_file_map


class FP8SafeTensorLoader(SafeTensorLoader):
    """Loader for FP8 expert weights with auto-detection of naming formats.

    Supported formats:
    - DeepSeek style: {base}.mlp.experts.{id}.{gate,up,down}_proj.weight
    - Mixtral/MiniMax style: {base}.block_sparse_moe.experts.{id}.{w1,w3,w2}.weight
    - Mistral style: {base}.experts.{id}.{w1,w3,w2}.weight

    Supported scale formats (auto-detected):
    - Block-wise: weight_scale_inv (DeepSeek FP8)
    - Per-channel: weight_scale (GLM-4.7-FP8)

    The format is auto-detected during initialization.
    """

    # Known MoE naming formats: (experts_path_template, gate_name, up_name, down_name)
    MOE_FORMATS = {
        "deepseek": ("{base}.mlp.experts", "gate_proj", "up_proj", "down_proj"),
        "mixtral": ("{base}.block_sparse_moe.experts", "w1", "w3", "w2"),
        "mistral": ("{base}.experts", "w1", "w3", "w2"),
    }

    def __init__(self, file_path: str, scale_suffix: str = None):
        """Initialize FP8 loader with optional scale suffix override.

        Args:
            file_path: Path to safetensor files
            scale_suffix: Optional scale key suffix. If None, auto-detect between
                         'weight_scale_inv' (block-wise) and 'weight_scale' (per-channel).
        """
        super().__init__(file_path)
        self._detected_format = None
        self._scale_suffix = scale_suffix  # None means auto-detect
        # Set per_channel based on explicit scale_suffix if provided
        if scale_suffix == "weight_scale":
            self._is_per_channel = True
        elif scale_suffix == "weight_scale_inv":
            self._is_per_channel = False
        else:
            self._is_per_channel = False  # Will be updated in _detect_format if auto-detect
        self._is_vl_model = False
        self._detect_format()

    def _detect_format(self):
        """Auto-detect the MoE naming format and scale format by checking tensor keys."""
        # Sample some tensor names to detect format
        sample_keys = list(self.tensor_file_map.keys())[:1000]

        for fmt_name, (path_tpl, gate, up, down) in self.MOE_FORMATS.items():
            # Check if any key matches this format pattern
            # Look for pattern like: model.layers.0.{experts_path}.0.{gate_name}.weight
            for key in sample_keys:
                if ".experts." in key and f".{gate}.weight" in key:
                    # Verify the path template matches
                    if "block_sparse_moe.experts" in key and fmt_name == "mixtral":
                        self._detected_format = fmt_name
                        print(f"[FP8SafeTensorLoader] Detected format: {fmt_name}")
                        break
                    elif "mlp.experts" in key and "block_sparse_moe" not in key and fmt_name == "deepseek":
                        self._detected_format = fmt_name
                        print(f"[FP8SafeTensorLoader] Detected format: {fmt_name}")
                        break
                    elif fmt_name == "mistral" and ".mlp.experts" not in key and ".block_sparse_moe.experts" not in key:
                        self._detected_format = fmt_name
                        print(f"[FP8SafeTensorLoader] Detected format: {fmt_name}")
                        break
            if self._detected_format:
                break

        # Default to deepseek if no format detected
        if not self._detected_format:
            self._detected_format = "deepseek"
            print("[FP8SafeTensorLoader] No MoE format detected, defaulting to: deepseek")

        # Auto-detect scale suffix if not specified
        if self._scale_suffix is None:
            _, gate, _, _ = self.MOE_FORMATS[self._detected_format]
            # Check for per-channel scale (weight_scale) vs block-wise (weight_scale_inv)
            for key in sample_keys:
                if f".{gate}.weight_scale_inv" in key:
                    self._scale_suffix = "weight_scale_inv"
                    self._is_per_channel = False
                    print("[FP8SafeTensorLoader] Detected scale format: block-wise (weight_scale_inv)")
                    if key.startswith("model.language_model.") and self._detected_format == "deepseek":
                        # VL models(Qwen3.5): model.layers.{N} -> model.language_model.layers.{N}
                        self._is_vl_model = True
                        print("[FP8SafeTensorLoader] Detected VL model")
                    return
                elif f".{gate}.weight_scale" in key and "weight_scale_inv" not in key:
                    self._scale_suffix = "weight_scale"
                    # Some models (e.g., Mistral) use block-wise FP8 scales but keep
                    # the key suffix as `weight_scale` (without `_inv`). Infer format
                    # from scale tensor shape instead of suffix alone:
                    # - per-channel: [N] or [N, 1]
                    # - block-wise: [N_block, K_block] (both dims > 1)
                    scale_tensor = self.load_tensor(key, device="cpu")
                    if scale_tensor.dim() == 1:
                        self._is_per_channel = True
                    elif scale_tensor.dim() == 2 and scale_tensor.shape[1] == 1:
                        self._is_per_channel = True
                    else:
                        self._is_per_channel = False

                    scale_kind = "per-channel" if self._is_per_channel else "block-wise"
                    print(f"[FP8SafeTensorLoader] Detected scale format: {scale_kind} (weight_scale)")
                    return
            # Default to weight_scale_inv
            self._scale_suffix = "weight_scale_inv"
            self._is_per_channel = False
            print("[FP8SafeTensorLoader] No scale format detected, defaulting to: weight_scale_inv")
        else:
            # Scale suffix was explicitly provided
            scale_type = "per-channel" if self._is_per_channel else "block-wise"
            print(f"[FP8SafeTensorLoader] Using explicit scale format: {scale_type} ({self._scale_suffix})")

    def _get_experts_prefix_candidates(self, base_key: str) -> list[str]:
        """Get candidate experts prefixes based on detected format and base key variants."""
        path_tpl, _, _, _ = self.MOE_FORMATS[self._detected_format]
        candidates = []
        if self._is_vl_model:
            base_key = base_key.replace("model.layers", "model.language_model.layers")
        candidates.append(path_tpl.format(base=base_key))

        # Some model weights (e.g., Mistral native format) do not have "model." prefix.
        if base_key.startswith("model."):
            candidates.append(path_tpl.format(base=base_key[len("model.") :]))

        # Deduplicate while preserving order.
        return list(dict.fromkeys(candidates))

    def _get_proj_names(self):
        """Get projection names (gate, up, down) based on detected format."""
        _, gate, up, down = self.MOE_FORMATS[self._detected_format]
        return gate, up, down

    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        if device == "cpu":
            return tensor
        return tensor.to(device)

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load FP8 expert weights and their scale tensors.

        Supports both block-wise (weight_scale_inv) and per-channel (weight_scale) formats.
        Per-channel scales are squeezed from [N, 1] to [N] if needed.
        """
        experts_prefix_candidates = self._get_experts_prefix_candidates(base_key)
        gate_name, up_name, down_name = self._get_proj_names()

        expert_count = 0
        experts_prefix = None
        for prefix in experts_prefix_candidates:
            expert_count = 0
            while self.has_tensor(f"{prefix}.{expert_count}.{gate_name}.weight"):
                expert_count += 1
            if expert_count > 0:
                experts_prefix = prefix
                break

        if expert_count == 0 or experts_prefix is None:
            raise ValueError(f"No experts found for keys: {experts_prefix_candidates}")

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count
        gate_scales = [None] * expert_count
        up_scales = [None] * expert_count
        down_scales = [None] * expert_count

        for exp_id in range(expert_count):
            gate_w_key = f"{experts_prefix}.{exp_id}.{gate_name}.weight"
            up_w_key = f"{experts_prefix}.{exp_id}.{up_name}.weight"
            down_w_key = f"{experts_prefix}.{exp_id}.{down_name}.weight"
            gate_s_key = f"{experts_prefix}.{exp_id}.{gate_name}.{self._scale_suffix}"
            up_s_key = f"{experts_prefix}.{exp_id}.{up_name}.{self._scale_suffix}"
            down_s_key = f"{experts_prefix}.{exp_id}.{down_name}.{self._scale_suffix}"

            gate_weights[exp_id] = self.load_tensor(gate_w_key, device).contiguous()
            up_weights[exp_id] = self.load_tensor(up_w_key, device).contiguous()
            down_weights[exp_id] = self.load_tensor(down_w_key, device).contiguous()

            gate_scale = self.load_tensor(gate_s_key, device)
            up_scale = self.load_tensor(up_s_key, device)
            down_scale = self.load_tensor(down_s_key, device)

            # For per-channel scales, squeeze [N, 1] -> [N] if needed
            if self._is_per_channel:
                if gate_scale.dim() == 2 and gate_scale.shape[1] == 1:
                    gate_scale = gate_scale.squeeze(1)
                if up_scale.dim() == 2 and up_scale.shape[1] == 1:
                    up_scale = up_scale.squeeze(1)
                if down_scale.dim() == 2 and down_scale.shape[1] == 1:
                    down_scale = down_scale.squeeze(1)

            gate_scales[exp_id] = gate_scale.contiguous()
            up_scales[exp_id] = up_scale.contiguous()
            down_scales[exp_id] = down_scale.contiguous()

        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }

    def is_per_channel(self) -> bool:
        """Return True if using per-channel quantization, False for block-wise."""
        return self._is_per_channel


class BF16SafeTensorLoader(SafeTensorLoader):
    """Loader for native BF16 expert weights (no quantization, no scales).

    Supported formats:
    - DeepSeek style: {base}.mlp.experts.{id}.{gate,up,down}_proj.weight
    - Mixtral/MiniMax style: {base}.block_sparse_moe.experts.{id}.{w1,w3,w2}.weight
    - Mistral style: {base}.experts.{id}.{w1,w3,w2}.weight

    The format is auto-detected during initialization.
    """

    MOE_FORMATS = {
        "deepseek": ("{base}.mlp.experts", "gate_proj", "up_proj", "down_proj"),
        "mixtral": ("{base}.block_sparse_moe.experts", "w1", "w3", "w2"),
        "mistral": ("{base}.experts", "w1", "w3", "w2"),
    }

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._detected_format = None
        self._detect_format()

    def _detect_format(self):
        """Auto-detect the MoE naming format by checking tensor keys."""
        sample_keys = list(self.tensor_file_map.keys())[:1000]

        # Check for packed format first (Qwen3.5 MoE style: all experts in one 3D tensor)
        for key in sample_keys:
            if key.endswith(".mlp.experts.gate_up_proj"):
                self._detected_format = "packed"
                print("[BF16SafeTensorLoader] Detected format: packed (Qwen3.5 MoE style)")
                return

        for fmt_name, (path_tpl, gate, up, down) in self.MOE_FORMATS.items():
            for key in sample_keys:
                if ".experts." in key and f".{gate}.weight" in key:
                    if "block_sparse_moe.experts" in key and fmt_name == "mixtral":
                        self._detected_format = fmt_name
                        print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                        return
                    elif "mlp.experts" in key and "block_sparse_moe" not in key and fmt_name == "deepseek":
                        self._detected_format = fmt_name
                        print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                        return
                    elif fmt_name == "mistral" and ".mlp.experts" not in key and ".block_sparse_moe.experts" not in key:
                        self._detected_format = fmt_name
                        print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                        return

        self._detected_format = "deepseek"
        print("[BF16SafeTensorLoader] No MoE format detected, defaulting to: deepseek")

    def _get_experts_prefix_candidates(self, base_key: str) -> list[str]:
        """Get candidate experts prefixes based on detected format and base key variants."""
        path_tpl, _, _, _ = self.MOE_FORMATS[self._detected_format]
        candidates = [path_tpl.format(base=base_key)]

        # Some model weights (e.g., Mistral native format) do not have "model." prefix.
        if base_key.startswith("model."):
            candidates.append(path_tpl.format(base=base_key[len("model.") :]))

        return list(dict.fromkeys(candidates))

    def _get_proj_names(self):
        """Get projection names (gate, up, down) based on detected format."""
        _, gate, up, down = self.MOE_FORMATS[self._detected_format]
        return gate, up, down

    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        if device == "cpu":
            return tensor
        return tensor.to(device)

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load BF16 expert weights (no scales needed)."""
        if self._detected_format == "packed":
            return self._load_experts_packed(base_key, device)

        experts_prefix_candidates = self._get_experts_prefix_candidates(base_key)
        gate_name, up_name, down_name = self._get_proj_names()

        expert_count = 0
        experts_prefix = None
        for prefix in experts_prefix_candidates:
            expert_count = 0
            while self.has_tensor(f"{prefix}.{expert_count}.{gate_name}.weight"):
                expert_count += 1
            if expert_count > 0:
                experts_prefix = prefix
                break

        if expert_count == 0 or experts_prefix is None:
            raise ValueError(f"No experts found for keys: {experts_prefix_candidates}")

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count

        for exp_id in range(expert_count):
            gate_w_key = f"{experts_prefix}.{exp_id}.{gate_name}.weight"
            up_w_key = f"{experts_prefix}.{exp_id}.{up_name}.weight"
            down_w_key = f"{experts_prefix}.{exp_id}.{down_name}.weight"

            gate_weights[exp_id] = self.load_tensor(gate_w_key, device).contiguous()
            up_weights[exp_id] = self.load_tensor(up_w_key, device).contiguous()
            down_weights[exp_id] = self.load_tensor(down_w_key, device).contiguous()

        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
        }

    def _resolve_packed_experts_prefix(self, base_key: str) -> str:
        """Resolve the experts prefix for packed format, trying fallbacks."""
        # Direct: model.layers.{N}.mlp.experts
        experts_prefix = f"{base_key}.mlp.experts"
        if self.has_tensor(f"{experts_prefix}.gate_up_proj"):
            return experts_prefix

        # VL models: model.layers.{N} -> model.language_model.layers.{N}
        parts = base_key.split(".", 1)
        if len(parts) == 2:
            alt_base = f"{parts[0]}.language_model.{parts[1]}"
            experts_prefix = f"{alt_base}.mlp.experts"
            if self.has_tensor(f"{experts_prefix}.gate_up_proj"):
                return experts_prefix

        raise ValueError(f"No packed experts found for base_key '{base_key}'.")

    def _load_experts_packed(self, base_key: str, device: str = "cpu"):
        """Load packed expert weights (Qwen3.5 MoE style).

        Packed format stores all experts in stacked 3D tensors:
        - gate_up_proj: [num_experts, 2 * intermediate_size, hidden_size]
        - down_proj:    [num_experts, hidden_size, intermediate_size]
        """
        experts_prefix = self._resolve_packed_experts_prefix(base_key)

        gate_up_key = f"{experts_prefix}.gate_up_proj"
        down_key = f"{experts_prefix}.down_proj"

        gate_up = self.load_tensor(gate_up_key, device)  # [E, 2*I, H]
        down = self.load_tensor(down_key, device)  # [E, H, I]

        mid = gate_up.shape[1] // 2
        gate_list = [gate_up[i, :mid, :].contiguous() for i in range(gate_up.shape[0])]
        up_list = [gate_up[i, mid:, :].contiguous() for i in range(gate_up.shape[0])]
        down_list = [down[i].contiguous() for i in range(down.shape[0])]

        return {
            "gate": gate_list,
            "up": up_list,
            "down": down_list,
        }


class CompressedSafeTensorLoader(SafeTensorLoader):
    """Loader for compressed SafeTensor layouts (RAWINT4 weights)."""

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load raw expert weights stored in compressed safetensor format."""

        experts_prefix = f"{base_key}.mlp.experts"

        expert_idx = 0
        while self.has_tensor(f"{experts_prefix}.{expert_idx}.up_proj.weight_packed"):
            expert_idx += 1

        if expert_idx == 0:
            experts_prefix = f"language_model.{base_key}.mlp.experts"
            expert_idx = 0
            while self.has_tensor(f"{experts_prefix}.{expert_idx}.up_proj.weight_packed"):
                expert_idx += 1
            if expert_idx == 0:
                raise ValueError(f"No experts found for key {experts_prefix}")

        def load_projection(proj_name: str):
            weight_entries = []
            scale_entries = []

            for exp_id in range(expert_idx):
                weight_key = f"{experts_prefix}.{exp_id}.{proj_name}_proj.weight_packed"
                scale_key = f"{experts_prefix}.{exp_id}.{proj_name}_proj.weight_scale"

                if not self.has_tensor(weight_key):
                    raise KeyError(f"Missing tensor: {weight_key}")
                if not self.has_tensor(scale_key):
                    raise KeyError(f"Missing tensor: {scale_key}")

                weight_tensor = self.load_tensor(weight_key, device).contiguous()
                scale_tensor = self.load_tensor(scale_key, device).contiguous()

                weight_entries.append(weight_tensor)
                scale_entries.append(scale_tensor)

            return weight_entries, scale_entries

        gate_weights, gate_scales = load_projection("gate")
        up_weights, up_scales = load_projection("up")
        down_weights, down_scales = load_projection("down")

        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }


class BF16SafeTensorLoader(SafeTensorLoader):
    """Loader for native BF16 expert weights (no quantization, no scales).

    Supported formats:
    - DeepSeek style: {base}.mlp.experts.{id}.{gate,up,down}_proj.weight
    - Mixtral/MiniMax style: {base}.block_sparse_moe.experts.{id}.{w1,w3,w2}.weight

    The format is auto-detected during initialization.
    """

    MOE_FORMATS = {
        "deepseek": ("{base}.mlp.experts", "gate_proj", "up_proj", "down_proj"),
        "mixtral": ("{base}.block_sparse_moe.experts", "w1", "w3", "w2"),
    }

    def __init__(self, file_path: str):
        super().__init__(file_path)
        self._detected_format = None
        self._detect_format()

    def _detect_format(self):
        """Auto-detect the MoE naming format by checking tensor keys."""
        sample_keys = list(self.tensor_file_map.keys())[:1000]

        for fmt_name, (path_tpl, gate, up, down) in self.MOE_FORMATS.items():
            for key in sample_keys:
                if ".experts." in key and f".{gate}.weight" in key:
                    if "block_sparse_moe.experts" in key and fmt_name == "mixtral":
                        self._detected_format = fmt_name
                        print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                        return
                    elif "mlp.experts" in key and "block_sparse_moe" not in key and fmt_name == "deepseek":
                        self._detected_format = fmt_name
                        print(f"[BF16SafeTensorLoader] Detected format: {fmt_name}")
                        return

        self._detected_format = "deepseek"
        print("[BF16SafeTensorLoader] No MoE format detected, defaulting to: deepseek")

    def _get_experts_prefix(self, base_key: str) -> str:
        """Get the experts prefix based on detected format."""
        path_tpl, _, _, _ = self.MOE_FORMATS[self._detected_format]
        return path_tpl.format(base=base_key)

    def _get_proj_names(self):
        """Get projection names (gate, up, down) based on detected format."""
        _, gate, up, down = self.MOE_FORMATS[self._detected_format]
        return gate, up, down

    def load_tensor(self, key: str, device: str = "cpu"):
        if key not in self.tensor_file_map:
            raise KeyError(f"Key {key} not found in Safetensor files")
        file = self.tensor_file_map[key]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(key)
        if device == "cpu":
            return tensor
        return tensor.to(device)

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load BF16 expert weights (no scales needed).

        Args:
            base_key: Base key like "model.layers.{layer_index}"
            device: Target device for tensors

        Returns:
            Dictionary with keys: gate, up, down, gate_scale (None), up_scale (None), down_scale (None)
            gate/up/down: list of tensors [expert_id] -> tensor
        """
        experts_prefix = self._get_experts_prefix(base_key)
        gate_name, up_name, down_name = self._get_proj_names()

        expert_count = 0
        while self.has_tensor(f"{experts_prefix}.{expert_count}.{gate_name}.weight"):
            expert_count += 1

        if expert_count == 0:
            raise ValueError(f"No experts found for key {experts_prefix}")

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count

        for exp_id in range(expert_count):
            gate_w_key = f"{experts_prefix}.{exp_id}.{gate_name}.weight"
            up_w_key = f"{experts_prefix}.{exp_id}.{up_name}.weight"
            down_w_key = f"{experts_prefix}.{exp_id}.{down_name}.weight"

            gate_weights[exp_id] = self.load_tensor(gate_w_key, device).contiguous()
            up_weights[exp_id] = self.load_tensor(up_w_key, device).contiguous()
            down_weights[exp_id] = self.load_tensor(down_w_key, device).contiguous()

        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": None,
            "up_scale": None,
            "down_scale": None,
        }


class GGUFLoader:
    """
    GGUF format loader using the official gguf library (gguf.gguf_reader.GGUFReader)

    This is a cleaner implementation compared to manual binary parsing.
    """

    def __init__(self, gguf_path: str):
        """
        Initialize GGUF loader from a file or directory

        Args:
            gguf_path: Path to a single GGUF file or a directory containing GGUF files
        """
        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF path not found: {gguf_path}")

        self.tensor_info = {}
        self.metadata = {}
        self.tensor_file_map = {}
        self.file_data_map = {}

        if os.path.isfile(gguf_path) and gguf_path.endswith(".gguf"):
            print(f"\n[GGUFLoader] Loading single GGUF file : {os.path.basename(gguf_path)}")
            self._load_single_file(gguf_path)
        elif os.path.isdir(gguf_path):
            print(f"\n[GGUFLoader] Loading GGUF files from directory: {gguf_path}")
            self._load_directory(gguf_path)
        else:
            raise ValueError(f"Path must be a .gguf file or a directory: {gguf_path}")

        print(f"[GGUFLoader] Summary:")
        print(f"  Files loaded: {len(self.file_data_map)}")
        print(f"  Total tensors: {len(self.tensor_info)}")
        print(f"  Metadata keys: {len(self.metadata)}")
        tensors = ["blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"]
        for key in tensors:
            if key in self.tensor_info:
                info = self.tensor_info[key]
                print(f" {'.'.join(key.split('.')[2:-1])}, Dtype: {info['dtype'].name}")

    def _load_single_file(self, file_path: str):
        """Load a single GGUF file"""
        reader = GGUFReader(file_path)

        for key, field in reader.fields.items():
            value = field.parts[field.data[0]]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            elif isinstance(value, np.ndarray) and value.dtype == np.uint8:
                try:
                    value = bytes(value).decode("utf-8")
                except:
                    pass
            self.metadata[key] = value

        for tensor in reader.tensors:
            self.tensor_info[tensor.name] = {
                "shape": list(reversed(tensor.shape)),  # Reverse to match PyTorch order
                "dtype": tensor.tensor_type,
                "offset": tensor.data_offset,
                "n_elements": tensor.n_elements,
            }
            self.tensor_file_map[tensor.name] = file_path

        self.file_data_map[file_path] = np.memmap(file_path, mode="r")

    def _load_directory(self, dir_path: str):
        """Load all GGUF files from a directory (non-recursive)"""
        found_gguf = False

        for file in sorted(os.listdir(dir_path)):
            if file.endswith(".gguf"):
                found_gguf = True
                file_path = os.path.join(dir_path, file)
                print(f"  Loading: {file}")

                reader = GGUFReader(file_path)

                for key, field in reader.fields.items():
                    value = field.parts[field.data[0]]
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    elif isinstance(value, np.ndarray) and value.dtype == np.uint8:
                        try:
                            value = bytes(value).decode("utf-8")
                        except:
                            pass
                    self.metadata[key] = value

                for tensor in reader.tensors:
                    self.tensor_info[tensor.name] = {
                        "shape": list(reversed(tensor.shape)),
                        "dtype": tensor.tensor_type,
                        "offset": tensor.data_offset,
                        "n_elements": tensor.n_elements,
                    }
                    self.tensor_file_map[tensor.name] = file_path

                self.file_data_map[file_path] = np.memmap(file_path, mode="r")

        if not found_gguf:
            raise FileNotFoundError(f"No .gguf files found in directory: {dir_path}")

    def get_model_config(self, layer_idx: int = 0):
        """
        Extract model configuration from GGUF metadata and tensor shapes.

        Args:
            layer_idx: Layer index to inspect (default: 0)

        Returns:
            dict with keys: num_experts, num_experts_per_tok, hidden_size, moe_intermediate_size
        """
        config = {}

        arch = self.metadata.get("general.architecture", "unknown")

        num_experts = None
        for key_suffix in [
            "expert_count",
            "expert.count",
            "moe.expert_count",
            "expert_feed_forward_length",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                num_experts = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        num_experts_per_tok = None
        for key_suffix in [
            "expert_used_count",
            "expert.used_count",
            "moe.num_experts_per_tok",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                num_experts_per_tok = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        hidden_size = None
        for key_suffix in [
            "embedding_length",
            "embed_length",
            "hidden_size",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                hidden_size = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        moe_intermediate_size = None
        for key_suffix in [
            "expert_feed_forward_length",
            "feed_forward_length",
            "ffn_length",
            "intermediate_size",
        ]:
            key = f"{arch}.{key_suffix}"
            if key in self.metadata:
                val = self.metadata[key]
                moe_intermediate_size = int(val[0]) if isinstance(val, (list, np.ndarray)) else int(val)
                break

        if any(v is None for v in [num_experts, hidden_size, moe_intermediate_size]):

            base_key = f"blk.{layer_idx}.ffn_gate_exps.weight"
            if base_key in self.tensor_info:
                gate_shape = self.tensor_info[base_key]["shape"]
                print(f"  Found tensor '{base_key}' with shape: {gate_shape}")

                if len(gate_shape) >= 3:
                    if num_experts is None:
                        num_experts = int(gate_shape[0])
                    if moe_intermediate_size is None:
                        moe_intermediate_size = int(gate_shape[1])
                    if hidden_size is None:
                        hidden_size = int(gate_shape[2])

        config = {
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "hidden_size": hidden_size,
            "moe_intermediate_size": moe_intermediate_size,
        }

        return config

    def print_metadata(self, filter_keywords=None):
        """
        Print GGUF file metadata for debugging.

        Args:
            filter_keywords: Optional list of keywords to filter metadata keys
        """
        print(f"\n[GGUFLoader] GGUF Metadata:")
        print(f"  Total metadata entries: {len(self.metadata)}")

        if filter_keywords:
            filtered = {
                k: v for k, v in self.metadata.items() if any(kw.lower() in k.lower() for kw in filter_keywords)
            }
            for k, v in sorted(filtered.items()):
                print(f"  {k}: {v}")
        else:
            for k, v in sorted(self.metadata.items()):
                print(f"  {k}: {v}")

    def has_tensor(self, name: str):
        """Check if tensor exists"""
        name = translate_name_to_gguf(name)
        return name in self.tensor_info

    def get_ggml_type(self, name: str):
        """Get GGML type of a tensor"""
        name = translate_name_to_gguf(name)
        if name not in self.tensor_info:
            raise KeyError(f"Tensor '{name}' not found in GGUF files")
        return self.tensor_info[name]["dtype"]

    def get_undequanted_tensor_and_ggml_type(self, name: str):
        """
        Get tensor data and its GGML type without dequantizing

        Args:
            name: Tensor name (in PyTorch format, will be translated to GGUF format)

        Returns:
            (data, ggml_type): Tuple of tensor data and GGML quantization type
        """
        name = translate_name_to_gguf(name)

        if name not in self.tensor_info:
            raise KeyError(f"Tensor '{name}' not found in GGUF files")

        info = self.tensor_info[name]
        file_path = self.tensor_file_map[name]
        mmap_data = self.file_data_map[file_path]

        offset = info["offset"]
        n_elements = info["n_elements"]
        ggml_type = info["dtype"]

        GGML_QUANT_SIZES = {
            GGMLQuantizationType.F32: (1, 4),
            GGMLQuantizationType.F16: (1, 2),
            GGMLQuantizationType.BF16: (1, 2),
            GGMLQuantizationType.Q4_0: (32, 2 + 16),
            GGMLQuantizationType.Q4_1: (32, 2 + 2 + 16),
            GGMLQuantizationType.Q5_0: (32, 2 + 4 + 16),
            GGMLQuantizationType.Q5_1: (32, 2 + 2 + 4 + 16),
            GGMLQuantizationType.Q8_0: (32, 2 + 32),
            GGMLQuantizationType.Q8_1: (32, 4 + 4 + 32),
            GGMLQuantizationType.Q2_K: (256, 2 + 2 + 256 // 16 + 256 // 4),
            GGMLQuantizationType.Q3_K: (256, 2 + 256 // 4 + 256 // 8 + 12),
            GGMLQuantizationType.Q4_K: (256, 2 + 2 + 256 // 2 + 12),
            GGMLQuantizationType.Q5_K: (256, 2 + 2 + 256 // 2 + 256 // 8 + 12),
            GGMLQuantizationType.Q6_K: (256, 2 + 256 // 2 + 256 // 4 + 256 // 16),
            GGMLQuantizationType.Q8_K: (256, 4 + 256 + 256 // 8),
            GGMLQuantizationType.IQ2_XXS: (256, 2 + 256 // 4),
            GGMLQuantizationType.IQ2_XS: (256, 2 + 256 // 4 + 256 // 32),
            GGMLQuantizationType.IQ3_XXS: (256, 2 + 256 // 4 + 256 // 8),
            GGMLQuantizationType.IQ1_S: (256, 2 + 256 // 8 + 256 // 16),
            GGMLQuantizationType.IQ4_NL: (32, 2 + 16),
            GGMLQuantizationType.IQ3_S: (256, 2 + 256 // 4 + 256 // 8 + 256 // 32 + 4),
            GGMLQuantizationType.IQ2_S: (256, 2 + 256 // 4 + 256 // 16),
            GGMLQuantizationType.IQ4_XS: (256, 2 + 2 + 256 // 2 + 256 // 64),
            GGMLQuantizationType.I8: (1, 1),
            GGMLQuantizationType.I16: (1, 2),
            GGMLQuantizationType.I32: (1, 4),
            GGMLQuantizationType.I64: (1, 8),
            GGMLQuantizationType.F64: (1, 8),
            GGMLQuantizationType.IQ1_M: (256, 256 // 8 + 256 // 16 + 256 // 32),
        }

        block_size, type_size = GGML_QUANT_SIZES[ggml_type]
        n_bytes = n_elements * type_size // block_size

        data_bytes = mmap_data[offset : offset + n_bytes]
        data = torch.from_numpy(np.frombuffer(data_bytes, dtype=np.uint8).copy())

        return data, ggml_type


class GPTQSafeTensorLoader(FP8SafeTensorLoader):
    """Loader for symmetric GPTQ-Int4 expert weights (qweight + scales, no qzeros).

    Only supports sym=true, desc_act=false GPTQ models.

    Tensor keys:
    - qweight: {prefix}.{id}.{proj}.qweight  (int32, packed 8x4-bit along K)
    - scales:  {prefix}.{id}.{proj}.scales    (fp16 -> converted to fp32)
    """

    def __init__(self, file_path: str):
        # Call FP8SafeTensorLoader init (which calls SafeTensorLoader init + format detection)
        super().__init__(file_path, scale_suffix="scales")
        # Verify GPTQ config
        self._verify_gptq_config(file_path)

    def _detect_format(self):
        """Override FP8 format detection to look for .qweight instead of .weight."""
        sample_keys = list(self.tensor_file_map.keys())[:2000]

        for fmt_name, (path_tpl, gate, up, down) in self.MOE_FORMATS.items():
            for key in sample_keys:
                if ".experts." in key and f".{gate}.qweight" in key:
                    if "block_sparse_moe.experts" in key and fmt_name == "mixtral":
                        self._detected_format = fmt_name
                        break
                    elif "mlp.experts" in key and "block_sparse_moe" not in key and fmt_name == "deepseek":
                        self._detected_format = fmt_name
                        # Check for VL model (language_model prefix)
                        if "language_model." in key:
                            self._is_vl_model = True
                        break
                    elif fmt_name == "mistral" and "block_sparse_moe" not in key and "mlp" not in key:
                        self._detected_format = fmt_name
                        break
            if self._detected_format is not None:
                break

        if self._detected_format is None:
            self._detected_format = "deepseek"

        vl_str = " (VL model)" if self._is_vl_model else ""
        print(f"[GPTQSafeTensorLoader] Detected format: {self._detected_format}{vl_str}")

    def _verify_gptq_config(self, file_path):
        """Check that the model uses sym=true, desc_act=false."""
        import json
        import os

        config_path = os.path.join(os.path.dirname(file_path), "config.json")
        if not os.path.exists(config_path):
            # Try parent directory
            config_path = os.path.join(file_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            qc = config.get("quantization_config", {})
            if qc.get("quant_method") == "gptq":
                if qc.get("desc_act", False):
                    raise NotImplementedError(
                        "GPTQ desc_act=true is not supported. Only desc_act=false models are supported."
                    )
                if not qc.get("sym", True):
                    raise NotImplementedError(
                        "GPTQ sym=false (asymmetric) is not supported. Only sym=true models are supported."
                    )
                print(f"[GPTQSafeTensorLoader] Verified: sym={qc.get('sym')}, desc_act={qc.get('desc_act')}, "
                      f"bits={qc.get('bits')}, group_size={qc.get('group_size')}")

    def load_experts(self, base_key: str, device: str = "cpu"):
        """Load GPTQ expert qweight and scales.

        Returns dict with keys: gate, up, down (qweight int32), gate_scale, up_scale, down_scale (fp32).
        """
        experts_prefix_candidates = self._get_experts_prefix_candidates(base_key)
        gate_name, up_name, down_name = self._get_proj_names()

        expert_count = 0
        experts_prefix = None
        for prefix in experts_prefix_candidates:
            expert_count = 0
            while self.has_tensor(f"{prefix}.{expert_count}.{gate_name}.qweight"):
                expert_count += 1
            if expert_count > 0:
                experts_prefix = prefix
                break

        if expert_count == 0 or experts_prefix is None:
            raise ValueError(f"No GPTQ experts found for keys: {experts_prefix_candidates}")

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count
        gate_scales = [None] * expert_count
        up_scales = [None] * expert_count
        down_scales = [None] * expert_count

        for exp_id in range(expert_count):
            gate_weights[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{gate_name}.qweight", device).contiguous()
            up_weights[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{up_name}.qweight", device).contiguous()
            down_weights[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{down_name}.qweight", device).contiguous()

            gate_scales[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{gate_name}.scales", device).float().contiguous()
            up_scales[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{up_name}.scales", device).float().contiguous()
            down_scales[exp_id] = self.load_tensor(f"{experts_prefix}.{exp_id}.{down_name}.scales", device).float().contiguous()

        print(f"[GPTQSafeTensorLoader] Loaded {expert_count} experts from {experts_prefix}")
        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }


class MXFP4SafeTensorLoader(SafeTensorLoader):
    """Loader for native MXFP4 expert weights (DeepSeek-V4-Flash format).

    Per expert layout:
      {base}.ffn.experts.{i}.w1.weight  I8       [N, K/2]   nibble-packed E2M1 (gate)
      {base}.ffn.experts.{i}.w1.scale   F8_E8M0  [N, K/32]  ue8m0 group scale
      {base}.ffn.experts.{i}.w3.{weight,scale}              up
      {base}.ffn.experts.{i}.w2.{weight,scale}              down

    V4 ckpt keys are not prefixed with ``model.``; we also probe the stripped form so
    callers can keep passing ``base_key="model.layers.{L}"``. ue8m0 → bf16 is a lossless
    bit shift (both have an 8-bit exponent and zero mantissa for ue8m0), and the AMX
    FP4 backend already consumes bf16 scales.
    """

    EXPERTS_PATH_TPL = "{base}.ffn.experts"
    PROJ_NAMES = ("w1", "w3", "w2")  # (gate, up, down)

    def _experts_prefix_candidates(self, base_key: str) -> list[str]:
        candidates = [self.EXPERTS_PATH_TPL.format(base=base_key)]
        if base_key.startswith("model."):
            candidates.append(self.EXPERTS_PATH_TPL.format(base=base_key[len("model.") :]))
        return list(dict.fromkeys(candidates))

    @staticmethod
    def _ue8m0_to_bf16(scale_t: torch.Tensor) -> torch.Tensor:
        if scale_t.dtype != torch.uint8:
            scale_t = scale_t.view(torch.uint8)
        # bf16 = [sign(1) | exp(8) | mant(7)]; setting mant=0, exp=e gives 2^(e-127),
        # which is exactly the value encoded by ue8m0 for e ∈ [1, 254]. e=0 → bf16 +0
        # (acceptable: ue8m0=0 represents 2^-127, below bf16 normal range), e=255 → +inf.
        # Compute in int32 then narrow to int16 (max value is 255<<7=32640, fits int16),
        # because torch CPU has no lshift kernel for uint16.
        return (scale_t.to(torch.int32) << 7).to(torch.int16).view(torch.bfloat16).contiguous()

    def load_experts(self, base_key: str, device: str = "cpu"):
        gate_name, up_name, down_name = self.PROJ_NAMES
        prefix = None
        expert_count = 0
        for cand in self._experts_prefix_candidates(base_key):
            expert_count = 0
            while self.has_tensor(f"{cand}.{expert_count}.{gate_name}.weight"):
                expert_count += 1
            if expert_count > 0:
                prefix = cand
                break
        if prefix is None:
            raise ValueError(
                f"No MXFP4 experts found under any of: {self._experts_prefix_candidates(base_key)}"
            )

        gate_weights = [None] * expert_count
        up_weights = [None] * expert_count
        down_weights = [None] * expert_count
        gate_scales = [None] * expert_count
        up_scales = [None] * expert_count
        down_scales = [None] * expert_count

        for exp_id in range(expert_count):
            for proj, dst in (
                (gate_name, gate_weights),
                (up_name, up_weights),
                (down_name, down_weights),
            ):
                w = self.load_tensor(f"{prefix}.{exp_id}.{proj}.weight", device).contiguous()
                if w.dtype != torch.uint8:
                    w = w.view(torch.uint8)
                dst[exp_id] = w

            for proj, dst in (
                (gate_name, gate_scales),
                (up_name, up_scales),
                (down_name, down_scales),
            ):
                s = self.load_tensor(f"{prefix}.{exp_id}.{proj}.scale", device)
                dst[exp_id] = self._ue8m0_to_bf16(s)

        print(f"[MXFP4SafeTensorLoader] Loaded {expert_count} experts from {prefix}")
        return {
            "gate": gate_weights,
            "up": up_weights,
            "down": down_weights,
            "gate_scale": gate_scales,
            "up_scale": up_scales,
            "down_scale": down_scales,
        }


class MXFP4PackedLoader:
    """V4-Flash MXFP4 expert loader backed by a single packed blob produced by
    `~/.local/bin/dsv4-pack-experts.py`.

    Disk layout: one ``experts.bin`` of contiguous bytes, with one
    ``experts.idx.json`` mapping ``"L.E.proj"`` to byte offsets and shapes.
    Each (layer, expert, proj) entry stores ``[weight_bytes][scale_bytes]``
    so weight and scale of one projection sit next to each other on SSD.
    Layout order: layer-major, expert, then proj in (w1, w3, w2).

    Why: the safetensors-backed MXFP4SafeTensorLoader scatters expert reads
    across 46 shards, each with safetensors metadata at its head. Demand-
    paging during decode pulls one (small) byte range per expert across many
    shard files, defeating SSD readahead. The packed blob reduces the
    expert-bytes path to a single file with predictable layer-contiguous
    layout, making both kernel-driven faults and explicit MADV_WILLNEED
    prefetch much cheaper.

    Compatibility: returns the same dict shape as
    MXFP4SafeTensorLoader.load_experts so it slots into NativeMoEWrapper.

    Origin: yiqiliu2 V4-Flash perf, 2026-05-07.
    """

    PROJ_NAMES = ("w1", "w3", "w2")  # (gate, up, down) — same as MXFP4SafeTensorLoader

    def __init__(self, packed_dir: str):
        bin_path = os.path.join(packed_dir, "experts.bin")
        idx_path = os.path.join(packed_dir, "experts.idx.json")
        if not os.path.isfile(bin_path):
            raise FileNotFoundError(f"experts.bin missing at {bin_path}")
        if not os.path.isfile(idx_path):
            raise FileNotFoundError(f"experts.idx.json missing at {idx_path}")
        with open(idx_path) as f:
            self.index = json.load(f)
        if self.index.get("format_version") != 1:
            raise ValueError(f"unsupported pack format_version={self.index.get('format_version')}")
        self.bin_path = bin_path
        # MAP_PRIVATE read-only mmap; np.memmap defaults to mode='r'
        self.mm = np.memmap(bin_path, dtype=np.uint8, mode="r")
        # Originally we set MADV_RANDOM here to "respect" the per-expert
        # access pattern. That tanked SSD bandwidth: each expert read became
        # a synchronous 4 KB page fault, and kernel readahead was disabled.
        # Profiled at only ~100 MB/s read bw on Gen4 NVMe, ~30× off peak —
        # because we have ~6 active experts × ~580 KB = ~3.5 MB per layer
        # per token, but they're issued as 875 separate 4 KB faults instead
        # of ~30 64-KB reads. With layer-major contiguous layout, default
        # kernel readahead (~128 KB) actually buys us coalesced reads inside
        # each layer's region without much waste. Keep `_libc` for explicit
        # MADV_WILLNEED prefetch via `prefetch_experts()`.
        # yiqiliu2 / 2026-05-08.
        try:
            import ctypes
            self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        except Exception:
            self._libc = None

        # yiqiliu2 / 2026-05-08: parallel pread() prefetch path. mmap-fault
        # bw is capped by per-VMA semaphore (~222 MB/s for 16 concurrent
        # threads on this WSL2 + Gen4 NVMe rig). Direct pread on the raw fd
        # populates the same shared page cache without taking the VMA lock,
        # so the AMX kernel's later mmap reads hit warm cache.
        try:
            self._raw_fd = os.open(bin_path, os.O_RDONLY)
        except Exception:
            self._raw_fd = -1
        try:
            from concurrent.futures import ThreadPoolExecutor
            workers = int(os.environ.get("KT_PREFETCH_WORKERS", "8"))
            self._prefetch_pool = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="kt-prefetch"
            )
        except Exception:
            self._prefetch_pool = None

        self.layer_count = int(self.index.get("layer_count", 0))
        self.expert_count = int(self.index.get("expert_count", 0))

        # yiqiliu2 / 2026-05-08: mlock the first N GB of the packed blob
        # (layer-major contiguous → covers experts of layers 0..K).
        # Without this, OS LRU evicts hot expert pages under cache pressure
        # and we burn SSD bw re-faulting experts. mlock guarantees no eviction.
        # Default: 10 GB (RLIMIT_MEMLOCK is 11.8 GB on this host without sudo).
        # Set KT_MLOCK_GB=0 to disable. Set higher if RLIMIT_MEMLOCK is raised.
        try:
            mlock_gb = float(os.environ.get("KT_MLOCK_GB", "8"))
        except Exception:
            mlock_gb = 0.0
        # File-flag fallback because env vars get stripped on spawn.
        if mlock_gb <= 0 and os.path.exists("/tmp/kt_mlock_gb"):
            try:
                with open("/tmp/kt_mlock_gb") as _f:
                    mlock_gb = float(_f.read().strip())
            except Exception:
                pass
        mlock_bytes = 0
        if mlock_gb > 0 and self._libc is not None:
            mlock_bytes = int(mlock_gb * 1024 * 1024 * 1024)
            mlock_bytes = min(mlock_bytes, len(self.mm))
            try:
                import ctypes as _c
                base_addr = self.mm.ctypes.data
                rc = self._libc.mlock(
                    _c.c_void_p(base_addr), _c.c_size_t(mlock_bytes)
                )
                if rc == 0:
                    print(f"[MXFP4PackedLoader] mlock'd {mlock_gb:.1f} GB of "
                          f"experts.bin (offset 0..{mlock_bytes/1e9:.1f} GB) — "
                          f"these pages will not evict.", flush=True)
                else:
                    err = ctypes.get_errno()
                    print(f"[MXFP4PackedLoader] mlock {mlock_gb:.1f} GB FAILED "
                          f"errno={err} ({os.strerror(err)}). "
                          f"RLIMIT_MEMLOCK may be too low.", flush=True)
            except Exception as e:
                print(f"[MXFP4PackedLoader] mlock failed with exception: {e}",
                      flush=True)

        # yiqiliu2 / 2026-05-08: parallel pread prewarm. mmap-fault path
        # caps at ~600 MB/s buffered single-threaded on this WSL2 +
        # Gen4 NVMe rig (3.4 GB/s direct dd, 3.3 GB/s 8-thread parallel
        # pread). Prewarming the page cache via the existing prefetch
        # pool lets later mmap reads hit warm cache instead of taking
        # the per-VMA semaphore on each new page fault. Set
        # KT_PREWARM_GB=N to prewarm N GB of the blob starting after
        # the mlock'd region. With 88 GB RAM, sglang anon ~13 GB,
        # mlock 8 GB, max useful prewarm is ~70 GB. Set =0 to skip.
        try:
            prewarm_gb = float(os.environ.get("KT_PREWARM_GB", "0"))
        except Exception:
            prewarm_gb = 0.0
        if prewarm_gb <= 0 and os.path.exists("/tmp/kt_prewarm_gb"):
            try:
                with open("/tmp/kt_prewarm_gb") as _f:
                    prewarm_gb = float(_f.read().strip())
            except Exception:
                pass
        if (prewarm_gb > 0
                and self._raw_fd >= 0
                and self._prefetch_pool is not None):
            import time as _time
            blob_len = len(self.mm)
            start_off = mlock_bytes
            end_off = min(start_off + int(prewarm_gb * 1024**3), blob_len)
            if end_off > start_off:
                CHUNK = 4 * 1024 * 1024  # 4 MB
                fd = self._raw_fd

                def _prewarm_chunk(off, ln):
                    return len(os.pread(fd, ln, off))

                print(f"[MXFP4PackedLoader] prewarming "
                      f"{(end_off-start_off)/1e9:.1f} GB "
                      f"(offset {start_off/1e9:.1f}..{end_off/1e9:.1f} GB) "
                      f"with {workers}-thread pread...", flush=True)
                _t0 = _time.time()
                futs = []
                for _off in range(start_off, end_off, CHUNK):
                    _ln = min(CHUNK, end_off - _off)
                    futs.append(self._prefetch_pool.submit(_prewarm_chunk, _off, _ln))
                _read = 0
                for _f in futs:
                    try:
                        _read += _f.result()
                    except Exception:
                        pass
                _el = _time.time() - _t0
                print(f"[MXFP4PackedLoader] prewarm done: "
                      f"{_read/1e9:.1f} GB in {_el:.1f}s "
                      f"= {(_read/1e9/_el if _el>0 else 0):.2f} GB/s",
                      flush=True)

        # yiqiliu2 / 2026-05-08: explicit anon-RSS buffer pool with
        # frequency-targeted fill. Page cache route (mlock + prewarm)
        # survives boot but not decode pressure. With swappiness=10
        # anon RSS is stickier than page cache.
        #
        # Two modes:
        #   - KT_ANON_BUFFER_GB=N alone: legacy mode, fills first N GB
        #     of blob sequentially. Coverage = N/137 GB ≈ 44%.
        #   - KT_ANON_BUFFER_GB=N + KT_HOT_EXPERTS_PT=path: frequency-
        #     targeted mode. Reads logical_count.pt warmup data, picks
        #     top-K hottest (L, E) chunks (each 13.4 MB), copies them
        #     into the anon buffer, builds a chunk_id→anon_offset map.
        #     With same-size budget, hot 44% of experts cover ~80% of
        #     routing in typical MoE distributions.
        #
        # File-flag fallbacks: /tmp/kt_anon_buffer_gb, /tmp/kt_hot_experts_pt.
        self._anon_buf = None
        self._anon_size = 0
        self._chunk_size = 0  # 0 means legacy sequential mode
        self._hot_anon_start = None
        try:
            anon_gb = float(os.environ.get("KT_ANON_BUFFER_GB", "0"))
        except Exception:
            anon_gb = 0.0
        if anon_gb <= 0 and os.path.exists("/tmp/kt_anon_buffer_gb"):
            try:
                with open("/tmp/kt_anon_buffer_gb") as _f:
                    anon_gb = float(_f.read().strip())
            except Exception:
                pass
        hot_pt_path = os.environ.get("KT_HOT_EXPERTS_PT", "")
        if not hot_pt_path and os.path.exists("/tmp/kt_hot_experts_pt"):
            try:
                with open("/tmp/kt_hot_experts_pt") as _f:
                    hot_pt_path = _f.read().strip()
            except Exception:
                hot_pt_path = ""

        if (anon_gb > 0
                and self._raw_fd >= 0
                and self._prefetch_pool is not None):
            import time as _time
            blob_len = len(self.mm)
            target_bytes = int(anon_gb * 1024**3)
            fd = self._raw_fd

            # ---- Frequency-targeted path ----
            if hot_pt_path and os.path.isfile(hot_pt_path):
                try:
                    pt = torch.load(hot_pt_path, map_location="cpu", weights_only=True)
                    if isinstance(pt, dict) and "logical_count" in pt:
                        counts = pt["logical_count"]
                    else:
                        counts = pt
                    if counts.dim() == 3:
                        freq = counts.sum(dim=0).float().flatten()  # [L*E]
                    else:
                        freq = counts.float().flatten()
                    n_chunks = self.layer_count * self.expert_count
                    if freq.numel() != n_chunks:
                        raise ValueError(
                            f"hot pt shape {freq.numel()} doesn't match "
                            f"L({self.layer_count})*E({self.expert_count})={n_chunks}"
                        )

                    # Determine chunk size from index (assumes all (L,E) experts have same byte size).
                    first_le = self.index["experts"].get("0.0.w1")
                    second_le = self.index["experts"].get("0.1.w1")
                    if first_le and second_le:
                        chunk_size = int(second_le["w_off"]) - int(first_le["w_off"])
                    else:
                        chunk_size = 13369344  # known-good default for V4-Flash

                    n_hot = min(target_bytes // chunk_size, n_chunks)
                    if n_hot <= 0:
                        raise ValueError(f"n_hot computed as {n_hot} (anon_gb too small for chunk_size {chunk_size})")

                    # Sort chunks by descending frequency, take top-N.
                    hot_indices = torch.argsort(freq, descending=True)[:n_hot].tolist()

                    anon_size = n_hot * chunk_size
                    print(f"[MXFP4PackedLoader] frequency-targeted: "
                          f"allocating {anon_size/1e9:.1f} GB anon for "
                          f"top {n_hot}/{n_chunks} ({100.0*n_hot/n_chunks:.1f}%) "
                          f"hot experts (chunk_size {chunk_size/1e6:.1f} MB)...",
                          flush=True)
                    self._anon_buf = np.empty(anon_size, dtype=np.uint8)
                    self._anon_size = anon_size
                    self._chunk_size = chunk_size
                    self._hot_anon_start = np.full(n_chunks, -1, dtype=np.int64)
                    for i, ch_idx in enumerate(hot_indices):
                        self._hot_anon_start[ch_idx] = i * chunk_size

                    buf = self._anon_buf
                    def _fill_expert(ch_idx, anon_off):
                        orig_off = ch_idx * chunk_size
                        mv = memoryview(buf)[anon_off:anon_off+chunk_size]
                        return os.preadv(fd, [mv], orig_off)

                    print(f"[MXFP4PackedLoader] filling {n_hot} hot experts "
                          f"with {workers}-thread pread...", flush=True)
                    _t0 = _time.time()
                    _futs = []
                    for i, ch_idx in enumerate(hot_indices):
                        _futs.append(self._prefetch_pool.submit(_fill_expert, ch_idx, i * chunk_size))
                    _read = 0
                    for _f in _futs:
                        try:
                            _read += _f.result()
                        except Exception as _ex:
                            print(f"[MXFP4PackedLoader] anon-fill chunk error: {_ex}", flush=True)
                    _el = _time.time() - _t0
                    print(f"[MXFP4PackedLoader] anon buffer ready (frequency-targeted): "
                          f"{_read/1e9:.1f} GB in {_el:.1f}s "
                          f"= {(_read/1e9/_el if _el>0 else 0):.2f} GB/s",
                          flush=True)
                except Exception as _e:
                    print(f"[MXFP4PackedLoader] frequency-targeted fill failed: {_e!r}; "
                          f"falling back to legacy sequential fill", flush=True)
                    self._anon_buf = None
                    self._chunk_size = 0
                    self._hot_anon_start = None

            # ---- Legacy sequential path (fallback or when no .pt) ----
            if self._anon_buf is None:
                anon_bytes = min(target_bytes, blob_len)
                try:
                    self._anon_buf = np.empty(anon_bytes, dtype=np.uint8)
                except MemoryError as e:
                    print(f"[MXFP4PackedLoader] anon buffer alloc failed: {e}; "
                          f"falling back to mmap-only", flush=True)
                    self._anon_buf = None
                if self._anon_buf is not None:
                    self._anon_size = anon_bytes
                    CHUNK = 4 * 1024 * 1024
                    buf = self._anon_buf

                    def _fill_chunk(off, ln):
                        mv = memoryview(buf)[off:off+ln]
                        return os.preadv(fd, [mv], off)

                    print(f"[MXFP4PackedLoader] filling anon buffer "
                          f"{anon_bytes/1e9:.1f} GB with {workers}-thread pread...",
                          flush=True)
                    _t0 = _time.time()
                    _futs = []
                    for _off in range(0, anon_bytes, CHUNK):
                        _ln = min(CHUNK, anon_bytes - _off)
                        _futs.append(self._prefetch_pool.submit(_fill_chunk, _off, _ln))
                    _read = 0
                    for _f in _futs:
                        try:
                            _read += _f.result()
                        except Exception as _ex:
                            print(f"[MXFP4PackedLoader] anon-fill chunk error: {_ex}", flush=True)
                    _el = _time.time() - _t0
                    print(f"[MXFP4PackedLoader] anon buffer ready (legacy sequential): "
                          f"{_read/1e9:.1f} GB in {_el:.1f}s "
                          f"= {(_read/1e9/_el if _el>0 else 0):.2f} GB/s",
                          flush=True)

    # API parity with MXFP4SafeTensorLoader (the only methods used by the kt-kernel path).
    def has_tensor(self, name: str) -> bool:
        # Best-effort emulation: parse "..layers.{L}.ffn.experts.{E}.{proj}.weight" or .scale
        m = MXFP4PackedLoader._FALLBACK_KEY_RE.match(name)
        if not m:
            return False
        return f"{m.group('L')}.{m.group('E')}.{m.group('proj')}" in self.index["experts"]

    _FALLBACK_KEY_RE = re.compile(
        r"^(?:model\.)?layers\.(?P<L>\d+)\.ffn\.experts\.(?P<E>\d+)\.(?P<proj>w[123])\.(?:weight|scale)$"
    )

    @staticmethod
    def _ue8m0_to_bf16(scale_t: torch.Tensor) -> torch.Tensor:
        # Same logic as MXFP4SafeTensorLoader._ue8m0_to_bf16 — see that for derivation.
        if scale_t.dtype != torch.uint8:
            scale_t = scale_t.view(torch.uint8)
        return (scale_t.to(torch.int32) << 7).to(torch.int16).view(torch.bfloat16).contiguous()

    def _slice_bytes(self, offset: int, length: int) -> np.ndarray:
        # yiqiliu2 / 2026-05-08: 3-tier lookup
        #   1. frequency-targeted chunked anon (per-(L,E) remap; chunk_size > 0)
        #   2. legacy sequential anon (offset < anon_size)
        #   3. mmap fallback
        if self._chunk_size > 0:
            ch_idx = offset // self._chunk_size
            offset_within = offset - ch_idx * self._chunk_size
            if (offset_within + length <= self._chunk_size
                    and 0 <= ch_idx < self._hot_anon_start.shape[0]):
                anon_off = int(self._hot_anon_start[ch_idx])
                if anon_off >= 0:
                    return self._anon_buf[anon_off + offset_within : anon_off + offset_within + length]
            return self.mm[offset:offset + length]
        if self._anon_buf is not None and (offset + length) <= self._anon_size:
            return self._anon_buf[offset:offset + length]
        return self.mm[offset:offset + length]

    def _torch_view(self, raw: np.ndarray, shape, dtype_str: str) -> torch.Tensor:
        """Create a torch tensor that aliases the mmap region (zero-copy).

        For uint8 we return a uint8 tensor of the right shape; the C++
        consumer reinterprets via the same channel as MXFP4SafeTensorLoader.
        """
        # np -> torch zero-copy. The torch tensor borrows the mmap memory;
        # downstream Phase A direct-pointer code calls .data_ptr() which
        # then resolves to the mmap address. Tensor must be contiguous to
        # match the C++ expectation.
        t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8))
        if dtype_str == "uint8":
            pass  # raw bytes
        else:
            raise NotImplementedError(f"dtype {dtype_str} not handled")
        t = t.view(*shape).contiguous()
        return t

    def load_experts(self, base_key: str, device: str = "cpu"):
        m = re.match(r"^(?:model\.)?layers\.(?P<L>\d+)$", base_key)
        if not m:
            raise ValueError(f"unexpected base_key {base_key!r}; expected layers.{{N}}")
        L = int(m.group("L"))

        # Discover expert count for this layer from the index
        E = 0
        while f"{L}.{E}.w1" in self.index["experts"]:
            E += 1
        if E == 0:
            raise ValueError(f"no experts in pack for layer {L}")

        out = {
            "gate": [None] * E,        # w1 weight
            "up": [None] * E,          # w3 weight
            "down": [None] * E,        # w2 weight
            "gate_scale": [None] * E,  # w1 scale, returned as bf16 for AMX compat
            "up_scale": [None] * E,    # w3 scale
            "down_scale": [None] * E,  # w2 scale
        }
        proj_to_dst = {
            "w1": ("gate", "gate_scale"),
            "w3": ("up", "up_scale"),
            "w2": ("down", "down_scale"),
        }
        # ue8m0 passthrough (default): return raw 1-byte uint8 mmap views for the
        # scale stream. The AMX kernel's `broadcast_scale` reads `s_u8[g]` and shifts
        # `((uint32_t)s_u8 << 23)` to fp32 inline (lossless: ue8m0 is a pure exponent,
        # so 2^(s-127) reproduces the value exactly). Skipping `_ue8m0_to_bf16`
        # avoids materialising ~393 MB transient bf16 per layer. Set
        # `KT_DISABLE_UE8M0_PASSTHROUGH=1` to fall back to the legacy bf16 layout
        # (debug only — the C++ kernel will then read fp32 from convert_or_copy).
        # yiqiliu2 / 2026-05-07 — TPU "store packed, dequant in compute" rule.
        keep_bf16_legacy = os.environ.get("KT_DISABLE_UE8M0_PASSTHROUGH", "") == "1"
        for e in range(E):
            for proj in self.PROJ_NAMES:
                rec = self.index["experts"].get(f"{L}.{e}.{proj}")
                if rec is None:
                    raise KeyError(f"missing entry for L={L} E={e} proj={proj}")
                w_raw = self._slice_bytes(rec["w_off"], rec["w_len"])
                s_raw = self._slice_bytes(rec["s_off"], rec["s_len"])
                w_t = self._torch_view(w_raw, rec["w_shape"], rec["w_dtype"])
                s_raw_t = self._torch_view(s_raw, rec["s_shape"], rec["s_dtype"])
                if keep_bf16_legacy:
                    s_out = self._ue8m0_to_bf16(s_raw_t)
                else:
                    # Raw uint8 mmap view (zero-copy). _torch_view returns contiguous;
                    # data_ptr() walks the mmap region linearly. Kernel does inline shift.
                    s_out = s_raw_t
                w_dst, s_dst = proj_to_dst[proj]
                out[w_dst][e] = w_t
                out[s_dst][e] = s_out
        return out

    def prefetch_experts(self, layer: int, expert_ids):
        """Pre-read the routed-expert byte ranges into the OS page cache via
        parallel ``pread()`` on the raw fd. Bypasses Linux's per-VMA fault
        serialisation: 16-thread mmap touch peaks at ~222 MB/s; direct dd
        at 2.1 GB/s; so pread is the correct tool for "warm the cache fast".
        Same inode shared with the mmap → AMX kernel's later mmap accesses
        hit the cache at memory speed instead of stalling on faults.

        Best-effort: if the prefetch pool isn't initialised (init failed)
        or the raw fd is closed, silently no-op.
        yiqiliu2 / 2026-05-08.
        """
        if not hasattr(self, "_prefetch_pool") or self._prefetch_pool is None:
            return
        if not hasattr(self, "_raw_fd") or self._raw_fd < 0:
            return
        # Collect (off, len) pairs for w1/w3/w2 weight+scale of each expert
        ranges = []
        for e in expert_ids:
            for proj in self.PROJ_NAMES:
                rec = self.index["experts"].get(f"{layer}.{int(e)}.{proj}")
                if rec is None:
                    continue
                # weight + scale are packed contiguously
                ranges.append((rec["w_off"], rec["s_off"] + rec["s_len"]))
        if not ranges:
            return
        ranges.sort()
        # Coalesce neighbouring ranges (within 256 KB) so each pread is a
        # bigger contiguous read — NVMe loves big sequential I/O.
        merged = [list(ranges[0])]
        for start, end in ranges[1:]:
            if start <= merged[-1][1] + 262144:
                if end > merged[-1][1]:
                    merged[-1][1] = end
            else:
                merged.append([start, end])

        fd = self._raw_fd

        def _pread_chunk(off: int, length: int) -> None:
            # 1 MB sub-chunks to interleave with other workers — pread
            # releases the GIL so other threads can run.
            CHUNK = 1024 * 1024
            pos = off
            remaining = length
            while remaining > 0:
                n = min(CHUNK, remaining)
                _ = os.pread(fd, n, pos)
                pos += n
                remaining -= n

        # Fire and forget — racing against GPU's parallel attn + dense MLP.
        for s, e in merged:
            length = e - s
            if length <= 0:
                continue
            try:
                self._prefetch_pool.submit(_pread_chunk, s, length)
            except Exception:
                pass

    def prefetch_experts_sync(self, layer: int, expert_ids):
        """Synchronous variant of ``prefetch_experts``: submits the per-range
        pread chunks to the 8-thread pool and **blocks** until every chunk
        completes. Use this when the caller needs page cache to be warm before
        the AMX kernel reads bytes — the inference forward path's biggest
        single-step lever, since mmap-fault is single-threaded ~100 MB/s while
        an 8-thread pread microbench measures 5.58 GB/s on the same fd.

        For ~80 MB of routed-expert bytes per layer (6 experts × 13.4 MB), the
        wait is ~14 ms vs ~800 ms of cumulative mmap-fault stall the AMX
        kernel would otherwise pay. yiqiliu2 / 2026-05-08.
        """
        if not hasattr(self, "_prefetch_pool") or self._prefetch_pool is None:
            return
        if not hasattr(self, "_raw_fd") or self._raw_fd < 0:
            return
        ranges = []
        for e in expert_ids:
            for proj in self.PROJ_NAMES:
                rec = self.index["experts"].get(f"{layer}.{int(e)}.{proj}")
                if rec is None:
                    continue
                ranges.append((rec["w_off"], rec["s_off"] + rec["s_len"]))
        if not ranges:
            return
        ranges.sort()
        merged = [list(ranges[0])]
        for start, end in ranges[1:]:
            if start <= merged[-1][1] + 262144:
                if end > merged[-1][1]:
                    merged[-1][1] = end
            else:
                merged.append([start, end])

        fd = self._raw_fd

        def _pread_chunk(off: int, length: int) -> None:
            CHUNK = 1024 * 1024
            pos = off
            remaining = length
            while remaining > 0:
                n = min(CHUNK, remaining)
                _ = os.pread(fd, n, pos)
                pos += n
                remaining -= n

        futs = []
        for s, e in merged:
            length = e - s
            if length <= 0:
                continue
            try:
                futs.append(self._prefetch_pool.submit(_pread_chunk, s, length))
            except Exception:
                pass
        for f in futs:
            try:
                f.result()
            except Exception:
                pass

    # Compatibility: optional close
    def close_all_handles(self):
        try:
            del self.mm
        except Exception:
            pass

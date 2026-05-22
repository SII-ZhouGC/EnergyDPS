from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange


__OPERATOR__: dict[str, type["NonLinearOperator"]] = {}
__NOISE__: dict[str, type["Noise"]] = {}


def register_operator(name: str):
    def wrapper(cls: type["NonLinearOperator"]):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls

    return wrapper


def get_operator(name: str, **kwargs: Any) -> "NonLinearOperator":
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass


@register_operator(name="cnnsae_feature")
class CNNSAEFeatureOperator(NonLinearOperator):
    """Map an image in [-1, 1] to a scalar CNNSAE feature activation."""

    def __init__(
        self,
        sae_path: str,
        dino_ckpt_path: str,
        hook_point: str,
        target_feature_idx: int,
        device,
        dtype: str = "float32",
        reduce: str = "mix",
        mix_alpha: float = 0.7,
        lse_temperature: float = 0.2,
        center_frac: float = 1.0,
        center_mode: str = "crop",
    ):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.target_feature_idx = int(target_feature_idx)
        self.reduce = str(reduce)
        self.mix_alpha = float(mix_alpha)
        self.lse_temperature = float(lse_temperature)
        self.center_frac = float(center_frac)
        self.center_mode = str(center_mode)

        dtype_map = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        self.dtype = dtype_map.get(str(dtype).lower(), torch.float32)

        from lm_saes.backend.dino import dinov3
        from lm_saes.cnnsae import CNNSparseAutoEncoder
        from lm_saes.config import CNNSAEConfig, LanguageModelConfig

        device_str = "cuda" if self.device.type == "cuda" else "cpu"
        lm_cfg = LanguageModelConfig(
            model_name="dinov3_large",
            model_from_pretrained_path=dino_ckpt_path,
            device=device_str,
            dtype=str(self.dtype),
        )
        self.dino_lm = dinov3(lm_cfg)
        self.dino_lm.model.eval()
        for param in self.dino_lm.model.parameters():
            param.requires_grad = False

        sae_cfg = CNNSAEConfig.from_pretrained(sae_path, device=device_str, dtype=self.dtype)
        self.sae = CNNSparseAutoEncoder.from_config(sae_cfg)
        self.sae.eval()
        for param in self.sae.parameters():
            param.requires_grad = False

        modules = dict(self.dino_lm.model.named_modules())
        if hook_point not in modules:
            raise KeyError(f"hook_point not found in Dino model: {hook_point}")
        self._hook_point = hook_point
        self._captured = None

        def _fwd_hook(_module, _inp, out):
            self._captured = out

        self._hook_handle = modules[hook_point].register_forward_hook(_fwd_hook)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _preprocess(self, img_m11: torch.Tensor) -> torch.Tensor:
        img01 = ((img_m11 + 1.0) / 2.0).clamp(0.0, 1.0)
        return (img01 - self._mean) / self._std

    def _center_mask(self, h: int, w: int, *, device: torch.device) -> torch.Tensor | None:
        if self.center_frac >= 1.0:
            return None
        frac = max(self.center_frac, 1.0 / float(max(min(h, w), 1)))
        ch = max(int(round(h * frac)), 1)
        cw = max(int(round(w * frac)), 1)
        hs = (h - ch) // 2
        ws = (w - cw) // 2
        mask = torch.zeros((h, w), dtype=torch.bool, device=device)
        mask[hs : hs + ch, ws : ws + cw] = True
        return mask

    def _feature_map(self, data: torch.Tensor) -> torch.Tensor:
        img = data.to(self.device, dtype=self.dtype)
        img_norm = self._preprocess(img)

        self._captured = None
        _ = self.dino_lm.model(img_norm)
        acts_hw = self._captured
        if acts_hw is None:
            raise RuntimeError(f"failed to capture activation at hook_point={self._hook_point}")

        acts_seq = rearrange(acts_hw, "b d h w -> b (h w) d")
        batch = {self.sae.cfg.hook_point_in: acts_seq}
        x, encoder_kwargs, _ = self.sae.prepare_input(batch)
        _, feature_acts = self.sae.encode(x, return_hidden_pre=True, **encoder_kwargs)
        _, _, h, w = acts_hw.shape
        return feature_acts[..., self.target_feature_idx].view(feature_acts.shape[0], h, w)

    def forward(self, data, **kwargs):
        feats = self._feature_map(data)
        mask = self._center_mask(feats.shape[1], feats.shape[2], device=feats.device)

        if self.reduce == "max":
            if mask is not None:
                feats = feats.masked_fill(~mask.unsqueeze(0), -torch.inf)
            return feats.flatten(1).amax(dim=1)

        if self.reduce == "mean":
            if mask is None:
                return feats.flatten(1).mean(dim=1)
            x = feats * mask.to(dtype=feats.dtype).unsqueeze(0)
            return x.flatten(1).sum(dim=1) / mask.sum().clamp_min(1).to(dtype=feats.dtype)

        if self.reduce == "mix":
            if mask is None:
                peak = feats.flatten(1).amax(dim=1)
                cover = torch.relu(feats).flatten(1).mean(dim=1)
            else:
                peak = feats.masked_fill(~mask.unsqueeze(0), -torch.inf).flatten(1).amax(dim=1)
                x = torch.relu(feats) * mask.to(dtype=feats.dtype).unsqueeze(0)
                cover = x.flatten(1).sum(dim=1) / mask.sum().clamp_min(1).to(dtype=feats.dtype)
            return self.mix_alpha * peak + (1.0 - self.mix_alpha) * cover

        if self.reduce == "lse":
            if self.lse_temperature <= 0:
                raise ValueError("lse_temperature must be > 0")
            x = torch.relu(feats).flatten(1)
            if mask is not None:
                x = x.clone()
                x[:, ~mask.flatten()] = -torch.inf
            return self.lse_temperature * torch.logsumexp(x / self.lse_temperature, dim=1)

        raise ValueError("reduce must be one of: max | mean | mix | lse")


def register_noise(name: str):
    def wrapper(cls: type["Noise"]):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls

    return wrapper


def get_noise(name: str, **kwargs: Any) -> "Noise":
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser


class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass


@register_noise(name="clean")
class Clean(Noise):
    def forward(self, data):
        return data

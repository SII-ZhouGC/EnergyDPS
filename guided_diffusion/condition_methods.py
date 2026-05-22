from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


__CONDITIONING_METHOD__: dict[str, type["ConditioningMethod"]] = {}


def register_conditioning_method(name: str):
    def wrapper(cls: type["ConditioningMethod"]):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls

    return wrapper


def get_conditioning_method(name: str, operator: Any, noiser: Any, **kwargs: Any) -> "ConditioningMethod":
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)


class ConditioningMethod(ABC):
    def __init__(self, operator: Any, noiser: Any, **kwargs: Any):
        self.operator = operator
        self.noiser = noiser

    @abstractmethod
    def conditioning(self, x_t, measurement=None, noisy_measurement=None, **kwargs):
        pass


@register_conditioning_method(name="act_max")
class ActivationMaximization(ConditioningMethod):
    """Gradient guidance that maximizes a CNNSAE feature activation."""

    def __init__(self, operator: Any, noiser: Any, **kwargs: Any):
        super().__init__(operator, noiser)
        self.scale = float(kwargs.get("scale", 1.0))
        self.objective = str(kwargs.get("objective", "max")).lower()
        self.cap_value = kwargs.get("cap_value", None)
        self.cap_mode = str(kwargs.get("cap_mode", "clip")).lower()
        self.cap_penalty = float(kwargs.get("cap_penalty", 10.0))

    def _loss(self, act: torch.Tensor) -> torch.Tensor:
        if act.ndim > 0:
            act = act.mean()

        if self.objective == "max":
            return -act

        if self.objective in ("capped", "target") and self.cap_value is None:
            raise ValueError("objective is capped/target but cap_value is not set")

        cap = float(self.cap_value)
        if self.objective == "capped":
            if self.cap_mode == "clip":
                return -torch.clamp(act, max=cap)
            if self.cap_mode == "hinge":
                return -act + self.cap_penalty * torch.relu(act - cap).pow(2)
            raise ValueError("cap_mode must be 'clip' or 'hinge'")

        if self.objective == "target":
            return (act - cap).pow(2)

        raise ValueError("objective must be one of: max | capped | target")

    def conditioning(self, x_prev, x_t, x_0_hat, measurement=None, noisy_measurement=None, **kwargs):
        act = self.operator.forward(x_0_hat, **kwargs)
        loss = self._loss(act)
        (grad,) = torch.autograd.grad(outputs=loss, inputs=x_prev)
        x_t = x_t - grad * self.scale
        return x_t, loss.detach()

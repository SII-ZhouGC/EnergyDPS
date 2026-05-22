from __future__ import annotations

import math
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm


__SAMPLER__: dict[str, type["SpacedDiffusion"]] = {}


def register_sampler(name: str):
    def wrapper(cls: type["SpacedDiffusion"]):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __SAMPLER__[name] = cls
        return cls

    return wrapper


def get_sampler(name: str) -> type["SpacedDiffusion"]:
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(
    sampler: str,
    steps: int,
    noise_schedule: str,
    model_mean_type: str,
    model_var_type: str,
    dynamic_threshold: bool,
    clip_denoised: bool,
    rescale_timesteps: bool,
    timestep_respacing: str = "",
) -> "SpacedDiffusion":
    sampler_cls = get_sampler(name=sampler)
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = str(steps)
    return sampler_cls(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=model_var_type,
        dynamic_threshold=dynamic_threshold,
        clip_denoised=clip_denoised,
        rescale_timesteps=rescale_timesteps,
    )


class GaussianDiffusion:
    def __init__(
        self,
        betas,
        model_mean_type: str,
        model_var_type: str,
        dynamic_threshold: bool,
        clip_denoised: bool,
        rescale_timesteps: bool,
    ):
        self.betas = np.array(betas, dtype=np.float64)
        if self.betas.ndim != 1:
            raise ValueError("betas must be 1-D")
        if not ((0 < self.betas).all() and (self.betas <= 1).all()):
            raise ValueError("betas must be in (0, 1]")

        self.model_mean_type = str(model_mean_type)
        self.model_var_type = str(model_var_type)
        self.dynamic_threshold = bool(dynamic_threshold)
        self.clip_denoised = bool(clip_denoised)
        self.rescale_timesteps = bool(rescale_timesteps)
        self.num_timesteps = int(self.betas.shape[0])

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract_and_expand(self.sqrt_alphas_cumprod, t, x_start) * x_start
            + extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start) * noise
        )

    def p_sample_loop(
        self,
        model,
        x_start: torch.Tensor,
        measurement: torch.Tensor,
        measurement_cond_fn: Callable,
        record: bool = False,
        save_root: str | None = None,
        record_interval: int = 10,
    ) -> torch.Tensor:
        del record, save_root, record_interval
        img = x_start
        device = x_start.device

        pbar = tqdm(list(range(self.num_timesteps))[::-1], desc="DPS sampling")
        for idx in pbar:
            time = torch.full((img.shape[0],), idx, device=device, dtype=torch.long)
            step_idx = (self.num_timesteps - 1) - idx

            img = img.requires_grad_()
            out = self.p_sample(model=model, x=img, t=time)
            img, distance = measurement_cond_fn(
                x_t=out["sample"],
                measurement=measurement,
                noisy_measurement=None,
                x_prev=img,
                x_0_hat=out["pred_xstart"],
                step_idx=step_idx,
                num_steps=self.num_timesteps,
                t=time,
            )
            img = img.detach()
            pbar.set_postfix({"loss": f"{float(distance.item()):.4f}"}, refresh=False)

        return img

    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def p_mean_variance(self, model, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        model_output = model(x, self._scale_timesteps(t))
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        else:
            model_var_values = model_output

        pred_xstart = self._predict_xstart(x, t, model_output)
        pred_xstart = self._process_xstart(pred_xstart)
        model_mean = (
            extract_and_expand(self.posterior_mean_coef1, t, x) * pred_xstart
            + extract_and_expand(self.posterior_mean_coef2, t, x) * x
        )
        model_variance, model_log_variance = self._model_variance(model_var_values, t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart(self, x_t: torch.Tensor, t: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        if self.model_mean_type != "epsilon":
            raise NotImplementedError("This minimal DPS build supports model_mean_type='epsilon' only.")
        return (
            extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t) * x_t
            - extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t) * model_output
        )

    def _process_xstart(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_threshold:
            s = torch.quantile(x.abs().flatten(1), 0.95, dim=1).clamp_min(1.0)
            while s.ndim < x.ndim:
                s = s.unsqueeze(-1)
            x = x.clamp(-s, s) / s
        if self.clip_denoised:
            x = x.clamp(-1, 1)
        return x

    def _model_variance(self, model_var_values: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model_var_type != "learned_range":
            raise NotImplementedError("This minimal DPS build supports model_var_type='learned_range' only.")
        min_log = extract_and_expand(self.posterior_log_variance_clipped, t, model_var_values)
        max_log = extract_and_expand(np.log(self.betas), t, model_var_values)
        frac = (model_var_values + 1.0) / 2.0
        model_log_variance = frac * max_log + (1.0 - frac) * min_log
        return torch.exp(model_log_variance), model_log_variance

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t


class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map: list[int] = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.rescale_timesteps, self.original_num_steps)

    def _scale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map: list[int], rescale_timesteps: bool, original_num_steps: int):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x: torch.Tensor, ts: torch.Tensor, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name="ddpm")
class DDPM(SpacedDiffusion):
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.p_mean_variance(model, x, t)
        sample = out["mean"]
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        sample = sample + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


@register_sampler(name="ddim")
class DDIM(SpacedDiffusion):
    def p_sample(self, model, x: torch.Tensor, t: torch.Tensor, eta: float = 0.0) -> dict[str, torch.Tensor]:
        out = self.p_mean_variance(model, x, t)
        eps = self.predict_eps_from_x_start(x, t, out["pred_xstart"])

        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = torch.randn_like(x)
        mean_pred = out["pred_xstart"] * torch.sqrt(alpha_bar_prev) + torch.sqrt(
            1 - alpha_bar_prev - sigma**2
        ) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (x.ndim - 1)))
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t: torch.Tensor, t: torch.Tensor, pred_xstart: torch.Tensor) -> torch.Tensor:
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2


def get_named_beta_schedule(schedule_name: str, num_diffusion_timesteps: int) -> np.ndarray:
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return np.linspace(scale * 0.0001, scale * 0.02, num_diffusion_timesteps, dtype=np.float64)
    if schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999) -> np.ndarray:
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps: int, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for stride in range(1, num_timesteps):
                if len(range(0, num_timesteps, stride)) == desired_count:
                    return set(range(0, num_timesteps, stride))
            raise ValueError(f"cannot create exactly {desired_count} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        for _ in range(section_count):
            all_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        start_idx += size
    return set(all_steps)


def extract_and_expand(array: np.ndarray, time: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tensor = torch.from_numpy(array).to(target.device)[time].float()
    while tensor.ndim < target.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor.expand_as(target)

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one image by DPS activation maximization for a CNNSAE feature."
    )
    parser.add_argument("--feature_idx", type=int, required=True, help="Target CNNSAE feature index.")
    parser.add_argument("--cap_value", type=float, required=True, help="Capped activation target value.")
    parser.add_argument("--sae_path", type=Path, required=True, help="CNNSAE checkpoint directory.")
    parser.add_argument("--dino_ckpt_path", type=Path, required=True, help="DINOv3 ConvNeXt checkpoint file.")
    parser.add_argument("--diffusion_model_path", type=Path, required=True, help="ImageNet256 diffusion checkpoint file.")
    parser.add_argument("--output_path", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--hook_point", default="stages.2.20.hook_resid_pre", help="DINO hook point.")
    parser.add_argument("--steps", type=int, default=1000, help="DDIM timestep respacing count.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--seed", type=int, default=45, help="Random seed.")
    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r") as handle:
        return yaml.safe_load(handle)


def tensor_to_rgb_image(x):
    import numpy as np
    from PIL import Image

    img = x.detach().cpu()[0]
    img = ((img + 1.0) / 2.0).clamp(0.0, 1.0)
    array = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def main() -> None:
    args = parse_args()
    require_path(args.sae_path, "CNNSAE checkpoint directory")
    require_path(args.dino_ckpt_path, "DINO checkpoint")
    require_path(args.diffusion_model_path, "Diffusion checkpoint")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    import torch

    from guided_diffusion.condition_methods import get_conditioning_method
    from guided_diffusion.gaussian_diffusion import create_sampler
    from guided_diffusion.measurements import get_noise, get_operator
    from guided_diffusion.unet import create_model

    device = args.device
    if "cuda" in device and not torch.cuda.is_available():
        print(f"CUDA is unavailable; falling back from {device} to cpu.")
        device = "cpu"

    seed_everything(args.seed)

    model_cfg = load_yaml(ROOT / "configs" / "imagenet_model_config.yaml")
    diff_cfg = load_yaml(ROOT / "configs" / "diffusion_config.yaml")
    model_cfg["model_path"] = str(args.diffusion_model_path)
    diff_cfg = dict(diff_cfg)
    diff_cfg["timestep_respacing"] = f"ddim{args.steps}"

    model = create_model(**model_cfg).to(device)
    model.eval()
    sampler = create_sampler(**diff_cfg)

    operator = get_operator(
        name="cnnsae_feature",
        device=torch.device(device),
        dino_ckpt_path=str(args.dino_ckpt_path),
        sae_path=str(args.sae_path),
        hook_point=args.hook_point,
        target_feature_idx=args.feature_idx,
        reduce="mix",
        mix_alpha=0.7,
        center_frac=0.5,
        center_mode="crop",
    )
    noiser = get_noise(name="clean")
    cond_method = get_conditioning_method(
        name="act_max",
        operator=operator,
        noiser=noiser,
        scale=1.0,
        objective="capped",
        cap_value=args.cap_value,
        cap_mode="clip",
    )

    measurement = torch.zeros(1, 3, 256, 256, device=device)
    x_start = torch.randn(1, 3, 256, 256, device=device).requires_grad_()
    sample_fn = partial(
        sampler.p_sample_loop,
        model=model,
        measurement_cond_fn=cond_method.conditioning,
    )
    recon = sample_fn(
        x_start=x_start,
        measurement=measurement,
        record=False,
        save_root=str(args.output_path.parent),
        record_interval=10,
    )

    with torch.no_grad():
        final_act = float(operator.forward(recon).item())

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_rgb_image(recon).save(args.output_path)
    print(f"Saved image: {args.output_path}")
    print(f"Final activation: {final_act:.6f}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

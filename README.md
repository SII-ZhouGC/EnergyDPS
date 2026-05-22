# Minimal DPS CNNSAE Feature Visualization

This directory contains only the code needed to generate one image with DPS
activation maximization for a single CNN-SAE feature.

It does not include model weights. Provide these paths at runtime:

- ImageNet256 diffusion checkpoint, for example `imagenet256.pt`
- DINOv3 ConvNeXt-Large checkpoint
- CNN-SAE checkpoint directory

## Run

```bash
python run_dps.py \
  --feature_idx 9863 \
  --cap_value 20.0 \
  --sae_path /path/to/cnnsae_checkpoint \
  --dino_ckpt_path /path/to/dinov3_convnext_large.pth \
  --diffusion_model_path /path/to/imagenet256.pt \
  --output_path outputs/feature_9863_dps.png
```

The script writes the generated PNG to `--output_path` and prints the final
CNN-SAE feature activation.

## Defaults

The implementation matches the DPS branch of the original batch visualization
script:

- `hook_point=stages.2.20.hook_resid_pre`
- `reduce=mix`
- `mix_alpha=0.7`
- `center_frac=0.5`
- `center_mode=crop`
- `objective=capped`
- `cap_mode=clip`
- `scale=1.0`
- `steps=1000`
- `seed=45`

## Dependencies

Create a fresh environment with `uv`:

```bash
cd /path/to/submit_code
uv sync
uv run python -c "import lm_saes; print(lm_saes.__file__)"
uv run run-dps --help
```

`uv sync` installs the copied `src/lm_saes` as the `lm_saes` package and installs
the copied `TransformerLens` directory as a local `transformer-lens` dependency.

If you do not use `uv`, create a Python 3.11 environment and install this
directory as a package:

```bash
pip install -e .
python -c "import lm_saes; print(lm_saes.__file__)"
python run_dps.py --help
```

Make sure the PyTorch/CUDA versions match the target machine.

`src/` and `TransformerLens/` are copied from the original repository without
modification.

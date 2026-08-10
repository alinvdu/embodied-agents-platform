# ACT right-arm training

This guide trains ACT from scratch on:

- Repository: `alindumitru/robot42_grab_to_basket_right_arm_v0`
- Local root: `datasets/robot42_grab_to_basket_right_arm_v0`
- Episodes: 183
- Frames: 92,832 at 30 FPS
- Inputs: head camera, right-wrist camera, and six right-arm joint positions
- Outputs: six right-arm joint actions

Run all commands from `/home/alin/Robot42`.

## 1. Preflight

```bash
cd /home/alin/Robot42
source /home/alin/miniconda3/etc/profile.d/conda.sh
conda activate lerobot-smolvla

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
test -f datasets/robot42_grab_to_basket_right_arm_v0/meta/info.json
mkdir -p artifacts/training
```

Do not start training unless `CUDA: True` is printed.

## 2. Batch-8 smoke test

This tests the complete batch-8 training path for 200 steps. Its checkpoint is
only for validating the setup and is not useful for robot evaluation.

```bash
set -o pipefail

lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --policy.push_to_hub=false \
  --dataset.repo_id=alindumitru/robot42_grab_to_basket_right_arm_v0 \
  --dataset.root=datasets/robot42_grab_to_basket_right_arm_v0 \
  --dataset.streaming=false \
  --dataset.image_transforms.enable=true \
  --batch_size=8 \
  --steps=200 \
  --num_workers=4 \
  --log_freq=20 \
  --save_checkpoint=true \
  --save_freq=200 \
  --output_dir=outputs/train/act_xlerobot_right_v0_b8_smoke_s200 \
  --job_name=act_xlerobot_right_v0_b8_smoke_s200 \
  --wandb.enable=false \
  2>&1 | tee artifacts/training/act_xlerobot_right_v0_b8_smoke_s200.log
```

The smoke test passes when:

- Training reports 183 episodes and 92,832 frames.
- The effective batch size is 8.
- The process completes all 200 steps without CUDA OOM, NaN loss, or video
  decoding errors.
- The final checkpoint exists under
  `outputs/train/act_xlerobot_right_v0_b8_smoke_s200/checkpoints/`.
- `nvidia-smi` in another terminal shows `lerobot-train` using the GPU.

The full run uses a different output directory, so the smoke output does not
need to be deleted first.

## 3. Full training

At batch size 8, 100,000 updates expose the model to 800,000 sampled frames,
which is approximately 8.6 passes over this dataset.

```bash
set -o pipefail

lerobot-train \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_lr_backbone=1e-5 \
  --policy.push_to_hub=false \
  --dataset.repo_id=alindumitru/robot42_grab_to_basket_right_arm_v0 \
  --dataset.root=datasets/robot42_grab_to_basket_right_arm_v0 \
  --dataset.streaming=false \
  --dataset.image_transforms.enable=true \
  --batch_size=8 \
  --steps=100000 \
  --num_workers=4 \
  --log_freq=200 \
  --save_checkpoint=true \
  --save_freq=10000 \
  --output_dir=outputs/train/act_xlerobot_right_v0_b8_s100k \
  --job_name=act_xlerobot_right_v0_b8_s100k \
  --wandb.enable=false \
  2>&1 | tee artifacts/training/act_xlerobot_right_v0_b8_s100k.log
```

This saves checkpoints every 10,000 steps, including the final 100,000-step
checkpoint. ACT uses its policy preset: AdamW with a constant `1e-5` learning
rate and `1e-4` weight decay. There is no learning-rate scheduler in this
installed ACT implementation.

## Batch-4 fallback

ACT should fit at batch size 8 on a 16 GB RTX 4060 Ti. If the smoke test
nevertheless runs out of memory, change `--batch_size=8` to `--batch_size=4`.
Use `--steps=200000` for the full run to preserve the same 800,000 sampled-frame
exposure as batch 8 at 100,000 steps.

Use a new output directory when retrying. LeRobot refuses to overwrite an
existing training directory unless the run is explicitly resumed.

## Notes

- No `rename_map` is needed. ACT accepts both dataset image keys directly.
- `n_action_steps=10` means inference executes about 0.33 seconds of actions at
  30 FPS before requesting a new observation. The policy still predicts
  50-step chunks during training.
- Built-in image augmentation is enabled. It includes color, sharpness, and
  small affine changes.
- The current local LeRobot version is `0.4.2`; these commands use flags
  available in that installed version.

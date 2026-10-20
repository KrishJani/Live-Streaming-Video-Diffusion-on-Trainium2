"""Warm-up for running g4_e2e.py on an arbitrary 512x512 clip (no CPU reference needed).

    NEURON_RT_VISIBLE_CORES=0 python g5_clip.py data/raw_512.mp4
    python g4_e2e.py --clip data/raw_512.mp4 --out out/tier2_clip.mp4

Same warm-up as g3_pipeline.py warm, but the first 3 latent frames come from this clip: TAEHV encoder on CPU
(pixel chunks 0-2), rescaled to the pipeline's raw-latent convention, then the 2-frame session start on the CPU
port and frame 2 of both denoising rows on the single-layer graphs. Writes the per-stage KV caches and the 3
warm-up latents to /dev/shm/g3 for the stage processes.
"""
import sys

import numpy as np
import torch

from g2_common import NOISE_SCALE, STEPS, noise_scale_and_step, use_repo
from g3_pipeline import SHM, save_warm_caches

MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, 16, 1, 1)
STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, 16, 1, 1)


def load_frames(path):
    """All frames of a 512x512 clip as [T, 3, H, W] in [0, 1], plus its fps."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps, frames = cap.get(cv2.CAP_PROP_FPS), []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 255.0
    assert x.shape[-2:] == (512, 512), x.shape
    return x, fps


def step_for(x01, f, scale):
    """Adaptive noise scale / first step for latent frame f (Wan timing: pixel frames 4f-3..4f, previous 4f-4)."""
    seg = (x01[4 * f - 4:4 * f + 1] * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0)
    return noise_scale_and_step(seg, scale, NOISE_SCALE)


def main():
    torch.set_grad_enabled(False)
    torch.manual_seed(1234)
    x01, fps = load_frames(sys.argv[1])
    use_repo()
    from models.wan.taehv_wrapper import TAEHV
    from g2_common import SDV2_ROOT
    from g2_neuron import Runner
    taehv = TAEHV(checkpoint_path=str(SDV2_ROOT / "ckpts" / "taew2_1.pth")).eval().requires_grad_(False)
    lat = taehv.encode_video(x01[:12].unsqueeze(0), parallel=False)[0]  # [3, 16, 64, 64], normalised
    lat = (lat * STD + MEAN).unsqueeze(0)                                 # [1, 3, 16, 64, 64], raw-latent convention

    run = Runner("neuron", None)
    noisy = torch.randn_like(lat[:, :2]) * NOISE_SCALE + lat[:, :2] * (1 - NOISE_SCALE)
    x0s = run.start(noisy, [torch.randn(2, 16, 64, 64)])
    scale, step = step_for(x01, 2, NOISE_SCALE)
    noisy2 = torch.randn_like(lat[:, 2:3]) * scale + lat[:, 2:3] * (1 - scale)
    x0_r0 = run.forward_row(0, noisy2, step, 2)
    x0_r1 = run.forward_row(1, run.host.renoise(x0_r0, torch.randn_like(x0_r0), STEPS[1]), STEPS[1], 2)
    save_warm_caches(run.rows)  # per stage of the current LIVEVID_STAGES
    torch.save(run.ctx, SHM / "ctx.pt")
    torch.save({"finals": [x0s[-1], x0_r1], "clip": sys.argv[1], "noise_scale": scale}, SHM / "warm.pt")
    print(f"WARM OK clip={sys.argv[1]} frames={x01.shape[0]} fps={fps} noise_scale={scale:.3f} step={step}", flush=True)


if __name__ == "__main__":
    main()

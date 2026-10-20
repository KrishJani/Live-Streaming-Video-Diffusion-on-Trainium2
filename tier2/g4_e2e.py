"""End to end on 4 NeuronCores: pixels -> TAEHV encode -> 30-block causal DiT (2 passes) -> TAEHV decode -> pixels.

    python g3_pipeline.py warm        (once; fills /dev/shm/g3 with the warmed KV caches)
    python g4_e2e.py --chunks 120

Same stage processes as g3_pipeline.py (layers 0-7, 8-15, 16-22, 23-29 by default, one core each, KV caches
resident; LIVEVID_STAGES changes the split, see g3_stage.py). The TAEHV graphs live inside two stage processes: by
default the encoder with the second-to-last stage and the decoder with the last one (--enc-stage, --dec-stage).
No extra processes. TAEHV temporal state (9 tensors per half) stays on the device when the resident graphs from
vae_resident.py exist (--vae auto, the default); otherwise it is carried on the host side of those processes
(--vae host, the original behaviour, 33 MB / 44 MB each way per call).

Finished work goes straight back to the main process on one result queue. Only DiT jobs travel stage to stage, so
an encoded latent never waits behind a DiT or decode call on another core.

--steps 1 skips the second denoising pass: row 0's x0 is decoded directly (experimental; quality not measured).

Per chunk: 4 pixel frames -> encoder -> latent (rescaled from TAEHV's normalised latents to the raw latents the
pipeline's default encode produces: lat * std + mean) -> noise mix -> DiT row 0 -> re-noise -> DiT row 1 ->
decoder -> 4 pixel frames. TAEHV chunk f is pixel frames 4f..4f+3 (3 frames of lookahead relative to the Wan
encoder); decoded latent f is pixel frames 4f-3..4f, so the decoder's first 3 frames are dropped.

Warm-up (not timed): KV caches from g3_pipeline.py warm; encoder state from pixel chunks 0-2; decoder state from
the 3 warm-up latents. The timed loop starts at latent frame 3, like g3_pipeline.py.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

import lnc
from block_port import ART_DIR, MASK_NEG, BlockCfg, rope_cos_sin
from g2_common import G2, OUT, STEPS, to_u8, write_mp4
from g3_pipeline import SHM, load_stage, monitor_parse, monitor_start, stage_context
from g3_stage import STAGES, stage_cfg

BF16 = torch.bfloat16
VAE_DIR = ART_DIR / "vae"
# Wan VAE latent statistics (repos/StreamDiffusionV2/models/wan/wan_wrapper.py:82-89)
MEAN = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                     0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, 16, 1, 1)
STD = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, 16, 1, 1)


def load_vae(kind, mode):
    """One TAEHV half for a stage process: (graph, mems). mems is None when the memories are resident on the device
    (vae_resident.py artifact), else the host-side memory list that every call carries in and out."""
    from vae_neuron import artifact, compile_meta
    from vae_resident import load_resident, resident_artifact
    if mode == "resident" or (mode == "auto" and resident_artifact(kind).exists()):
        return load_resident(kind), None
    graph = torch.jit.load(str(artifact(kind)))
    shapes = json.loads(compile_meta(kind).read_text())["input_shapes"][1:]
    return graph, [torch.zeros(s, dtype=BF16) for s in shapes]


def worker(stage, core, a, b, kinds, steps, vae_mode, q_in, q_next, q_done, ready):
    """One stage process. DiT jobs go on to the next stage (or back to main after the last one); TAEHV jobs only
    ever arrive at the stage that hosts that half and go straight back to main."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    import torch_neuronx  # noqa: F401
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    ctx, extra = stage_context()
    rows = [load_stage(stage, a, b, r) for r in range(steps)]  # 1-step mode never runs row 1
    vaes = {kind: load_vae(kind, vae_mode) for kind in kinds}
    ready.put((stage, {k: "host" if mems is not None else "resident" for k, (_, mems) in vaes.items()}))
    while True:
        job = q_in.get()
        if job is None:
            if q_next is not None:
                q_next.put(None)
            return
        t = time.perf_counter()
        if job["kind"] == "dit":
            job["x"] = rows[job["row"]](job["x"], job["e0"], ctx, job["cos"], job["sin"], job["bias"], *extra)[0].cpu()
            dest = q_done if q_next is None else q_next
        elif job["kind"] in vaes:
            graph, mems = vaes[job["kind"]]
            if mems is None:
                job["x"] = graph(job["x"])[0].cpu()
            else:
                res = graph(job["x"], *mems)
                vaes[job["kind"]] = (graph, list(res[1:]))
                job["x"] = res[0].cpu()
            dest = q_done
        else:
            raise RuntimeError(f"stage {stage} got a {job['kind']} job but hosts {sorted(vaes)}")
        job["times"].append((stage, job["kind"], time.perf_counter() - t))
        dest.put(job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", type=int, default=120)
    ap.add_argument("--inflight", type=int, default=6, help="chunks in flight")
    ap.add_argument("--name", default="e2e_4core")
    ap.add_argument("--clip", default=None, help="512x512 clip to stylise once, no looping (warm with g5_clip.py first)")
    ap.add_argument("--out", default=None, help="output video for --clip (stylised frames only)")
    ap.add_argument("--cores", default=None, help="one core per stage, comma separated (default 0..N-1)")
    ap.add_argument("--enc-stage", type=int, default=None, help="stage hosting the TAEHV encoder (default N-2)")
    ap.add_argument("--dec-stage", type=int, default=None, help="stage hosting the TAEHV decoder (default N-1)")
    ap.add_argument("--vae", choices=["auto", "resident", "host"], default="auto",
                    help="TAEHV memories on the device (resident) or through the host; auto = resident if compiled")
    ap.add_argument("--steps", type=int, choices=[1, 2], default=2,
                    help="denoising passes per chunk; 1 decodes row 0 directly (experimental)")
    args = ap.parse_args()
    from g2_neuron import Host
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.manual_seed(1234)
    cfg = BlockCfg()
    host = Host(cfg, torch.load(G2 / "prompt_anime.pt")["prompt_embeds"])
    warm = torch.load(SHM / "warm.pt")
    fps_out, scale_state = 15, [warm.get("noise_scale", 0.8)]
    if args.clip:
        from g5_clip import load_frames, step_for
        assert warm.get("clip") == args.clip, f"caches in {SHM} were warmed for {warm.get('clip')}"
        x01, fps_out = load_frames(args.clip)
        args.chunks, calls = x01.shape[0] // 4 - 3, None
    else:
        ref = torch.load(G2 / "ref.pt")
        calls = ref["calls"]
        x01 = ref["frames"][0].permute(1, 0, 2, 3) * 0.5 + 0.5  # [29, 3, H, W] in [0, 1]
    n_pix = x01.shape[0]

    def pix_chunk(f):  # TAEHV chunk f = pixel frames 4f..4f+3; the clip is looped
        return torch.stack([x01[(4 * f + k) % n_pix] for k in range(4)]).to(BF16)

    n_st = len(STAGES)
    cores = [int(c) for c in args.cores.split(",")] if args.cores else list(range(n_st))
    enc_stage = max(n_st - 2, 0) if args.enc_stage is None else args.enc_stage
    dec_stage = n_st - 1 if args.dec_stage is None else args.dec_stage
    if len(cores) != n_st or not (0 <= enc_stage < n_st and 0 <= dec_stage < n_st):
        raise ValueError(f"{n_st} stages {STAGES}: need {n_st} cores (got {cores}) and VAE stages in 0..{n_st - 1}")
    kinds = [[] for _ in range(n_st)]
    kinds[enc_stage].append("enc")
    kinds[dec_stage].append("dec")

    ctxm = mp.get_context("spawn")
    qs = [ctxm.Queue() for _ in range(n_st)]
    q_done = ctxm.Queue()
    ready = ctxm.Queue()
    procs = [ctxm.Process(target=worker, args=(s, cores[s], a, b, kinds[s], args.steps, args.vae, qs[s],
                                               qs[s + 1] if s + 1 < n_st else None, q_done, ready), daemon=True)
             for s, (a, b) in enumerate(STAGES)]
    t0 = time.time()
    for p in procs:
        p.start()
    vae_state = {}
    for _ in procs:
        vae_state.update(ready.get(timeout=900)[1])
    print(f"{n_st} stage processes ready in {time.time() - t0:.0f}s; layers {STAGES} on cores {cores}; "
          f"TAEHV enc on stage {enc_stage}, dec on stage {dec_stage}, state {vae_state}", flush=True)

    def vae_job(kind, x, tag):
        qs[enc_stage if kind == "enc" else dec_stage].put({"kind": kind, "x": x, "tag": tag, "times": []})

    # ---- warm-up, not timed: encoder state from chunks 0-2, decoder state from the 3 warm-up latents ----
    out_frames = []
    for f in range(3):
        vae_job("enc", pix_chunk(f), ("warm", f))
        q_done.get(timeout=300)
    warm_lat = torch.cat([warm["finals"][0], warm["finals"][1]], dim=1)[0]  # [3, 16, 64, 64]
    for f in range(3):
        vae_job("dec", warm_lat[f:f + 1].to(BF16), ("warm", f))
        fr = q_done.get(timeout=300)["x"].float()
        out_frames.append(fr[3:] if f == 0 else fr)

    def bias_for(call_idx):
        b = torch.zeros(cfg.num_kv_cache)
        b[3:3 + max(0, 2 - call_idx)] = MASK_NEG
        return b.repeat_interleave(cfg.frame_tokens).view(1, 1, 1, -1).to(BF16)

    meta = {}

    def submit_dit(row, j, lat, step):
        e, e0 = host.time(step, 1)
        cos_, sin_ = (t.to(BF16) for t in rope_cos_sin(cfg, 3 + j))
        meta[(row, j)] = {"lat": lat, "step": step, "e": e}
        qs[0].put({"kind": "dit", "row": row, "tag": j, "x": host.patch(lat).to(BF16), "e0": e0.to(BF16),
                   "cos": cos_, "sin": sin_, "bias": bias_for(j), "times": []})

    n = args.chunks
    t_submit, t_done, per_kind = {}, {}, {}
    active, next_j, done = 0, 0, 0
    mon_path = SHM / f"monitor_{args.name}.jsonl"
    mon = monitor_start(mon_path)
    time.sleep(2.5)
    t_start_wall = time.time()
    while done < n:
        while active < args.inflight and next_j < n:
            t_submit[next_j] = time.perf_counter()
            vae_job("enc", pix_chunk(3 + next_j), next_j)
            active, next_j = active + 1, next_j + 1
        job = q_done.get(timeout=300)
        j, kind = job["tag"], job["kind"]
        for stage, k, dt in job["times"]:
            per_kind.setdefault(f"stage{stage}_{k}", []).append(dt)
        if kind == "enc":
            lat = (job["x"].float() * STD + MEAN).unsqueeze(0)  # raw-latent convention of the pipeline's encode
            if args.clip:  # adaptive noise scale from the real frames (streamv2v/inference.py:51-57)
                ns, step = step_for(x01, 3 + j, scale_state[0])
                scale_state[0] = ns
            else:
                c = calls[1 + (j % (len(calls) - 1))]
                ns, step = c["noise_scale"], c["current_step"]
            submit_dit(0, j, torch.randn_like(lat) * ns + lat * (1 - ns), step)
        elif kind == "dit":
            m = meta.pop((job["row"], j))
            x0 = host.to_x0(host.head(job["x"].float(), m["e"], 1), m["lat"], m["step"])
            if job["row"] == 0 and args.steps == 2:
                submit_dit(1, j, host.renoise(x0, torch.randn_like(x0), STEPS[1]), STEPS[1])
            else:
                vae_job("dec", x0[0].to(BF16), j)
        else:
            t_done[j] = time.perf_counter()
            if j < 5 or args.clip:
                out_frames.append(job["x"].float())
            active, done = active - 1, done + 1
    t_end_wall = time.time()
    time.sleep(1.5)
    mon.terminate()
    qs[0].put(None)

    skip = min(5, n // 4)
    order = sorted(t_done.values())
    steady_s = order[-1] - order[skip - 1]
    lat_ms = [(t_done[j] - t_submit[j]) * 1e3 for j in range(skip, n)]
    res = {"name": args.name, "chunks": n, "chunks_in_flight": args.inflight, "skipped_for_steady_state": skip,
           "steady_state_fps": round(4 * (n - skip) / steady_s, 2),
           "steady_state_s_per_chunk": round(steady_s / (n - skip), 4),
           "chunk_latency_ms": {"mean": round(float(np.mean(lat_ms)), 1), "p50": round(float(np.median(lat_ms)), 1),
                                "p99": round(float(np.percentile(lat_ms, 99)), 1), "max": round(max(lat_ms), 1)},
           "call_ms_mean": {k: round(1e3 * float(np.mean(v)), 2) for k, v in sorted(per_kind.items())},
           "vae_placement": {"enc": f"stage {enc_stage} / core {cores[enc_stage]}",
                             "dec": f"stage {dec_stage} / core {cores[dec_stage]}"},
           "vae_state": vae_state, "stage_layers": STAGES, "cores": cores, "steps": args.steps,
           "text_slots": stage_cfg().text_slots, "lnc": lnc.lnc(),
           "monitor": monitor_parse(mon_path, t_start_wall, t_end_wall)}
    (ART_DIR / f"g4_{args.name}.json").write_text(json.dumps(res, indent=2) + "\n")
    print("E2E " + json.dumps(res), flush=True)

    if args.clip:
        out = to_u8(torch.cat(out_frames).mul(2).sub(1))
        write_mp4(args.out, out, fps=round(fps_out))
        print(f"CLIP OK {len(out)} frames -> {args.out}", flush=True)
        for p in procs:
            p.join(timeout=20)
        return
    if args.name != "e2e_4core":  # only the default run writes the video
        for p in procs:
            p.join(timeout=20)
        return
    out = to_u8(torch.cat(out_frames).mul(2).sub(1))  # decoder output is [0, 1]
    inp = to_u8(ref["frames"][0].permute(1, 0, 2, 3)[:len(out)])
    write_mp4(OUT / "e2e_4core.mp4", np.concatenate([inp, out], axis=2))
    import cv2
    idx = [0, 8, 16, len(out) - 1]
    sheet = np.concatenate([np.concatenate([v[i] for i in idx], axis=1) for v in (inp, out)], axis=0)
    cv2.imwrite(str(OUT / "e2e_contact.png"), cv2.cvtColor(sheet[::2, ::2], cv2.COLOR_RGB2BGR))
    print(f"VIDEO OK {len(out)} frames", flush=True)
    for p in procs:
        p.join(timeout=20)


if __name__ == "__main__":
    main()

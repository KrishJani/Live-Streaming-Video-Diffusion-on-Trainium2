"""4-stage layer pipeline with resident KV caches: one process per stage, each pinned to a NeuronCore.

    python g3_pipeline.py warm                          session start + sink fill on the host path, dump cache states
    python g3_pipeline.py run --cores 0,1,2,3           4-core pipeline, chunks fed back to back
    python g3_pipeline.py run1 --name 1core             same graphs in one process on one core, sequential
                                                        (several processes cannot share a NeuronCore)

Stages: layers 0-7, 8-15, 16-22, 23-29 by default; LIVEVID_STAGES changes the split and the number of stages
(g3_stage.py), and must be the same for compile, warm and run. --cores defaults to one core per stage, in order.
Each stage process holds TWO instances of its graph, one
per denoising row (row 0 = first step, row 1 = step 500), because every row has its own caches. Only the hidden
state x (3 MB bf16) and the small per-call inputs (time modulation, RoPE, bias) move between stages.

Warm-up is on the host, as allowed: `warm` runs the session start (CPU port) and the first stream frame of each
row (frame 2, the last sink frame) on the single-layer graphs with host-side caches, then saves every layer's K/V.
The stage processes load those as the initial resident state. From frame 3 on the caches never leave the device.

Chunk j of the pipeline = latent frame 3 + j. Its row-0 job can start any time; its row-1 job needs row 0's
output for the same frame (re-noised to step 500), exactly as in the repo's batched stream schedule.
"""
import argparse
import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import torch
import torch.multiprocessing as mp

import lnc
from block_port import ART_DIR, MASK_NEG, BlockCfg, compact_context, cosine, max_abs, rope_cos_sin
from g2_common import G2, OUT, STEPS
from g3_stage import STAGES, stage_cfg, stage_path

BF16 = torch.bfloat16
SHM = Path("/dev/shm/g3")


def warm_path(row, a, b):
    """Warmed K/V of layers a..b-1 for one denoising row. Named by layer range, so a run whose LIVEVID_STAGES
    differs from the warm-up's fails loudly instead of loading the wrong layers."""
    return SHM / f"row{row}_layers{a:02d}_{b:02d}.pt"


def save_warm_caches(rows):
    """rows[r][layer] is that layer's CacheState; writes one file per (row, stage) of the current STAGES."""
    SHM.mkdir(parents=True, exist_ok=True)
    for r, states in enumerate(rows):
        for a, b in STAGES:
            flat = []
            for st in states[a:b]:
                flat += [st.k.contiguous(), st.v.contiguous()]
            torch.save(flat, warm_path(r, a, b))


def stage_context(cfg=None):
    """The text context the stage graphs take, plus any extra trailing graph inputs.

    Full context: (ctx [1, 512, dim], ()). Compact (LIVEVID_TEXT_SLOTS > 0): (ctx [1, slots, dim], (logw,)).
    """
    cfg = cfg or stage_cfg()
    ctx = torch.load(SHM / "ctx.pt")
    if not cfg.text_slots:
        return ctx, ()
    ctx, logw = compact_context(ctx, cfg.text_slots)
    return ctx, (logw,)


def warm():
    from g2_neuron import Runner
    ref = torch.load(G2 / "ref.pt")
    run = Runner("neuron", ref)
    x0s = run.start(ref["start"]["noisy"], ref["start"]["noises"])
    c0, c1 = ref["calls"][0], ref["calls"][1]
    x0_r0 = run.forward_row(0, c0["noisy"][:, :1], c0["current_step"], 2)
    hidden = run.host.renoise(x0_r0, c0["noise"].view_as(x0_r0), STEPS[1])
    x0_r1 = run.forward_row(1, hidden, STEPS[1], 2)
    for r in (0, 1):
        assert run.rows[r][0].frames == [0, 1, 2, None, None, None]
    save_warm_caches(run.rows)
    torch.save(run.ctx, SHM / "ctx.pt")
    torch.save({"finals": [x0s[-1], x0_r1], "frame2_cos": cosine(x0_r1, c1["x0"][1])}, SHM / "warm.pt")
    print(f"WARM OK frame-2 final cos vs ref {cosine(x0_r1, c1['x0'][1]):.6f}", flush=True)


def load_stage(stage, a, b, row):
    """Load one stage graph for one row with its warmed K/V as resident state.

    The state is assigned on the host and then moved with move_trace_to_device, before any call is made.
    """
    import torch_neuronx
    m = torch.jit.load(str(stage_path(a, b)))
    path = warm_path(row, a, b)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the warm-up (g3_pipeline.py warm or g5_clip.py) with the "
                                f"same LIVEVID_STAGES as this run ({STAGES})")
    state = torch.load(path)  # saved as k0, v0, k1, v1, ...
    # torch_neuronx re-keys the alias map in module-parameter order while tracing (hlo_conversion.py
    # move_state_to_device), so the traced states are k0..kN-1 followed by v0..vN-1, not interleaved.
    state = state[0::2] + state[1::2]
    names = list(m.states._parameters.keys())
    assert len(names) == len(state), (len(names), len(state))
    for name, t in zip(names, state):
        m.states._parameters[name] = t
    torch_neuronx.move_trace_to_device(m, 0)
    return m


def run_one_process(args):
    """Same 4 graphs, one process, one core, strictly sequential (row 0 then row 1 for every chunk)."""
    os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
    from g2_neuron import Host
    cfg = BlockCfg()
    ref = torch.load(G2 / "ref.pt")
    host = Host(cfg, torch.load(G2 / "prompt_anime.pt")["prompt_embeds"])
    warm_data = torch.load(SHM / "warm.pt")
    calls = ref["calls"]
    ctx, extra = stage_context()
    t0 = time.time()
    graphs = [[load_stage(s, a, b, r) for s, (a, b) in enumerate(STAGES)] for r in (0, 1)]
    print(f"8 graphs loaded on core 0 in {time.time() - t0:.0f}s", flush=True)

    def bias_for(call_idx):
        b = torch.zeros(cfg.num_kv_cache)
        b[3:3 + max(0, 2 - call_idx)] = MASK_NEG
        return b.repeat_interleave(cfg.frame_tokens).view(1, 1, 1, -1).to(BF16)

    def dit(row, j, lat, step):
        e, e0 = host.time(step, 1)
        cos_, sin_ = (t.to(BF16) for t in rope_cos_sin(cfg, 3 + j))
        x, e0, bias = host.patch(lat).to(BF16), e0.to(BF16), bias_for(j)
        t = time.perf_counter()
        for g in graphs[row]:
            x = g(x, e0, ctx, cos_, sin_, bias, *extra)[0].cpu()
        dit_s.append(time.perf_counter() - t)
        return host.to_x0(host.head(x.float(), e, 1), lat, step)

    n, parity, chunk_s, dit_s = args.chunks, [], [], []
    for j in range(n):
        c = calls[1 + (j % (len(calls) - 1))]
        t = time.perf_counter()
        x0_r0 = dit(0, j, c["noisy"][:, :1], c["current_step"])
        x0_r1 = dit(1, j, host.renoise(x0_r0, c["noise"].view_as(x0_r0), STEPS[1]), STEPS[1])
        chunk_s.append(time.perf_counter() - t)
        if j < len(calls) - 1:
            row = {"chunk": j, "frame": 3 + j, "row0_cos": cosine(x0_r0, calls[j + 1]["x0"][0])}
            if j + 2 < len(calls):
                row["final_cos"] = cosine(x0_r1, calls[j + 2]["x0"][1])
                row["final_max_abs"] = max_abs(x0_r1, calls[j + 2]["x0"][1])
            parity.append(row)
            print("chunk", json.dumps(row), f"{chunk_s[-1] * 1e3:.0f} ms", flush=True)
    skip = min(5, n // 4)
    steady = chunk_s[skip:]
    finals = [c["final_cos"] for c in parity if "final_cos" in c]
    res = {"name": args.name, "cores": [0], "mode": "one process, sequential", "chunks": n,
           "skipped_for_steady_state": skip, "steady_state_s_per_chunk": round(sum(steady) / len(steady), 4),
           "steady_state_fps": round(4 * len(steady) / sum(steady), 2),
           "chunk_latency_ms": {"mean": round(1e3 * sum(steady) / len(steady), 1),
                                "p50": round(1e3 * sorted(steady)[len(steady) // 2], 1), "max": round(1e3 * max(steady), 1)},
           "dit_ms_per_pass_mean": round(1e3 * sum(dit_s[2 * skip:]) / len(dit_s[2 * skip:]), 2),
           "parity": parity, "frame2_final_cos_warm": warm_data["frame2_cos"],
           "min_final_cos": min(finals + [warm_data["frame2_cos"]])}
    res["pass"] = bool(res["min_final_cos"] >= 0.999)
    (ART_DIR / f"g3_pipeline_{args.name}.json").write_text(json.dumps(res, indent=2) + "\n")
    print("PIPELINE " + json.dumps(res), flush=True)


def worker(stage, core, a, b, q_in, q_out, ready):
    os.environ["NEURON_RT_VISIBLE_CORES"] = str(core)
    import torch_neuronx
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    ctx, extra = stage_context()
    rows = [load_stage(stage, a, b, r) for r in (0, 1)]
    ready.put(stage)
    while True:
        job = q_in.get()
        if job is None:
            q_out.put(None)
            return
        jid, row, x, e0, cos_, sin_, bias, times = job
        t = time.perf_counter()
        y = rows[row](x, e0, ctx, cos_, sin_, bias, *extra)[0].cpu()
        q_out.put((jid, row, y, e0, cos_, sin_, bias, times + [time.perf_counter() - t]))


def monitor_start(path):
    cfg = {"period": "1s", "neuron_runtimes": [{"tag_filter": ".*", "metrics": [{"type": "neuroncore_counters"}]}]}
    cfg_path = SHM / "monitor.json"
    cfg_path.write_text(json.dumps(cfg))
    return subprocess.Popen(["neuron-monitor", "-c", str(cfg_path)], stdout=open(path, "w"),
                            stderr=subprocess.DEVNULL)


def monitor_parse(path, t_start, t_end):
    """Mean utilisation per neuron-monitor core index over samples taken inside [t_start, t_end] (unix seconds)."""
    acc = {}
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        per = {}
        for rt in d.get("neuron_runtime_data") or []:
            rep = (rt.get("report") or {}).get("neuroncore_counters") or {}
            ts = rep.get("period_end") or rt.get("period_end")
            for idx, v in (rep.get("neuroncores_in_use") or {}).items():
                per[idx] = per.get(idx, 0.0) + float(v.get("neuroncore_utilization", 0.0))
        acc.setdefault("samples", []).append(per)
    samples = acc.get("samples", [])
    core = samples[2:-1] if len(samples) > 5 else samples  # drop ramp-up / tail samples
    idxs = sorted({k for s in core for k in s}, key=int)
    return {"num_samples": len(core),
            "mean_utilization_by_monitor_index": {i: round(sum(s.get(i, 0.0) for s in core) / max(len(core), 1), 1)
                                                  for i in idxs}}


def run(args):
    from g2_neuron import Host
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    cfg = BlockCfg()
    cores = [int(c) for c in args.cores.split(",")] if args.cores else list(range(len(STAGES)))
    if len(cores) != len(STAGES):
        raise ValueError(f"--cores lists {len(cores)} cores for {len(STAGES)} stages ({STAGES})")
    ref = torch.load(G2 / "ref.pt")
    host = Host(cfg, torch.load(G2 / "prompt_anime.pt")["prompt_embeds"])
    warm_data = torch.load(SHM / "warm.pt")
    calls = ref["calls"]

    ctxm = mp.get_context("spawn")
    qs = [ctxm.Queue() for _ in range(len(STAGES) + 1)]
    ready = ctxm.Queue()
    procs = [ctxm.Process(target=worker, args=(s, cores[s], a, b, qs[s], qs[s + 1], ready), daemon=True)
             for s, (a, b) in enumerate(STAGES)]
    t0 = time.time()
    for p in procs:
        p.start()
    for _ in procs:
        ready.get(timeout=900)
    print(f"{len(procs)} stage processes ready on cores {cores} in {time.time() - t0:.0f}s", flush=True)

    n = args.chunks

    def inputs_r0(j):  # chunk j = frame 3 + j; loop the reference's stream inputs after they run out
        c = calls[1 + (j % (len(calls) - 1))]
        return c["noisy"][:, :1], c["current_step"], c["noise"]

    def bias_for(call_idx):  # slots 3 and 4 are still empty for a row's first resident call, slot 3 for its second
        b = torch.zeros(cfg.num_kv_cache)
        b[3:3 + max(0, 2 - call_idx)] = MASK_NEG
        return b.repeat_interleave(cfg.frame_tokens).view(1, 1, 1, -1).to(BF16)

    def submit(row, j, lat, step):
        e, e0 = host.time(step, 1)
        cos_, sin_ = (t.to(BF16) for t in rope_cos_sin(cfg, 3 + j))
        meta[(row, j)] = {"lat": lat, "step": step, "e": e, "t_submit": time.perf_counter()}
        qs[0].put(((row, j), row, host.patch(lat).to(BF16), e0.to(BF16), cos_, sin_, bias_for(j), []))

    meta, x0 = {}, {}
    pending, ready_r1, inflight, done_r1 = deque(range(n)), deque(), 0, []
    stage_times = [[] for _ in STAGES]
    mon_path = SHM / f"monitor_{args.name}.jsonl"
    mon = monitor_start(mon_path)
    time.sleep(2.5)
    t_start_wall, t_start = time.time(), time.perf_counter()
    while len(done_r1) < n:
        while inflight < args.inflight and (ready_r1 or pending):
            if ready_r1:
                j, lat = ready_r1.popleft()
                submit(1, j, lat, STEPS[1])
            else:
                j = pending.popleft()
                lat, step, _ = inputs_r0(j)
                submit(0, j, lat, step)
            inflight += 1
        (row, j), _, y, *_rest, times = qs[-1].get(timeout=300)
        inflight -= 1
        for s, dt in enumerate(times):
            stage_times[s].append(dt)
        m = meta[(row, j)]
        out = host.to_x0(host.head(y.float(), m["e"], 1), m["lat"], m["step"])
        x0[(row, j)] = out
        m["t_done"] = time.perf_counter()
        if row == 0:
            ready_r1.append((j, host.renoise(out, inputs_r0(j)[2].view_as(out), STEPS[1])))
        else:
            done_r1.append(time.perf_counter())
    t_end, t_end_wall = time.perf_counter(), time.time()
    time.sleep(1.5)
    mon.terminate()
    qs[0].put(None)

    skip = min(5, n // 4)  # steady state: ignore the first chunks while the pipeline fills
    steady_s = done_r1[-1] - done_r1[skip - 1]
    fps = 4 * (n - skip) / steady_s
    lat_ms = [(meta[(1, j)]["t_done"] - meta[(0, j)]["t_submit"]) * 1e3 for j in range(skip, n)]
    parity = []
    for j in range(min(n, len(calls) - 1)):
        row = {"chunk": j, "frame": 3 + j, "row0_cos": cosine(x0[(0, j)], calls[j + 1]["x0"][0])}
        if j + 2 < len(calls):
            row["final_cos"] = cosine(x0[(1, j)], calls[j + 2]["x0"][1])
            row["final_max_abs"] = max_abs(x0[(1, j)], calls[j + 2]["x0"][1])
        parity.append(row)
    finals = [c["final_cos"] for c in parity if "final_cos" in c]
    res = {"name": args.name, "cores": cores, "inflight": args.inflight, "chunks": n, "skipped_for_steady_state": skip,
           "steady_state_fps": round(fps, 2), "steady_state_s_per_chunk": round(steady_s / (n - skip), 4),
           "total_s": round(t_end - t_start, 3),
           "chunk_latency_ms": {"mean": round(sum(lat_ms) / len(lat_ms), 1), "p50": round(sorted(lat_ms)[len(lat_ms) // 2], 1),
                                "max": round(max(lat_ms), 1)},
           "stage_call_ms_mean": [round(1e3 * sum(t) / len(t), 2) for t in stage_times],
           "stage_layers": STAGES, "text_slots": stage_cfg().text_slots, "lnc": lnc.lnc(), "parity": parity, "frame2_final_cos_warm": warm_data["frame2_cos"],
           "min_final_cos": min(finals + [warm_data["frame2_cos"]]),
           "monitor": monitor_parse(mon_path, t_start_wall, t_end_wall)}
    res["pass"] = bool(res["min_final_cos"] >= 0.999)
    (ART_DIR / f"g3_pipeline_{args.name}.json").write_text(json.dumps(res, indent=2) + "\n")
    print("PIPELINE " + json.dumps(res), flush=True)
    for p in procs:
        p.join(timeout=20)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["warm", "run", "run1"])
    ap.add_argument("--cores", default=None, help="one core per stage, comma separated (default 0..N-1)")
    ap.add_argument("--inflight", type=int, default=6)
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--name", default="4core")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    {"warm": warm, "run": lambda: run(a), "run1": lambda: run_one_process(a)}[a.mode]()

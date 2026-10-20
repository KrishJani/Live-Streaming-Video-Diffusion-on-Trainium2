# Pipeline options: throughput and latency changes, and how to measure them

These changes target the bottlenecks identified in [tier2-overview.md](tier2-overview.md) and
[tier1-profile.md](tier1-profile.md). Every option defaults to the previous behaviour or is a pure fix, so existing
artifacts and commands keep working. **None of the effects below has been measured on Trainium2 yet.** Only the
compact-context exactness was checked, on CPU. Every expected gain is labelled a projection, worked out from
numbers already in this repository.

| # | Change | How to turn it on | Status |
|---|---|---|---|
| 1 | First-stage input made contiguous | always on (`Host.patch`) | pure fix, values unchanged; effect not measured |
| 2 | TAEHV memories resident on the device in the end-to-end run | `vae_resident.py`, then `g4_e2e.py` (`--vae auto`) | not measured |
| 3 | Finished jobs go straight back to the main process | always on (`g4_e2e.py`) | not measured |
| 4 | Configurable stage split, core list and VAE placement | `LIVEVID_STAGES`, `--cores`, `--enc-stage`, `--dec-stage` | not measured |
| 5 | Compact cross-attention context, exact | `LIVEVID_TEXT_SLOTS=64` (stage graphs only) | exact on CPU (fp32 cosine 0.9999999999998); Neuron not measured |
| 6 | LNC=1 builds (8 cores per chip) for both tiers | `NEURON_LOGICAL_NC_CONFIG=1`, `LIVEVID_CORES` (Tier 1) | not measured |
| 7 | 1-step mode | `g4_e2e.py --steps 1` | experimental, output quality not checked |

Build options are environment variables. The same values must be exported for compile, warm-up and run. A
mismatch fails loudly: stage artifacts and warmed caches are named by layer range, context size and LNC.

## 1. First-stage input made contiguous

The same 8-layer graph (layers 0 to 7) takes 34.96 ms when called on its own (tier2-overview section 4). Inside
the pipeline it takes 43.6 to 44.1 ms (sections 6 and 6b), and it is the DiT bottleneck. One difference between
stage 0 and the other stages is its input.

- `Host.patch` returned `x.flatten(2).transpose(1, 2)`, a strided view. `.to(bfloat16)` keeps those strides.
- So stage 0 received a non-contiguous 3 MB tensor, and had to repack it on a worker limited to one CPU thread.
- Stages 1 to 3 receive contiguous `.cpu()` outputs.
- `Host.patch` now returns `.contiguous()`. The values are identical.

This is a hypothesis for the 9 ms gap, not a measured cause. **Check:** the end-to-end JSON field
`call_ms_mean.stage0_dit`, against 43.6 ms before the change. If the gap remains, the next suspect is CPU
contention: the pod is capped at 11 CPUs, shared by the main process (4 threads), the stage workers and
neuron-monitor.

## 2. Resident TAEHV memories in the end-to-end run

`g4_e2e.py` used the TAEHV graphs whose 9 memories are graph inputs and outputs. That moves 33 MB (encoder) and
44 MB (decoder) across the host in each direction on every call. The decoder runs on core 3, the end-to-end
bottleneck: 2 x 31.4 ms of DiT plus 38.9 ms of decode is 102 ms of the 104 ms period.

`vae_resident.py` already traced graphs that keep the memories on the device, but it never saved them. Now:

- it saves them as `artifacts/tier2/vae/taehv_<enc|dec>_512x512_resident_bf16.pt`, still checking parity
  against CPU TAEHV;
- `g4_e2e.py --vae auto` (the default) uses them when present, and otherwise falls back to the host-state graphs
  (`--vae host` forces the old behaviour);
- the result JSON records which path ran, in `vae_state`.

**Projection:** in the G1 profile, about 84 MB in and out per call cost 7.2 ms including call overhead. The
decoder moves 88 MB per call, so several milliseconds should come off the 38.9 ms decode, on the bottleneck core.

## 3. Direct result queue

Before, each stage process forwarded every job it did not own to the next stage. So an encoded latent from the
encoder stage queued behind whatever the last stage was running: a 31 ms DiT call or a 39 ms decode. Only then
did it reach the main process, and that delay sits directly before the chunk's first DiT pass.

Now TAEHV results and the last stage's DiT outputs go to one result queue read by the main process. Only DiT
jobs travel stage to stage. Chunk order is unchanged, because each TAEHV half is still fed in chunk order by one
FIFO queue.

**Check:** `chunk_latency_ms` against 629 ms mean (6 chunks in flight) and 458 ms (3 in flight). Throughput
should not change.

## 4. Stage split, cores and VAE placement

- `LIVEVID_STAGES` sets the layer ranges, for example `0-9,9-18,18-25,25-30`. Any number of stages works, as long
  as the ranges tile layers 0 to 29 in order.
- Default: `0-8,8-16,16-23,23-30`, the layout every existing number was measured with.
- `g3_stage.py list` prints the configured stages.
- `g3_compile_all.sh` compiles whatever is configured (`MAX_JOBS` at a time, default 4).
- `g3_pipeline.py` and `g4_e2e.py` default `--cores` to one core per stage.
- `g4_e2e.py --enc-stage / --dec-stage` choose which stage processes host the TAEHV halves. Defaults: the
  second-to-last and the last stage.

The 8/8/7/7 split balances the DiT alone. End to end, cores 2 and 3 also run the encoder (19 ms per chunk) and the
decoder (39 ms per chunk).

**Projection:**

| | Value |
|---|---|
| Measured cost per layer per chunk (2 passes) | about 8.96 ms |
| Total work per chunk | 30 x 8.96 + 19 + 39 = 327 ms |
| Ideal period on 4 cores | about 82 ms per core |

A 9/9/7/5 split (`0-9,9-18,18-25,25-30`) gives these per-core loads:

| Core | Layers | Load per chunk |
|---|---|---|
| 0 | 9 | 80.6 ms |
| 1 | 9 | 80.6 ms |
| 2 | 7, plus encode | 81.7 ms |
| 3 | 5, plus decode | 83.7 ms |

That is about 84 ms per chunk, or **about 47 FPS** end to end against the measured 38.4. It assumes stage 0 loses
its extra 9 ms (change 1). If it doesn't, a 9-layer first stage would be the bottleneck at about 98 ms. So measure
`call_ms_mean` with the default split first, then pick the split from the measured per-stage costs.

With LNC=1 (change 6) the same variable takes 8 ranges, for example
`0-4,4-8,8-12,12-16,16-20,20-24,24-27,27-30`.

```bash
tier2/remote.sh push
tier2/remote.sh run "export LIVEVID_STAGES=0-9,9-18,18-25,25-30 && ./g3_compile_all.sh"
tier2/remote.sh run "export LIVEVID_STAGES=0-9,9-18,18-25,25-30 && NEURON_RT_VISIBLE_CORES=0 python -u g3_pipeline.py warm"
tier2/remote.sh run "export LIVEVID_STAGES=0-9,9-18,18-25,25-30 && python -u g4_e2e.py --chunks 120 --name e2e_9_9_7_5"
```

## 5. Compact cross-attention context (exact)

The repo zero-pads the prompt embedding to 512 tokens and cross-attends without a mask. The anime prompt has 29
real tokens, so 483 of the 512 context rows are padding. They all become the same vector after `text_embedding`,
so they have identical keys and values.

Softmax over n identical keys equals softmax over one copy with log(n) added to its score. So:

- `block_port.compact_context` collapses the context to one row per distinct row, plus fp32 log-counts;
- the stage graphs take a `[1, 64, 1536]` context and a `[1, 1, 1, 64]` fp32 weight instead of `[1, 512, 1536]`;
- scores and softmax of that branch run in fp32, so log(n) is not rounded to bf16;
- prompts with up to 64 distinct rows work without recompiling (`LIVEVID_TEXT_SLOTS` raises the limit).

CPU check, `python check_ctx_compact.py` (demo shapes, random weights; real weights are used automatically on the
pod):

| | Compact vs full context |
|---|---|
| fp32 block output y, cosine / max abs | 0.9999999999998 / 4.8e-7 (output max 4.7) |
| fp32 new K and V caches | identical (max abs 0) |
| fp32 cross-attention branch alone, cosine | 0.9999999999984 |
| bf16 on CPU, cosine vs the fp32 full-context output | 0.99999518 (full) vs 0.99999518 (compact) |
| Cross-attention GFLOP per block call | 8.05 -> 1.01 (block total 132.1) |
| Context bytes per graph call | 1.57 MB -> 0.20 MB |

Prompt lengths of 1 and 63 real tokens also pass. A context with more distinct rows than slots raises an error
instead of truncating.

**Projection:** up to about 5% of DiT time, because the change removes 5.3% of each block's FLOPs and most of its
context transfer. The single-layer warm-up graphs are unchanged; they produce the same caches either way.

```bash
tier2/remote.sh run "python -u check_ctx_compact.py --layer 0"     # real weights on the pod
tier2/remote.sh run "export LIVEVID_TEXT_SLOTS=64 && ./g3_compile_all.sh && ./g3_run_all.sh"
```

`g3_run_all.sh` runs the existing parity gate (final latent cosine >= 0.999 against the CPU reference), so the
compact graphs are gated the same way as the originals.

## 6. LNC=1

Trn2 defaults to LNC=2: 4 logical cores per chip, each made of two physical cores. The Tier 1 profile shows the two
physical cores of every logical core at about 95% and 61%, so half of each pair is underused.

`NEURON_LOGICAL_NC_CONFIG=1` exposes 8 single-physical-core devices.

- Compile and runtime must use the same value, so the one variable drives both: it adds
  `--logical-nc-config=1` to every compile (Tier 1 components, Tier 2 single-layer, stage and TAEHV graphs).
- LNC=1 artifacts get an `_lnc1` suffix, so they never mix with LNC=2 ones.
- Tier 1 workers come from `LIVEVID_CORES`, used by `server.py` and the multi-core run of `bench.py`.

Tier 1, 8 workers:

```bash
tier1/remote.sh "export NEURON_LOGICAL_NC_CONFIG=1 && cd tier1 && for c in taesd_enc unet taesd_dec; do python pipeline_neuron.py compile --component \$c; done"
tier1/remote.sh "export NEURON_LOGICAL_NC_CONFIG=1 LIVEVID_CORES=0,1,2,3,4,5,6,7 && cd tier1 && python bench.py"
```

Each frame will be slower on one physical core. The question is whether 8 workers deliver more than the measured
43.8 FPS. The core-index mapping that `server.py` uses for its utilisation display was only checked for 4 cores
under LNC=2.

Tier 2: compile with `NEURON_LOGICAL_NC_CONFIG=1` and an 8-range `LIVEVID_STAGES`. This adds pipeline
throughput, not latency.

## 7. 1-step mode

`g4_e2e.py --steps 1` decodes row 0's x0 directly and skips the second DiT pass, roughly halving DiT work per chunk.
Stage processes then load only the row-0 graphs.

The model is distilled for the 2-step schedule used everywhere else here, so output quality has to be judged
(`--clip ... --out ...`) before any FPS number from this mode is quoted.

## Not in this change

These need iteration on the hardware rather than a blind rewrite:

- RoPE position rewind and adaptive sink refresh for long or live streams;
- session start on Neuron;
- a live Tier 2 server;
- tensor parallelism to cut per-chunk latency;
- NKI attention kernels;
- batching the two denoising rows into one call.

"""Compile one pipeline stage: several consecutive blocks in one graph with their KV caches as resident state.

    python g3_stage.py 0 8      layers 0..7 -> artifacts/tier2/stage_00_08_<tag>_resident_bf16.pt
    python g3_stage.py list     print the configured stages, one "a b" pair per line (used by g3_compile_all.sh)

Initial state is zeros; the pipeline overwrites it with the warmed caches before moving it to the device.

Build options, read from the environment so that compile, warm-up and run agree (export them for all three):

    LIVEVID_STAGES=0-8,8-16,16-23,23-30   layer split, one range per stage process / NeuronCore (this is the default).
                                          Must cover layers 0..29 in order. Any number of stages is allowed.
    LIVEVID_TEXT_SLOTS=64                 compact cross-attention context (block_port.compact_context); 0 = off
    NEURON_LOGICAL_NC_CONFIG=1            compile for LNC=1 (lnc.py)
"""
import json
import os
import sys
import time

import torch

import lnc
from block_port import ART_DIR, BlockCfg, CausalBlockPort, rope_cos_sin
from g2_resident import COMPILER_ARGS, ResidentBlocks

BF16 = torch.bfloat16
NUM_LAYERS = 30
DEFAULT_STAGES = "0-8,8-16,16-23,23-30"


def parse_stages(spec, num_layers=NUM_LAYERS):
    """"0-8,8-16,..." -> [(0, 8), (8, 16), ...]; ranges are half-open and must tile 0..num_layers in order."""
    stages, expect = [], 0
    for part in spec.replace(" ", "").split(","):
        a, b = (int(v) for v in part.split("-"))
        if a != expect or b <= a:
            raise ValueError(f"LIVEVID_STAGES={spec!r}: range {a}-{b} must start at {expect} and be non-empty")
        stages.append((a, b))
        expect = b
    if expect != num_layers:
        raise ValueError(f"LIVEVID_STAGES={spec!r} ends at layer {expect}, expected {num_layers}")
    return stages


STAGES = parse_stages(os.environ.get("LIVEVID_STAGES", DEFAULT_STAGES))


def stage_cfg():
    return BlockCfg(text_slots=int(os.environ.get("LIVEVID_TEXT_SLOTS", "0")))


def stage_path(a, b, cfg=None):
    cfg = cfg or stage_cfg()
    return ART_DIR / f"stage_{a:02d}_{b:02d}_{cfg.tag}_resident_bf16{lnc.suffix()}.pt"


def main():
    if sys.argv[1] == "list":
        print("\n".join(f"{a} {b}" for a, b in STAGES))
        return
    import torch_neuronx
    a, b = int(sys.argv[1]), int(sys.argv[2])
    cfg = stage_cfg()
    out = stage_path(a, b, cfg)
    if out.exists():
        print("exists, skip")
        return
    ports = []
    for i in range(a, b):
        m = CausalBlockPort(cfg)
        m.load_state_dict(torch.load(ART_DIR / f"block{i}_fp32.pt")["block"], strict=True)
        ports.append(m.eval().requires_grad_(False))
    z = torch.zeros(1, cfg.l_cache, cfg.num_heads, cfg.head_dim)
    module = ResidentBlocks(ports, [(z, z)] * (b - a)).to(BF16)
    aliases = {}
    for i in range(b - a):
        aliases[module.k[i]] = 1 + 2 * i
        aliases[module.v[i]] = 2 + 2 * i
    cos_, sin_ = rope_cos_sin(cfg, 0)
    ex = tuple(t.to(BF16) for t in (torch.randn(1, cfg.l_chunk, cfg.dim), torch.randn(1, 1, 6, cfg.dim),
                                    torch.randn(1, cfg.ctx_len, cfg.dim), cos_, sin_,
                                    torch.zeros(1, 1, 1, cfg.l_cache)))
    if cfg.text_slots:  # log-count weights of the compact context stay fp32 (see CausalBlockPort._attend_counted)
        ex = ex + (torch.zeros(1, 1, 1, cfg.text_slots),)
    work = ART_DIR / "work" / out.stem
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(module, ex, input_output_aliases=aliases, compiler_workdir=str(work),
                                 compiler_args=COMPILER_ARGS)
    dt = time.perf_counter() - t0
    torch.jit.save(traced, str(out))
    (ART_DIR / f"{out.stem}_compile.json").write_text(json.dumps(
        {"layers": [a, b], "compile_s": round(dt, 1), "cfg": cfg.tag, "compiler_args": COMPILER_ARGS}) + "\n")
    print(f"STAGE {a}-{b} COMPILE OK {dt:.1f}s", flush=True)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

"""TAEHV with its 9 memories resident on the NeuronCore (same mechanism as g2_resident.py).

    python vae_resident.py              compile (or reuse) both graphs, then parity and latency on core 0
    python vae_resident.py enc          only the encoder (same for dec)

The memories are nn.Parameters of the traced module; `input_output_aliases` marks graph output 1 + i as the new
value of memory i, and `move_trace_to_device` keeps them on the device. Each call then only moves the 4 frames
or the 1 latent across the host boundary, instead of 33 MB (encoder) or 44 MB (decoder) of state each way.

The traced graphs are saved to artifacts/tier2/vae/taehv_<enc|dec>_512x512_resident_bf16.pt; g4_e2e.py loads them
with `load_resident` when present. Checks parity chunk by chunk against CPU TAEHV (a stale state would show up
from chunk 1 on), then latency over 100 calls including bringing the result back to the host.
"""
import json
import statistics
import sys
import time

import torch
import torch.nn as nn

import lnc
from vae_neuron import BF16, COMPILER_ARGS, VAE_DIR, StreamTAE, cos, example_inputs, load_taehv


class ResidentTAE(nn.Module):
    def __init__(self, inner, mems):
        super().__init__()
        self.inner = inner
        self.mems = nn.ParameterList([nn.Parameter(m.clone(), requires_grad=False) for m in mems])

    def forward(self, x):
        return self.inner(x, *self.mems)


def resident_artifact(which):
    return VAE_DIR / f"taehv_{which}_512x512_resident_bf16{lnc.suffix()}.pt"


def load_resident(which):
    """Load a saved resident graph, reset its memories to zero (a fresh stream) and move them to the device.

    Call with NEURON_RT_VISIBLE_CORES already set; the graph goes to the first visible core. Returns a module
    called as `y = m(x)[0]`; the memories advance on the device with every call.
    """
    import torch_neuronx
    m = torch.jit.load(str(resident_artifact(which)))
    for name, t in list(m.states._parameters.items()):
        m.states._parameters[name] = torch.zeros_like(t)
    torch_neuronx.move_trace_to_device(m, 0)
    return m


def compile_resident(which, tae):
    """Trace one resident graph and save it (skipped when the artifact exists). Returns compile seconds or None."""
    import torch_neuronx
    out = resident_artifact(which)
    if out.exists():
        print(f"{which}: {out.name} exists, skip compile", flush=True)
        return None
    inner = StreamTAE(tae.encoder if which == "enc" else tae.decoder).eval()
    ex = example_inputs(inner, which)  # memories are zeros
    module = ResidentTAE(inner, ex[1:]).to(BF16)
    aliases = {p: 1 + i for i, p in enumerate(module.mems)}
    work = VAE_DIR / "work" / out.stem
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    traced = torch_neuronx.trace(module, (ex[0].to(BF16),), input_output_aliases=aliases,
                                 compiler_workdir=str(work), compiler_args=COMPILER_ARGS)
    compile_s = time.perf_counter() - t0
    tmp = out.with_suffix(".pt.tmp")
    torch.jit.save(traced, str(tmp))
    tmp.replace(out)
    print(f"{which}: COMPILE OK {compile_s:.1f}s -> {out.name}", flush=True)
    return compile_s


def main():
    torch.set_grad_enabled(False)
    which_list = sys.argv[1:] or ["enc", "dec"]
    tae = load_taehv()
    r = torch.load(VAE_DIR / "ref.pt")
    chunks = {"enc": (r["x32"].split(4), [l for l in r["lat_tae"][0].split(1)]),
              "dec": ([l for l in r["lat_tae"][0].split(1)], r["dec_tae"][0].split(4))}
    res = {}
    for which in which_list:
        compile_s = compile_resident(which, tae)
        traced = load_resident(which)
        devices = sorted({str(p.device) for p in traced.states._parameters.values()})
        print(f"{which}: states on {devices}", flush=True)
        xs, refs = chunks[which]
        cosines = [cos(traced(x.to(BF16))[0].float().cpu(), y) for x, y in zip(xs, refs)]
        print(f"{which}: per-chunk cos vs CPU TAEHV {[round(c, 6) for c in cosines]}", flush=True)
        x = xs[1].to(BF16)
        ms = []
        for i in range(105):
            t = time.perf_counter()
            _ = traced(x)[0].cpu()
            if i >= 5:
                ms.append((time.perf_counter() - t) * 1e3)
        ms.sort()
        res[which] = {"compile_s": None if compile_s is None else round(compile_s, 1), "state_devices": devices,
                      "cos_per_chunk": cosines, "min_cos": min(cosines), "mean_ms": statistics.mean(ms),
                      "p50_ms": ms[50], "p99_ms": ms[98], "artifact": str(resident_artifact(which))}
        print(which, json.dumps(res[which]), flush=True)
    (VAE_DIR / f"vae_resident{lnc.suffix()}.json").write_text(json.dumps(res, indent=2) + "\n")
    print("RESIDENT OK", flush=True)


if __name__ == "__main__":
    main()

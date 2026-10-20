"""CPU check: the compact cross-attention context gives the same block output as the full 512-token context.

    python check_ctx_compact.py                 real layer weights if artifacts/tier2/block<i>_fp32.pt exist, else random
    python check_ctx_compact.py --layer 7 --real-tokens 29 --slots 64

The repo zero-pads the prompt embedding to 512 tokens and cross-attends without a mask. Every padding row becomes
the same vector after text_embedding, so its K and V are identical too, and softmax over n identical keys equals
softmax over one copy with log(n) added to its score. block_port.compact_context builds that compact form (one row
per distinct context row plus fp32 log-counts); CausalBlockPort(..., ctx_logw=...) attends over it.

Checks, at the demo shapes (x 1024 tokens, cache 6144 tokens, context 512 -> slots):
  1. fp32: compact vs full, same inputs and cache -> y and both new caches (gate: cosine >= 0.999999).
  2. bf16 on CPU: both paths against the fp32 full-context result, to show the compact path is no less accurate.
Runs without Trainium; no torch_neuronx import.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from block_port import ART_DIR, BlockCfg, CausalBlockPort, compact_context, cosine, max_abs, rope_cos_sin

GATE = 0.999999


def build(cfg, layer, seed):
    """Block (real weights when available) and a text_embedding MLP to turn a padded prompt into a context."""
    g = torch.Generator().manual_seed(seed)
    port = CausalBlockPort(cfg).eval().requires_grad_(False)
    path = ART_DIR / f"block{layer}_fp32.pt"
    real = path.exists()
    if real:
        data = torch.load(path, map_location="cpu")
        port.load_state_dict(data["block"], strict=True)
        embed = data.get("embed") or torch.load(ART_DIR / "block0_fp32.pt", map_location="cpu")["embed"]
        te = {k: embed[f"text_embedding.{k}"] for k in ("0.weight", "0.bias", "2.weight", "2.bias")}
    else:  # random weights; the modulation is randomised too so every residual branch is exercised
        with torch.no_grad():
            port.modulation.normal_(0, 0.1, generator=g)
            for m in (port.self_attn, port.cross_attn):
                m.norm_q.weight.normal_(1, 0.1, generator=g)
                m.norm_k.weight.normal_(1, 0.1, generator=g)
        te = {"0.weight": torch.randn(cfg.dim, cfg.text_dim, generator=g) * cfg.text_dim ** -0.5,
              "0.bias": torch.randn(cfg.dim, generator=g) * 0.1,
              "2.weight": torch.randn(cfg.dim, cfg.dim, generator=g) * cfg.dim ** -0.5,
              "2.bias": torch.randn(cfg.dim, generator=g) * 0.1}
    return port, te, real


def make_context(cfg, te, real_tokens, seed):
    """Prompt embedding with real_tokens rows, zero-padded to text_len (as the repo does), through text_embedding."""
    g = torch.Generator().manual_seed(seed)
    text = torch.zeros(1, cfg.text_len, cfg.text_dim)
    text[:, :real_tokens] = torch.randn(1, real_tokens, cfg.text_dim, generator=g) * 0.1
    h = F.gelu(F.linear(text, te["0.weight"], te["0.bias"]), approximate="tanh")
    return F.linear(h, te["2.weight"], te["2.bias"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--real-tokens", type=int, default=29, help="29 = the anime prompt used for all G2-G4 numbers")
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    cfg = BlockCfg()
    port, te, real = build(cfg, args.layer, args.seed)
    context = make_context(cfg, te, args.real_tokens, args.seed + 1)
    ctx_c, logw = compact_context(context, args.slots)
    distinct = int((logw > -1).sum())

    g = torch.Generator().manual_seed(args.seed + 2)
    x = torch.randn(1, cfg.l_chunk, cfg.dim, generator=g)
    e = torch.randn(1, 1, 6, cfg.dim, generator=g) * 0.1
    k_cache = torch.randn(1, cfg.l_cache, cfg.num_heads, cfg.head_dim, generator=g)
    v_cache = torch.randn(1, cfg.l_cache, cfg.num_heads, cfg.head_dim, generator=g)
    cos_, sin_ = rope_cos_sin(cfg, 6)
    bias = torch.zeros(1, 1, 1, cfg.l_cache)

    full = port(x, e, context, cos_, sin_, k_cache, v_cache, bias)
    comp = port(x, e, ctx_c, cos_, sin_, k_cache, v_cache, bias, logw)
    fp32 = {name: {"cos": cosine(a, b), "max_abs": max_abs(a, b), "ref_abs_max": a.abs().max().item()}
            for name, a, b in zip(("y", "k_cache", "v_cache"), full, comp)}

    # the cross-attention branch alone, so the comparison is not dominated by the unchanged residual stream
    port_noxattn = CausalBlockPort(cfg).eval().requires_grad_(False)
    port_noxattn.load_state_dict(port.state_dict())
    with torch.no_grad():
        port_noxattn.cross_attn.o.weight.zero_()
        port_noxattn.cross_attn.o.bias.zero_()
    base = port_noxattn(x, e, context, cos_, sin_, k_cache, v_cache, bias)[0]
    branch = {"cos": cosine(full[0] - base, comp[0] - base),
              "branch_rms": (full[0] - base).pow(2).mean().sqrt().item()}

    bf = torch.bfloat16
    port_bf = CausalBlockPort(cfg).eval().requires_grad_(False)
    port_bf.load_state_dict(port.state_dict())
    port_bf = port_bf.to(bf)
    args_bf = [t.to(bf) for t in (x, e)]
    rest_bf = [t.to(bf) for t in (cos_, sin_, k_cache, v_cache, bias)]
    y_full_bf = port_bf(*args_bf, context.to(bf), *rest_bf)[0].float()
    y_comp_bf = port_bf(*args_bf, ctx_c.to(bf), *rest_bf, logw)[0].float()
    bf16 = {"full_vs_fp32_cos": cosine(y_full_bf, full[0]), "compact_vs_fp32_cos": cosine(y_comp_bf, full[0]),
            "full_vs_fp32_max_abs": max_abs(y_full_bf, full[0]), "compact_vs_fp32_max_abs": max_abs(y_comp_bf, full[0])}

    # per block call: K/V projections over the context + the two cross-attention matmuls (2 FLOPs per MAC)
    def xattn_gflop(lk):
        return (2 * 2 * lk * cfg.dim * cfg.dim + 2 * 2 * cfg.l_chunk * lk * cfg.dim) / 1e9

    res = {"weights": f"layer {args.layer} (real)" if real else "random", "real_tokens": args.real_tokens,
           "context_rows": cfg.text_len, "distinct_rows": distinct, "slots": args.slots, "fp32": fp32,
           "cross_attn_branch_fp32": branch, "bf16_cpu": bf16,
           "cross_attn_gflop_per_call": {"full": round(xattn_gflop(cfg.text_len), 2),
                                         "compact": round(xattn_gflop(args.slots), 2)},
           "context_bytes_per_call_bf16": {"full": cfg.text_len * cfg.dim * 2,
                                           "compact": args.slots * cfg.dim * 2 + args.slots * 4}}
    res["pass"] = bool(min(v["cos"] for v in fp32.values()) >= GATE and branch["cos"] >= GATE)
    print(json.dumps(res, indent=2))
    print(f"CTX COMPACT {'PASS' if res['pass'] else 'FAIL'} (gate {GATE})")


if __name__ == "__main__":
    main()

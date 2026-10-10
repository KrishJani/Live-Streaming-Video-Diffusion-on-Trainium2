"""tier2/out/fps_waterfall.png: FPS at each engineering step (run in the pod; reads measured JSONs)."""
import json

from PIL import Image, ImageDraw, ImageFont

from block_port import ART_DIR
from g2_common import OUT

TAG = "512x512_L1024_cache6144_sink3072_text512"
PASSES, LAYERS, FRAMES = 2, 30, 4
fps_from_block = lambda ms: FRAMES / (ms / 1e3 * LAYERS * PASSES)
g1 = json.load(open(ART_DIR / f"g1_block0_{TAG}_bench.json"))["latency_ms"]["mean"]
r1 = json.load(open(ART_DIR / "g2_resident_1layers.json"))["latency_ms_call_plus_y_to_host"]["mean"]
one = json.load(open(ART_DIR / "g3_pipeline_1core.json"))["steady_state_fps"]
four = json.load(open(ART_DIR / "g3_pipeline_4core.json"))["steady_state_fps"]
e2e = json.load(open(ART_DIR / "g4_e2e_4core.json"))["steady_state_fps"]

steps = [
    ("G1 baseline", "caches via host", fps_from_block(g1), "from block latency"),
    ("Resident KV cache", "1 layer per graph", fps_from_block(r1), "from block latency"),
    ("8-layer graphs", "4 graphs, 1 core", one, "measured, full DiT"),
    ("4-core pipeline", "1 graph per core", four, "measured, full DiT"),
    ("End to end, 4 cores", "incl. TAEHV enc + dec", e2e, "measured, pixels to pixels"),
]

S = 2
W, H = 1100 * S, 560 * S
SURFACE, INK, INK2, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
img = Image.new("RGB", (W, H), SURFACE)
d = ImageDraw.Draw(img)
f_title, f_lab, f_small, f_val = (ImageFont.load_default(size=s * S) for s in (22, 16, 13, 20))
d.text((40 * S, 22 * S), "Frames per second on Trainium2, step by step", font=f_title, fill=INK)
d.text((40 * S, 54 * S), "Wan2.1-1.3B causal DiT, 512x512, 2 denoising passes per chunk, 4 frames per chunk. Only the last bar includes the VAE.",
       font=f_small, fill=INK2)

x0, x1, base, top = 90 * S, W - 40 * S, H - 110 * S, 110 * S
ymax = 50.0
py = lambda v: base - (base - top) * v / ymax
for t in range(0, 51, 10):
    d.line([(x0, py(t)), (x1, py(t))], fill=GRID, width=S)
    d.text((x0 - 10 * S, py(t)), str(t), font=f_small, fill=INK2, anchor="rm")
d.text((x0 - 10 * S, top - 22 * S), "FPS", font=f_small, fill=INK2, anchor="rm")

slot = (x1 - x0) / len(steps)
bw = 110 * S
prev = None
for i, (name, sub, fps, basis) in enumerate(steps):
    cx = x0 + slot * (i + 0.5)
    left, right, y = cx - bw / 2, cx + bw / 2, py(fps)
    d.rounded_rectangle([left, y, right, base], radius=4 * S, fill=BAR)
    d.rectangle([left, base - 4 * S, right, base], fill=BAR)  # square at the baseline
    d.text((cx, y - 10 * S), f"{fps:.1f}", font=f_val, fill=INK, anchor="mb")
    if prev is not None:
        d.text((cx - slot / 2, py(max(fps, prev[1])) - 14 * S), f"x{fps / prev[1]:.1f}", font=f_lab, fill=INK2, anchor="mb")
    d.text((cx, base + 12 * S), name, font=f_lab, fill=INK, anchor="ma")
    d.text((cx, base + 36 * S), sub, font=f_small, fill=INK2, anchor="ma")
    d.text((cx, base + 56 * S), basis, font=f_small, fill=INK2, anchor="ma")
    prev = (name, fps)

OUT.mkdir(parents=True, exist_ok=True)
img.resize((W // S, H // S), Image.LANCZOS).save(OUT / "fps_waterfall.png")
print("WATERFALL OK", [(s[0], round(s[2], 2)) for s in steps])

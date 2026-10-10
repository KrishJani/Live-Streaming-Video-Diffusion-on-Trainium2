"""out/compare_labels.png: a 1536x48 label strip for the 3-panel comparison video (local ffmpeg has no drawtext)."""
from PIL import Image, ImageDraw, ImageFont

from g2_common import OUT

LABELS = ["Input", "Image model, per frame", "Video model, KV cache, 38.4 FPS on Trainium2"]
W, H = 512, 48
img = Image.new("RGB", (W * len(LABELS), H), "#111111")
d = ImageDraw.Draw(img)
font = ImageFont.load_default(size=22)
for i, text in enumerate(LABELS):
    d.text((W * i + W // 2, H // 2), text, font=font, fill="#ffffff", anchor="mm")
OUT.mkdir(parents=True, exist_ok=True)
img.save(OUT / "compare_labels.png")
print("LABELS OK")

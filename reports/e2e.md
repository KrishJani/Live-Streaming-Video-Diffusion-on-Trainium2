# End to end on Trainium2: TAEHV encoder -> causal Wan DiT -> TAEHV decoder

**Result: MEASURED 5.64 FPS on one NeuronCore**, with the encoder, all 30 DiT layers and the decoder running as
traced Neuron graphs for every stream chunk. One 4-frame chunk at 512 x 512 takes 0.709 s. The output is the same
stylised video as Track 2's G2 run (which used the Wan VAE on CPU): 26.2 dB PSNR against it with identical noise.

Run on seat-230 (trn2.48xlarge, `NEURON_RT_VISIBLE_CORES=0`), 2026-10-10. Code: `tier2/e2e.py`. Raw numbers:
`tier2/results/e2e.json`. Video: `tier2/out/e2e_trainium.mp4` (input | output). Still: `tier2/out/e2e_compare.png`.

One stage is not on Trainium: the **session start** (the first 2 latent frames, a one-off 2048-token chunk) runs
the DiT on CPU with the fp32 port, exactly as in Track 2's G2 run, because the traced block graph is fixed at one
frame. It took 15.6 s once. Its encode and decode are on Neuron. All 6 stream chunks are Neuron only.

## Measured timing per chunk (one core)

Per stream chunk: 1 encoder call, 2 DiT passes (60 block calls), 1 decoder call, 4 output frames.

| Chunk | Wall | TAEHV encode | DiT (60 block calls) | TAEHV decode | Host overhead | FPS |
|---|---|---|---|---|---|---|
| 0 | 0.414 s | 18.4 ms | 0.339 s (1 pass) | none | 57 ms | no output yet |
| 1 | 0.893 s | 18.8 ms | 0.775 s | 38.0 ms | 61 ms | 4.48 |
| 2 | 0.795 s | 23.1 ms | 0.718 s | 38.0 ms | 16 ms | 5.03 |
| 3 | 0.705 s | 18.7 ms | 0.633 s | 38.0 ms | 15 ms | 5.67 |
| 4 | 0.708 s | 18.7 ms | 0.633 s | 38.0 ms | 19 ms | 5.65 |
| 5 | 0.714 s | 18.7 ms | 0.642 s | 38.0 ms | 16 ms | 5.60 |
| **Mean of chunks 3-5** | **0.709 s** | **18.7 ms (2.6%)** | **0.636 s (89.7%)** | **38.0 ms (5.4%)** | **16.6 ms (2.3%)** | **5.64** |

- FPS = 4 output frames / wall seconds per chunk. The headline uses chunks 3 to 5; chunks 1 and 2 are still
  warming up. Over all five chunks that emit frames the mean is 0.763 s, 5.24 FPS.
- Chunk 0 runs one DiT pass and emits nothing: a frame's final latent comes out one chunk after it goes in.
- "DiT" is host call to host return for the 60 block graphs, so it includes moving each block's KV cache both ways.
- "Host overhead" is everything else inside the chunk: noise-scale computation, latent rescale, noise mixing,
  patch embed, time embed, head, scheduler math, and converting the decoded frames to uint8.
- A separate run with freshly seeded noise instead of replayed noise gave the same timing: 0.711 s per chunk.
- Only 3 steady-state chunks were measured; the clip is short. Not measured: frame capture, video writing, a
  longer run (the `t_refresh` position rewind after about 45 chunks and adaptive sink refresh are still not
  implemented in the host code).

Compared with the VAE on CPU (Track 2's figures: Wan encode 12.1 s, Wan decode 15.8 s per chunk), the same DiT
went from about 0.14 FPS to 5.64 FPS end to end. The DiT is now 90% of the chunk.

## How it is wired

- **DiT:** the 13 layers missing on seat-230 were traced with the G1 method (one graph per layer, 3 at a time,
  about 33 s each); all 30 are in `/workspace/livevid/artifacts/tier2/`. The host code is Track 2's
  `g2_neuron.Runner`, unchanged: steps [700, 500] with the adaptive first step (610, 601, then 600), one KV cache
  set per denoising row, 3 sink frames, cached T5 embedding for the "anime" prompt.
- **Lookahead fix:** TAEHV latent k is pixel frames 4k..4k+3, while Wan latent k is frames 4k-3..4k. The clip is
  padded with 3 copies of its last frame at the end, and every encoder call is one plain 4-frame chunk. In a live
  stream this means the output runs 3 frames (about 0.2 s at 16 fps) later than with the Wan encoder.
- **Scale fix:** TAEHV latents are mean/std normalised and the pipeline encodes un-normalised
  (`streamv2v/inference.py:148`), so latents are mapped with `lat * std + mean` before noise is added.
- **Decode:** the DiT output is already in the normalised space TAEHV decodes; the first 3 decoded frames are
  dropped. Output is 25 frames (1 + 4 x 6), the same 25 as Track 2's video.
- The encoder fallback (Wan-encoded latents) was not needed.

## Quality against Track 2's G2 output

Same clip, prompt and schedule. To make the comparison meaningful the noise tensors of a CPU reference run
(`g2_ref.py` on seat-230, same seed as Track 2's) were replayed, so the only differences are the TAEHV encoder,
the TAEHV decoder and bf16.

Encoder check inside the run: rescaled TAEHV latents vs the Wan encoder's latents, cosine 0.9980 to 0.9985 on all
7 encode calls (RMS difference 0.12 to 0.20 on latents of RMS 2.3 to 3.2).

PSNR of the all-Trainium output, per chunk of output frames:

| Compared with | Frames 0-4 | 5-8 | 9-12 | 13-16 | 17-20 | 21-24 | All |
|---|---|---|---|---|---|---|---|
| Track 2 G2 video, Neuron DiT + Wan VAE (`g2_side_by_side.mp4`, right panel) | 26.3 | 25.6 | 25.6 | 26.1 | 26.8 | 27.1 | **26.2 dB** |
| Track 2 G2 video, CPU reference + Wan VAE (middle panel) | 26.3 | 25.6 | 25.5 | 26.1 | 26.8 | 27.1 | 26.2 dB |
| CPU reference latents from seat-230, decoded by the same Neuron TAEHV decoder | 27.1 | 26.2 | 26.0 | 26.7 | 27.5 | 28.1 | 26.9 dB |

- No drift over the 6 chunks; the last chunks are the closest.
- The third row removes the decoder difference and still shows 26.9 dB, so most of the gap comes from the TAEHV
  encoder's latents being amplified by the DiT, not from the TAEHV decoder.
- Track 2's frames were read back from an mp4, so those two rows include video compression error.
- Visually (`tier2/out/e2e_compare.png`, frames 0, 8, 16, 24): same character, pose, colours and ball position;
  small differences in fine detail such as the eyes. With different noise the style holds but the content differs
  much more (18.1 dB), which is sampling variation, not an error.

## Not done

- **Resident TAEHV state** (step 5, `input_output_aliases`): not attempted. Encode plus decode is 57 ms of the
  709 ms chunk, so the most it could save is a few percent; the same trick on the DiT's KV caches is where the
  time is.
- Session start on Neuron (needs a 2-frame block graph).
- seat-231 was not used; seat-232 and seat-234 were not touched.

## Reproduce (seat-230)

```
tier2/remote.sh push
tier2/remote.sh run "./g2_compile_all.sh"                       # skips layers that already exist
tier2/remote.sh run "python -u vae_neuron.py compile enc && python -u vae_neuron.py compile dec"
tier2/remote.sh run "python -u g2_ref.py run"                   # optional: CPU reference whose noise is replayed
tier2/remote.sh run "NEURON_RT_VISIBLE_CORES=0 python -u e2e.py"
```

# Nemotron 3.5 Bolt 74B-A9B (24GB) on one RTX 5090

A working vLLM recipe for [nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB)
on a single 32 GB RTX 5090. It starts from the model card's command and changes only what a 32 GB card needs.

## Quick start

```bash
pip install -U huggingface_hub
export MODELS=$HOME/models/bolt
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB         --local-dir $MODELS/model
hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr --local-dir $MODELS/drafter   # for the "fast" profile
./serve.sh fast        # or: long | multi
curl localhost:8000/v1/models
```

Needs Docker with the NVIDIA container runtime. Image: `vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989`
(from the model card). The container installs `b12x` and runs `vllm download-kernels` on each start.
`serve.sh` downloads the model's reasoning parser (`ultra_v3_reasoning_parser.py`) from the model repo on first run.
API model name: `nemotron-3.5-bolt`. Boot takes about 2 minutes.

## Profiles

| profile | drafter | max seqs | context | GPU mem util |
|---|---|---:|---:|---:|
| `fast`  | DFlash LiLiCorr, 5 tokens | 1 | 262,144 | 0.95 |
| `long`  | none | 1 | 262,144 | 0.96 |
| `multi` | none | 8 | 32,768 | 0.88 (+ `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`) |

## Results

RTX 5090 32 GB, power limit 408 W, stock clocks. fp8 KV cache. Warm runs, 512-token decode.

| profile | prefill 8K | prefill 32K | decode, 1 stream | concurrent |
|---|---:|---:|---:|---|
| `fast`  | 6,277 tok/s | 6,003 tok/s | ~218 tok/s (see below) | - |
| `multi` | 6,537 tok/s | 6,092 tok/s | 172 tok/s | 8 requests: 95 tok/s per stream, 738 tok/s total; 4 requests: 121 per stream, 463 total (all run at once) |
| `long`  | - | - | - | 200K-token needle test passed; 200K prefill at 4,195 tok/s |

`fast` long-context checks (drafter on, 262K window): needle at 196K tokens passed (prefill 4,391 tok/s),
needle at 250K tokens passed (prefill 3,704 tok/s, measured with the 2048 chunk). Run them with `python3 needle.py 250000 [seed]`.
Use a new seed for each cold run: the same seed gives the same prompt, and the prefix cache answers it in ~2 s.

Decode speed changes from boot to boot by about ±5%, so each decode figure is the median of 8 runs per boot.

Power during single-stream decode: mean ~390 W, peaks at the 408 W limit.

### Draft token count (`fast` profile)

The first version of `fast` used 7 draft tokens and 98K context. 5 draft tokens is faster and fits the full context:

| draft tokens | max context with drafter | decode, 1 stream | mean acceptance length |
|---:|---:|---:|---:|
| 7 (old `fast`) | ~108K (served at 98K) | ~203 tok/s (2 boots: 204, 202) | 2.23-2.30 |
| 5 (new `fast`) | 262K (KV pool 301K tokens) | ~218 tok/s (3 boots: 226, 213, 217) | 2.22-2.25 |
| 3 | ~195K | 214-220 tok/s (3 runs) | 2.14 |
| 2 | - | 197-215 tok/s (3 runs) | 1.96 |

Why it is faster: draft positions 6 and 7 are rarely accepted (mean acceptance length stays at ~2.3),
so 5 tokens gives about the same accepted tokens per step with a cheaper verify pass (6 tokens, not 8):
96-99 verify steps/s instead of 89-90.

Why more context fits: in `--mamba-cache-mode align`, every Mamba layer keeps one state page per draft token,
so the per-request KV need grows with the draft count (2.59 / 2.69 / 2.88 / 2.99 GiB for a 262K request at
3 / 4 / 6 / 7 tokens, against ~2.27 GiB available). At 5 tokens the attention block (4,416 tokens) matches
the Mamba page size exactly, no padding is added, and a 262K request fits. The LiLiCorr drafter accepts at most 7.

## Notes for 32 GB cards

- Set `--limit-mm-per-prompt`. Without it, the vision encoder's memory profiling runs out of memory at startup.
- The DFlash drafter takes most of the KV headroom. With 7 draft tokens about 107K tokens of context fit; with 5, the full 262K fits (KV pool ~301K tokens). Without the drafter the KV pool is ~776K tokens.
- Prefill chunk (`--max-num-batched-tokens`, default 2048 here). `fast` uses 4096: 196K-token prefill 4,148-4,176 -> 4,384-4,391 tok/s (+5%),
  32K +3%, 8K unchanged, decode unchanged within boot-to-boot noise, KV pool 301K -> 296K tokens (262K still fits).
  8192 runs out of memory at startup with the drafter loaded (the profile run needs a 2 GiB buffer).
- `multi`: turn off vLLM's CUDA-graph memory estimate. With 4+ sequences it reserves ~3.8 GiB for CUDA graphs that use ~0.04 GiB,
  which left 0.4 GiB of KV cache: only 3 short requests ran at once (each request holds a large Mamba state) and a 4th waited.
  With `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` the KV pool grows from 34K to 148K tokens:

  | `multi` setting | KV pool | 4 requests total | 8 requests total |
  |---|---:|---:|---:|
  | old: 4 seqs, GMU 0.96 | 34K tokens | 271 tok/s (3 at once) | - |
  | 4 seqs, GMU 0.88, estimate off | 148K tokens | 463 tok/s | - |
  | new: 8 seqs, GMU 0.88, estimate off | 148K tokens | 429-434 tok/s | 738 tok/s (all 8 at once) |
  | 16 seqs, GMU 0.86, estimate off | 91K tokens | 434 tok/s | 741 tok/s; 16 requests 692 tok/s (only ~8 at once) |

  GMU has to drop with the estimate off, because about 4 GiB of GPU memory sits outside PyTorch and is not counted:
  at 0.96 and 0.90 the server runs out of memory at startup; 0.88 boots. 16 sequences need 0.86, and the smaller pool
  then holds only ~8 requests. Stress test with 8 seqs: 8 concurrent 30K-token prompts all complete
  (the scheduler preempts and resumes when the pool is full).

## Files

- `serve.sh` - start the server with one of the profiles
- `bench.py` - smoke test, prefill and decode checks
- `bench_s4.py` - fires N concurrent streams and reports per-stream and total decode speed
- `needle.py` - long-context needle test: `python3 needle.py 250000 [seed]`
- `stress.py` - N concurrent long prompts: `python3 stress.py 8 30000`

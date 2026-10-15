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
| `multi` | none | 4 | 32,768 | 0.96 |

## Results

RTX 5090 32 GB, power limit 408 W, stock clocks. fp8 KV cache. Warm runs, 512-token decode.

| profile | prefill 8K | prefill 32K | decode, 1 stream | concurrent |
|---|---:|---:|---:|---|
| `fast`  | 6,300 tok/s | 5,857 tok/s | 226 tok/s (median of 8, 210-230) | - |
| `multi` | 6,516 tok/s | 6,084 tok/s | 173 tok/s | 4 requests: 131 tok/s per stream, 271 tok/s total (3 run at once, see notes) |
| `long`  | - | - | - | 200K-token needle test passed; 200K prefill at 4,195 tok/s |

`fast` long-context checks (drafter on, 262K window): needle at 196K tokens passed (prefill 4,153 tok/s),
needle at 250K tokens passed (prefill 3,704 tok/s). Run them with `python3 needle.py 250000`.

Power during single-stream decode: mean ~390 W, peaks at the 408 W limit.

### Draft token count (`fast` profile)

The first version of `fast` used 7 draft tokens and 98K context. 5 draft tokens is faster and fits the full context:

| draft tokens | max context with drafter | decode, 1 stream | mean acceptance length |
|---:|---:|---:|---:|
| 7 (old `fast`) | ~108K (served at 98K) | 204 tok/s (median of 8, 190-218) | 2.30 |
| 5 (new `fast`) | 262K (KV pool 301K tokens) | 226 tok/s (median of 8, 210-230) | 2.25 |
| 3 | ~195K | 214-220 tok/s (3 runs) | 2.14 |
| 2 | - | 197-215 tok/s (3 runs) | 1.96 |

Why it is faster: draft positions 6 and 7 are rarely accepted (mean acceptance length stays at ~2.3),
so 5 tokens gives the same accepted tokens per step with a cheaper verify pass (6 tokens, not 8).

Why more context fits: in `--mamba-cache-mode align`, every Mamba layer keeps one state page per draft token,
so the per-request KV need grows with the draft count (2.59 / 2.69 / 2.88 / 2.99 GiB for a 262K request at
3 / 4 / 6 / 7 tokens, against ~2.27 GiB available). At 5 tokens the attention block (4,416 tokens) matches
the Mamba page size exactly, no padding is added, and a 262K request fits. The LiLiCorr drafter accepts at most 7.

## Notes for 32 GB cards

- Set `--limit-mm-per-prompt`. Without it, the vision encoder's memory profiling runs out of memory at startup.
- The DFlash drafter takes most of the KV headroom. With 7 draft tokens about 107K tokens of context fit; with 5, the full 262K fits (KV pool ~301K tokens). Without the drafter the KV pool is ~776K tokens.
- `--max-num-batched-tokens 8192` runs out of memory at startup with the drafter loaded (the profile run needs a 2 GiB buffer). Keep the default.
- Each running request holds a large Mamba state. With 4 sequences only ~32K context fits, and in practice 3 requests run at once; a 4th waits for a free slot.
- Other processes on the GPU (about 1 GB here) limit `--gpu-memory-utilization` to about 0.96.

## Files

- `serve.sh` - start the server with one of the profiles
- `bench.py` - smoke test, prefill and decode checks
- `bench_s4.py` - fires N concurrent streams and reports per-stream and total decode speed
- `needle.py` - long-context needle test: `python3 needle.py 250000`

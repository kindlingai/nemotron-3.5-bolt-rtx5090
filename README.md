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
| `fast`  | DFlash LiLiCorr, 7 tokens | 1 | 98,304 | 0.95 |
| `long`  | none | 1 | 262,144 | 0.96 |
| `multi` | none | 4 | 32,768 | 0.96 |

## Results

RTX 5090 32 GB, power limit 408 W, stock clocks. fp8 KV cache. Warm runs, 512-token decode.

| profile | prefill 8K | prefill 32K | decode, 1 stream | concurrent |
|---|---:|---:|---:|---|
| `fast`  | 5,915 tok/s | 6,130 tok/s | 200-211 tok/s | - |
| `multi` | 6,516 tok/s | 6,084 tok/s | 173 tok/s | 4 requests: 131 tok/s per stream, 271 tok/s total (3 run at once, see notes) |
| `long`  | - | - | - | 200K-token needle test passed; 200K prefill at 4,195 tok/s |

Power during single-stream decode: mean ~367 W, peaks at the 408 W limit.

## Notes for 32 GB cards

- Set `--limit-mm-per-prompt`. Without it, the vision encoder's memory profiling runs out of memory at startup.
- The DFlash drafter takes most of the KV headroom. With the drafter, about 107K tokens of context fit; without it, the full 262K fits (KV pool ~776K tokens).
- Each running request holds a large Mamba state. With 4 sequences only ~32K context fits, and in practice 3 requests run at once; a 4th waits for a free slot.
- Other processes on the GPU (about 1 GB here) limit `--gpu-memory-utilization` to about 0.96.

## Files

- `serve.sh` - start the server with one of the profiles
- `bench.py` - smoke test, prefill and decode checks
- `bench_s4.py` - fires N concurrent streams and reports per-stream and total decode speed

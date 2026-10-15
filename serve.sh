#!/bin/bash
# Nemotron 3.5 Bolt 74B-A9B 24GB on one RTX 5090 (32 GB) with vLLM.
# Usage: ./serve.sh [fast|long|multi]
#   fast  : DFlash LiLiCorr drafter (5 draft tokens), 1 sequence, full 262K context (best single-user speed)
#   long  : no drafter, 1 sequence, full 262K context (larger KV pool, no draft model)
#   multi : no drafter, 8 sequences, 32K context
# Needs: docker with the NVIDIA runtime, the model + drafter downloaded to $MODELS
#   hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB         --local-dir $MODELS/model
#   hf download nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr --local-dir $MODELS/drafter
set -e
PROFILE=${1:-fast}
MODELS=${MODELS:-$HOME/models/bolt}
PORT=${PORT:-8000}
IMAGE=vllm/vllm-openai:nightly-77154ca6116ae34cbfe314beef1016bde9b63989
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/ultra_v3_reasoning_parser.py" ] || wget -q -O "$HERE/ultra_v3_reasoning_parser.py" \
  https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB/resolve/main/ultra_v3_reasoning_parser.py
SPEC=(); EXTRA=(); ENVS=()
case $PROFILE in
  fast)  LEN=262144; SEQS=1; GMU=0.95
         # 5 draft tokens, not 7: same accepted tokens per step, cheaper verify, and the full 262K fits (see README)
         EXTRA=(--max-num-batched-tokens 4096)   # +5% prefill on long prompts; 8192 does not fit with the drafter
         SPEC=(--speculative-config '{"method":"dflash","model":"/models/drafter","num_speculative_tokens":5,"rejection_sample_method":"standard","attention_backend":"FLASHINFER","draft_load_config":{"load_format":"auto"}}') ;;
  long)  LEN=262144; SEQS=1; GMU=0.96
         EXTRA=(--max-num-batched-tokens 4096) ;;   # +4.5% prefill at 196K; 8192 does not fit
  multi) LEN=32768;  SEQS=8; GMU=0.88
         # vLLM's CUDA-graph memory estimate reserves ~3.8 GiB here; the real graph pool is ~0.04 GiB.
         # Turn it off and lower GMU (about 4 GiB of GPU memory is outside PyTorch and not counted).
         ENVS=(-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0) ;;
  *) echo "profile: fast|long|multi"; exit 1 ;;
esac
docker rm -f bolt 2>/dev/null || true
docker run -d --name bolt --gpus all --ipc=host -p $PORT:8000 \
  -v "$MODELS":/models:ro -v "$HERE":/work:ro -v bolt-cache:/root/.cache \
  -e VLLM_USE_V2_MODEL_RUNNER=1 "${ENVS[@]}" \
  --entrypoint /bin/bash $IMAGE \
  -c 'pip install -q b12x && vllm download-kernels && exec vllm serve /models/model "$@"' _ \
  --served-model-name nemotron-3.5-bolt \
  --trust-remote-code --load-format auto --tensor-parallel-size 1 \
  --max-model-len $LEN --max-num-seqs $SEQS --gpu-memory-utilization $GMU \
  --enable-chunked-prefill --kv-cache-dtype fp8 \
  --linear-backend b12x --moe-backend b12x \
  --attention-backend FLASHINFER --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype bfloat16 --mamba-cache-dtype bfloat16 --mamba-cache-mode align \
  --limit-mm-per-prompt '{"video":1,"image":1}' --video-pruning-rate 0.5 \
  --media-io-kwargs '{"video":{"fps":2,"num_frames":256}}' \
  --seed 42 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin /work/ultra_v3_reasoning_parser.py --reasoning-parser ultra_v3 \
  "${SPEC[@]}" "${EXTRA[@]}"
echo "started; follow with: docker logs -f bolt   (ready in ~2 min)"

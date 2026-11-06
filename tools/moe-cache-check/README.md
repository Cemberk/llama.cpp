# MoE cache tools

Tools for the MoE cache of this fork (`--moe-cache-*`, `src/llama-moe-cache.cpp`).

## check.mjs: does the cache give the right output, and how often does it hit?

Runs `llama-server` with each cache configuration on a MoE model and compares
the outputs with the run that keeps every expert on the GPU. Needs Node 18 or
later and a build with a GPU backend (the cache needs one).

```sh
# a small MoE model that fits any machine (4.2 GB)
hf download allenai/OLMoE-1B-7B-0924-Instruct-GGUF olmoe-1b-7b-0924-instruct-q4_k_m.gguf

node tools/moe-cache-check/check.mjs --server build/bin/llama-server --model <path to the gguf>
node tools/moe-cache-check/check.mjs --server build/bin/llama-server --model <path to the gguf> --probs
```

On Windows the server is `build\bin\Release\llama-server.exe`. It exits with 1
when a check fails:

- `cache-slru`, `cache-lfu`, `cache-next` generate the same text as `gpu`, and
  with `--probs` the same log-probabilities (to 1e-4).
- `hybrid` computes the misses on the CPU: it differs from `gpu` by the
  CPU/GPU rounding that `cpu-experts` shows, no more.
- `profile-1` saves a profile; `profile-2` starts from it and should hit more.

Each line reports the generation speed and the cache's hits and misses. On a
GPU that shares system memory (Apple silicon, integrated graphics) the cache
is expected to be slower than none: it is for discrete GPUs.

Results on a Radeon 780M (Windows, Vulkan) are in qwen-code's
`docs/design/2026-11-05-adaptive-local-inference-design.md`.

## ../moe-sim/moe_cache_sim.py: which policy would hit most?

Replays a trace (`--moe-cache-trace FILE`, which `check.mjs` records in its
`cache-slru` run) against the cache's policies and the offline optimum.

```sh
python tools/moe-sim/moe_cache_sim.py <trace> --slots 70
python tools/moe-sim/moe_cache_sim.py /tmp/synthetic.txt --synthetic --fraction 0.25
python tools/moe-sim/moe_cache_sim.py --self-test
```

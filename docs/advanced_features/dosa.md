# DoSA prefill indexer

DoSA reduces score computation, score writes and Top-K input for the FP8 DSA
indexer. The implementation targets NVIDIA SM100 and 128-dimensional index
heads. Few supports the 32-head GLM-5.2-FP8 and 64-head DeepSeek-V3.2 indexers;
Full supports 32 heads. Unsupported calls retain SGLang's ordinary indexer path.

## Install and start SGLang

Build and install a normal SGLang wheel from this checkout with its declared
[dependencies](../../python/pyproject.toml), including the default bundled Rust
extensions. This revision uses Python 3.11 or later, PyTorch 2.14.1, DeepGEMM
0.2.2 and FlashInfer 0.7.1. Native compilation requires a compatible CUDA
toolkit with `nvcc`, SM100 support and C++20, plus DeepGEMM's bundled CuTe and
CUTLASS headers.

Also build and install `sglang-kernel` from this checkout's
[AOT source directory](../../python/sglang/kernels/aot/README.md#building-from-source)
using the same PyTorch environment. Its exact Top-K overflow repair is separate
from the main SGLang wheel; installing the main wheel alone does not install
that repair. Historical research binaries are not installation dependencies.

Use your normal SGLang command with one DoSA setting:

```bash
SGLANG_DOSA_MODE=few python -m sglang.launch_server \
  --model-path /path/to/GLM-5.2-FP8 \
  --tp 8
```

Keep the usual node count, rank and rendezvous arguments when running across
multiple nodes. Set `SGLANG_DOSA_MODE=full` for the shared full-computation
baseline, or omit the setting to use ordinary SGLang.

For DeepSeek-V3.2, replace the model path; no additional DoSA option is needed.
The model's configured context limit remains unchanged. If you extend it for a
performance experiment, use the same context extension for each comparison
and assess long-context task quality separately.

DoSA extensions build from the packaged sources during backend setup and reuse
the standard PyTorch extension cache. Normal serving does not require a private
native manifest, research path or manual kernel selection. Startup and build
time are separate from request latency.

## Memory and measurement

DoSA reserves its workspaces during model construction, before SGLang sizes the
KV cache. It does not silently change SGLang's chunk size, memory fraction,
selector, sampling policy or communication settings. Few sizes its physical
query workspace from the configured query and key capacities, and processes
larger logical calls in bounded query slabs. One joint byte budget covers
Record5 candidate records, causal ID pages and both task tables, using the same
planning mechanism for 32 and 64 score heads. The normal 1M-key slot uses 4096
physical query rows, 8359936 ID pages and 114688 records per row for logical
16K or 32K chunks. The logical chunk size stays unchanged.

This budget covers candidate storage, not the entire model or every workspace.
Record overflow and numerical-domain guards retain exact full-head recovery;
they do not truncate candidates. Each independent request constructs and
maintains its own prior; prior state is not shared across requests.

As with ordinary SGLang, long-context capacity depends on the model, chunk size
and available memory. A KV capacity limit must include the allocator's page
headroom as well as the request's input and output tokens. Benchmark reports
must state any nondefault memory or chunk settings and use the same capacity
contract for the compared implementations.

For controlled experiments, an explicit `SGLANG_DOSA_NATIVE_MANIFEST` remains
supported. It freezes the native bindings for the process; invalid bindings
raise an error. This advanced option is not needed for normal serving.

## Measure independent requests

From the repository root, with a dedicated server already running, supply a real UTF-8 text file:

```bash
python benchmark/dosa/bench_ttft.py --text /path/to/prompt.txt --output few.json
```

The client sends one warmup and five measured requests, flushes prefix cache
before each, and assigns each a unique request ID. Server tokenization, prefill
and request-owned prior work remain inside TTFT. It retains every sample and
natural EOS, reports median and standard deviation, and preserves failures
without retrying. A response with zero output tokens is not a valid TTFT.

For a controlled 16K/32K comparison, start separate Stock and Few servers with
the same model and serving arguments. Use `--chunked-prefill-size 16384` for
both modes, then repeat with `--chunked-prefill-size 32768`. Select Stock with
`SGLANG_DOSA_MODE=off` and Few with `SGLANG_DOSA_MODE=few`; keep the same text,
parallelism, FP8 KV, context limit, memory fraction and sampling settings.

To use one warmup and three measured requests, as in the reproduction campaign:

```bash
python benchmark/dosa/bench_ttft.py --text /path/to/prompt.txt \
  --loaded 3 --output few-q16k.json
```

Run the client separately for each mode and chunk size with a new output file.
Client JSON serialization and cache flush are outside TTFT; server tokenization,
prefill and request-owned prior work are inside. The client does not change the
server's chunk size or other settings.

Use a new output filename for each run. The client records relevant server
settings and does not alter them. Compare modes with the same hardware, text
and declared server configuration; startup/build time remains separate.

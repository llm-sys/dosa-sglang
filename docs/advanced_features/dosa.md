# DoSA prefill indexer

DoSA reduces score computation, score writes and Top-K input for the FP8 DSA
indexer. The implementation targets NVIDIA SM100 and 128-dimensional index
heads. GLM-5.2-FP8 uses 32 index heads. Unsupported calls retain
SGLang's ordinary indexer path.

See the [TTFT reproduction guide](../../benchmark/dosa/README.md) for this
branch's GLM-5.2 results and complete launch commands.

## Start SGLang

Use your normal SGLang command with one DoSA setting:

```bash
SGLANG_DOSA_MODE=few python -m sglang.launch_server \
  --model-path /path/to/GLM-5.2-FP8 \
  --tp 8
```

Keep the usual node count, rank and rendezvous arguments when running across
multiple nodes. Set `SGLANG_DOSA_MODE=full` for the shared full-computation
baseline, or omit the setting to use ordinary SGLang.

The model's configured context limit remains unchanged. If you extend it for a
performance experiment, use the same context extension for each comparison
and assess long-context task quality separately.

Native extensions build from the packaged sources on first use and reuse the
standard PyTorch extension cache. A compatible CUDA toolkit with `nvcc`, the
matching PyTorch installation, DeepGEMM and FlashInfer are required. No private
native manifest, absolute research path, generated launch script or manual
kernel selection is required. First-build time is separate from request latency.

## Memory and measurement

DoSA reserves its workspaces during model construction, before SGLang sizes the
KV cache. It does not silently change SGLang's chunk size, memory fraction,
selector, sampling policy or communication settings. Few sizes its physical
query workspace from the configured query and key capacities, and processes
larger logical calls in bounded query slabs. This keeps candidate storage within
its slot budget without changing the requested chunk size. Rows that exceed record capacity are recomputed exactly. Each
independent request constructs and maintains its own prior; prior state is not
shared across requests.

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

Use a new output filename for each run. The client records relevant server
settings and does not alter them. Compare modes with the same hardware, text
and declared server configuration; startup/build time remains separate.

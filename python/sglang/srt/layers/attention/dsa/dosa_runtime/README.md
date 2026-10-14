# DoSA SM100 runtime

Current Few builds use device-owned logical K extents to reuse captured graphs
across changing prefill K. The builder emits21 native roles; older sixteen-role
manifests remain supported. Validate the complete Indexer and serving workload
on the intended model and hardware. Component timings do not establish an
all-settings or complete-Indexer speedup.

This package provides capacity-bound Full and Few FP8 prefill backends with
shared caller, full-head completion and recovery components. Full preserves its
surrogate interval and original-H32 refinement. Few's completed full-head records use
a separate global20 score domain; the common storage format does not make the
two domains interchangeable.

Default SGLang dispatch is unchanged: `SGLANG_DOSA_MODE` defaults to `off`.
Explicit `full` and `few` modes prepare their native bindings and actual storage
during model construction, before KV sizing. Service admission retains the
single-request SM100 FP8/D128/topk2048 PAGED restrictions and original
unordered selector policy. Few supports configured query capacities from4096
through32768 and logical prefill tails down to one row. Physical execution pads
to eight rows. Calls larger than the capacity-bound physical workspace use
query slabs and one logical carry transaction; an explicit contiguous causal
interval advances each slab's scan and hint. The final carry covers the last1536 logical rows, even across slabs.
Service admission starts at131072 visible keys; the internal scoring floor is
the original12288-key seed. These are shared policies, not per-layer or dataset
rules. Unsupported cases retain the original Indexer path; selected-mode
source, ABI and initialization errors are explicit.

Few also supports 64 score heads through the same scoring, record and recovery
sources. Both head geometries size internal query batches against one byte
budget covering Record5, causal ID pages and both task tables. The allowance is
`min(logical_Q,16384) * (65536*5 + 256*528)` bytes. A physical block uses
`physical_Q*114688*5 + pages*528` bytes, where `528=(128+2+2)*4` covers an ID page
and its two task-table entries. The planner selects the largest halved Q8 block
that fits, reserving the original page allowance or the complete causal ID
bound, whichever is larger. Capture operands and
other query workspaces use the same physical query extent before KV sizing.
For a 1M-key slot with logical Q16K or Q32K, the physical workspace has 4096 rows,
8359936 ID pages and 114688 records per row. Logical chunk sizes and output order
remain unchanged. Exact recovery handles rows exceeding record capacity; it
never truncates candidates. This bounds candidate storage, not total model or
workspace memory; model/KV capacity must be measured with the intended settings.
Full allocation remains unchanged. Recovery always recomputes the model's full
head set. Normal builds propagate the actual head count to every head-domain
consumer, including the guards in inactive certificate families and compaction.

## Build and use a service mode

Build and install a normal SGLang wheel from this checkout with the dependency
versions in `python/pyproject.toml` and its default bundled Rust extensions.
This revision uses PyTorch 2.14.1, DeepGEMM 0.2.2 and FlashInfer 0.7.1. Native
builds require C++20 and a CUDA toolkit with `sm_100a` support. DeepGEMM must
include its CuTe and CUTLASS headers. Native sources are packaged; the default
loader builds and caches them before model KV reservation. Normal use adds one
mode switch to the model's usual SGLang command:

```bash
SGLANG_DOSA_MODE=few python -m sglang.launch_server --model-path /path/to/model
```

Use `full` for the all-score path or `off` for default SGLang dispatch. Keep the
model's usual parallelism and serving arguments. DoSA does not silently change
chunk size, memory fraction, selector or communication settings. A native
manifest and private research paths are not required.

`few-decode` uses the same Few prefill backend and additionally enables the
experimental H64 paged decoder. It computes the ordinary full FP8 scores, then
uses two request-local sets of previously selected keys as exact current-score
witnesses to reduce Top-K work. A new request invalidates both sets; the first
decode step pays the full selection. Equality and nonfinite values survive the
filter. The normal `few` mode retains the original decoder.

This decoder is experimental. Measure the complete decode workload on the
intended model, context lengths and batch sizes before making a performance
claim. Select it with the same single mode switch:

```bash
SGLANG_DOSA_MODE=few-decode python -m sglang.launch_server --model-path /path/to/model
```

The AOT `fast_topk` selectors and the DSV4-v2 selector have an exact coarse-bucket
overflow repair in this checkout. Unequal FP32 candidates are refined through
complete-input rescans before bounded collection; equal-score tie order remains
unspecified. These repairs do not describe every other SGLang selector backend.
Rebuild `sglang-kernel` from `python/sglang/kernels/aot` following its
[build instructions](../../../../../kernels/aot/README.md); the main `sglang`
wheel does not rebuild this separate AOT package. Use the same PyTorch
environment for both packages and keep the default SGLang selector. Historical
research binaries are not substitutes for the source build.

| Mode | Native bindings |
|---|---|
| Full | primary, caller, post/recovery, H32 sample |
| Few | thirteen base keys, three record5 roles, five live-NK roles |
| Few-decode | the same Few bindings, plus the shared SGLang selector and request witnesses |

For a frozen build, an **optional** manifest can be created before launch:

```bash
python -m sglang.srt.layers.attention.dsa.dosa_runtime.build_manifest \
  --mode few --build-directory /absolute/path/dosa-build \
  --output /absolute/path/dosa-build/few-native.json
export SGLANG_DOSA_NATIVE_MANIFEST=/absolute/path/dosa-build/few-native.json
```

Add `--score-heads 64` when exporting an optional H64 Few manifest. Normal
serving detects the model's head count without this build command or a manifest.

Use separate build directories/manifests for distinct revisions and modes.
The builder validates native methods and file identities without constructing
an execution backend or running numerical kernels. It does not overwrite an
existing manifest. One service process keeps immutable mode/native bindings.
Legacy sixteen-role Few manifests remain supported; new builds add all five
live-NK roles together. A canonical library can supply multiple logical roles
without copying or renaming the binary.

`native_binding.py` owns dependency plans, ABI and immutable bindings for both
service and explicit factories. `assembly.py` constructs records, the cold
bootstrap and capture ownership before publishing a slot. Both modes then use
the same paged-call transaction; the bootstrap has one owner per slot.

The current-scale RMS auxiliary also builds from packaged source at backend
construction; `KEY_RHO_NATIVE.json` is an optional immutable pin. Missing or
incompatible dependencies raise errors instead of substituting an older kernel.

Both modes reserve real DG/FI and carry storage before KV sizing. Eligible stock
outputs can produce a request-local prior through the shared publisher; adopting
it pays the validation wait. Without a valid prior, the call pays the cold H32
bootstrap. Few additionally installs its completed-H32 record/direct-seed path.
Full/Few share the public paged-call entry and execution-slot state. Large Few
calls use bounded slabs without replacing that entry at runtime; one logical
transaction owns carry, diagnostics and completion fences. Request state is not
reused across independent requests.

Both modes reuse retired outer-graph allocations through an owned CUDA memory
pool. Native reset must complete before reuse; a retained graph gets a separate
generation. An empty host-anchor graph owns the pool on a separate private
stream. Recovery/body/dispatch graphs retain their independent factories and
lifetimes. Device synchronization and optional GC remain. Full skips global
device-cache reclamation only when reusing a retired generation with matching
query-row and norm-reduction allocation shapes. New generations and changed
shapes reclaim first. Direct and Few captures retain their original reclamation.
This policy requires PyTorch's native CUDA allocator.
These private DoSA pools are separate from SGLang's general model-graph pool.

Few retains a bounded cache of two physical query geometries, including each
parent's original conditional-recovery graph and operands. A shape switch first
fences the entire preceding call, then rebinds the same shared scratch views.
Eviction releases the oldest private graph lease; no caller Q/K or carry values
are cached. The cache does not change numerical work or slab partitioning.
Original recovery and selector output storage remains shared.

Pool creation is lazy in the first real capture and is charged to that call.
Retained allocations still occupy device memory. Account for all live pool
generations and their high-water storage when measuring model/KV capacity.

Few uses paid14/16/20 routing for H32 and paid14/16/28 for H64. When positive
heads leave exact-work slots,
negative fills use their original weighted norm and the minority sign count from
the existing128 samples. The planner recomputes the remaining DD8 support and
full guards; this does not add another QK sampling pass. The native builder
explicitly enables this selection rule.

Paid14 screening uses four mathematical warp groups, each owning two queries,
within one640-thread CTA. It retains the original K256 tiles, four raw-key
buffers, two MMA slots, F32 reduction and score guards. Each group writes only
its two channels of a Q4 bitmap. The native wrapper requests maximum shared
memory carveout; this attribute is included in the measured change. Its scheduler
selects the minimum estimated work over valid key partitions, with an exact
bounded search for up to8192 key tiles and the original policy beyond that
range. Empty families skip the search. Query IDs retain their original stable
order and capacity. The scheduler passes its launch grid explicitly to the
native trusted entry; older native bindings are rejected before execution.
This changes workload distribution, not the score arithmetic or field budget.

The high family (paid20 for H32, paid28 for H64) stores its weights as
upward-rounded BF16 pairs and
reconstructs F32 values with integer operations. The original RN reduction,
guards, FP8 inputs and full-head refinement remain. The upper bound can become looser;
this representation does not reduce input or final-score precision. A per-call
GPU pack checks the admitted field domain and assigns each Q4 group to either
the packed kernel or the original F32 high-family fallback. Out-of-domain key scales
keep visible keys. Packed records contain 64 words per Q4 group for H32 and
80 words for H64: `(physical_Q/4)*record_words*4` bytes, plus
`physical_Q*2` bytes for the two routing-budget arrays. These allocations use
the physical workspace extent, not the server's logical chunk size.
Preparation, both dispatches and recovery remain
inside the normal call and graph lifetime. The `fp16` native export names are
legacy ABI names; this record's encoding is BF16.

## Explicit APIs and qualification

For an independent HTTP comparison, keep the same real UTF-8 text, model,
parallelism, FP8 KV, context limit, memory fraction and sampling settings. Run
Stock (`SGLANG_DOSA_MODE=off`) and Few with `--chunked-prefill-size 16384`, then
repeat both with `--chunked-prefill-size 32768`. With each dedicated server
already running, the repository client can use one warmup and three measured
requests:

```bash
python benchmark/dosa/bench_ttft.py --text /path/to/prompt.txt \
  --loaded 3 --output few-q16k.json
```

Use a new output file per mode/chunk. The client defaults to five measured
requests when `--loaded` is omitted. It preserves text bytes, flushes prefix
cache and uses a fresh request ID for every call. Client serialization and cache
flush are outside TTFT; server tokenization, prefill and request-owned prior work
are inside. Startup/build time is separate; all samples and failures are retained.

`create_backend` exposes capacity-bound implementations. `create_live_backend`
installs the selected record/cold-bootstrap policy before reservation. Every
`run_paged` call supplies its actual operands, visibility, page tables, output
and request key; these factories do not bind a fixed fixture or fabricate a prior.

A native build or component result is not a model or performance qualification.
Validate actual KV capacity, working-set headroom, graph lifetimes, tail and
decode behavior in the intended model and hardware. Full and Few have different
real workspace requirements; do not assume equal KV capacity from the same memory
fraction. This release makes no all-settings or universal speedup claim.

No diagnostic observer, model profiling hook, captured request data, benchmark
output or fixed research assembly is packaged. Existing DeepGEMM and FlashInfer
caches remain under the caller's control.

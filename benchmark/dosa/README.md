# Reproduce GLM-5.2 TTFT

DoSA reduces GLM-5.2-FP8's 1M-token TTFT from **131.79 s to 73.49 s (1.79×)**
on eight GB200 GPUs. The 512K and 768K results improve by 1.36× and 1.61×.

This branch is based on official SGLang v0.5.18
(`71de97b264b04dcd514cf904003028aefe9775c8`) and preserves its full Git history.
It contains the measured DoSA implementation and the candidate-capacity repair.

## Install

Use Linux, Python 3.12, CUDA 13 with `nvcc`, and two nodes with four GB200 GPUs
each in one NVLink domain. Source installation also needs a C++ compiler and
Rust (`cargo` and `rustc`). Run these commands on both nodes:

```bash
git clone --branch dosa-v5.18 https://github.com/llm-sys/dosa-sglang.git
cd dosa-sglang
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./python
python benchmark/dosa/build_exact_topk.py --output .dosa_topk
export PYTHONPATH="$PWD/.dosa_topk/python:$PWD/python"
```

The branch pins PyTorch 2.13.0, FlashInfer 0.6.17 and DeepGEMM 0.1.5.post3.
The build command compiles this branch's exact Top-K source with the original
compiler flags. The generated `sgl_kernel` package loads these three CUDA
selectors alongside the installed package. Both SGLang and DoSA use it.
DoSA's native kernels compile during server initialization and reuse the normal
PyTorch extension cache.

## Prepare the original inputs

The five prompts are prefixes of Chinese long-book sample `id=123` from
[InfiniteBench](https://github.com/OpenBMB/InfiniteBench). The preparation script
uses dataset revision `90f0394333616266d9fe85824ceaf505093cbaa5` and the original
character boundaries:

```bash
python benchmark/dosa/prepare_ttft_inputs.py --output-dir ttft-inputs
```

If the dataset is already downloaded, reuse it:

```bash
python benchmark/dosa/prepare_ttft_inputs.py \
  --dataset /path/to/longbook_qa_chn.jsonl --output-dir ttft-inputs
```

| GLM tokenizer length | Prefix characters |
|---:|---:|
| 131072 | 183198 |
| 262144 | 364946 |
| 524288 | 728585 |
| 786432 | 1092950 |
| 1048569 | 1459229 |

The script writes the original UTF-8 text. The server performs tokenization.

## Start each mode

Use the GLM-5.2-FP8 checkpoint on both nodes. Set `NODE_RANK=0` on the first
node and `NODE_RANK=1` on the second. `NODE0_ADDR` is the first node's address.
Run one mode at a time: `off` for SGLang, then `few` for DoSA.

```bash
source .venv/bin/activate
export MODEL_PATH=/path/to/GLM-5.2-FP8
export NODE0_ADDR=192.0.2.10
export NODE_RANK=0
export CHUNK=32768
export SGLANG_DOSA_MODE=off
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD/.dosa_topk/python:$PWD/python"
export TORCH_EXTENSIONS_DIR="/tmp/dosa-v518-node-${NODE_RANK}/torch_extensions"
unset SGLANG_DOSA_NATIVE_MANIFEST

python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 8 --ep 1 --nnodes 2 --node-rank "$NODE_RANK" \
  --dist-init-addr "${NODE0_ADDR}:29784" \
  --host 0.0.0.0 --port 30184 \
  --context-length 1048576 \
  --chunked-prefill-size "$CHUNK" --max-prefill-tokens 16384 \
  --mem-fraction-static 0.88 --page-size 64 --kv-cache-dtype fp8_e4m3 \
  --dsa-topk-backend sgl-kernel \
  --dsa-prefill-backend trtllm --dsa-decode-backend trtllm \
  --moe-runner-backend flashinfer_trtllm --moe-a2a-backend none \
  --sampling-defaults model --tokenizer-worker-num 1 \
  --schedule-policy fcfs --schedule-conservativeness 1.0 \
  --num-continuous-decode-steps 1 \
  --random-seed 233767787 --stream-interval 1 \
  --cuda-graph-config "$(cat benchmark/dosa/graph-config.json)"
```

The configuration uses FP8 KV, TP8/EP1, 32K prefill chunks, full decode graphs
and breakable prefill graphs. The saved graph file contains the original bucket
sizes. To run the 16K configuration, set `CHUNK=16384` for both modes.

## Measure TTFT

After the server reports ready, run the client on node 0:

```bash
mkdir -p ttft-results
for tokens in 131072 262144 524288 786432 1048569; do
  python benchmark/dosa/bench_ttft.py \
    --url "http://${NODE0_ADDR}:30184" \
    --text "ttft-inputs/prefix-${tokens}.txt" \
    --loaded 3 \
    --output "ttft-results/${SGLANG_DOSA_MODE}-q${CHUNK}-${tokens}.json"
done
```

Then stop the servers, set `SGLANG_DOSA_MODE=few` on both nodes, launch them with
the same command, and run the same client loop.

Each prompt gets one warmup and three measured requests. Every request has a
fresh ID and starts after a prefix-cache flush. TTFT ends at the first streamed
completion token and includes server tokenization, prefill and request-owned
DoSA prior work. Generation uses `max_new_tokens=2` and the model's default EOS.
The client saves every timing sample and reports the median and standard
deviation. Server startup and kernel compilation are outside the measurements.

## Results

| Input tokens | SGLang TTFT (s) | DoSA TTFT (s) | Speedup |
|---:|---:|---:|---:|
| 131072 | 5.717 | 5.676 | 1.007× |
| 262144 | 14.287 | 12.806 | 1.116× |
| 524288 | 40.128 | 29.550 | 1.358× |
| 786432 | 80.020 | 49.690 | 1.610× |
| 1048569 | 131.786 | 73.487 | 1.793× |

[ttft-results.json](ttft-results.json) contains all three measured samples for
each entry. These measurements use 32K chunks and the configuration above.

DoSA reserves its workspaces before KV-cache sizing. Its bounded physical query
slabs process the requested logical chunk, and rows exceeding candidate
capacity are recomputed exactly. The model input, score precision and selected
Top-K semantics stay the same.

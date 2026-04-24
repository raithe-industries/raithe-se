#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# © RAiTHE INDUSTRIES INCORPORATED 2026
# raithe.sh — Single entrypoint for raithe-se
#
# Usage:
#   ./raithe.sh                    # verify models, then run
#   ./raithe.sh --install-only     # install/verify models, do not run
#   ./raithe.sh --force-install    # re-export all models, then run
#   ./raithe.sh --run-only         # skip model check, run immediately
#   ./raithe.sh --json-logs        # passed through to raithe-se binary
#   ./raithe.sh --seeds <path>     # passed through to raithe-se binary
#   ./raithe.sh --config <path>    # passed through to raithe-se binary
#
# This script is the sole operational entrypoint. It owns:
#   1. System preflight (apt deps, pip, PATH)
#   2. Hardware detection and generator selection
#   3. ORT shared library — auto-download if missing
#   4. ONNX model installation and validation
#   5. Binary launch
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/target/release/raithe-se"
MODEL_BASE="$SCRIPT_DIR/data/models"

ORT_VERSION="1.20.1"
ORT_CACHE="${HOME}/.cache/ort.pyke.io"
# GPU-enabled ORT tarball: includes CUDA 12 kernels (compatible with the
# CUDA 13 stubs driver 580+ ships), plus CPU kernels as unconditional
# backstop for the embedder/reranker. Same filename, different build.
ORT_TARBALL="onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TARBALL}"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

step() { echo -e "${CYAN}▶ $1${RESET}"; }
sub()  { echo -e "  ${CYAN}·${RESET} $1"; }
ok()   { echo -e "${GREEN}✓ $1${RESET}"; }
warn() { echo -e "${YELLOW}! $1${RESET}"; }
fail() { echo -e "${RED}✗ $1${RESET}"; exit 1; }
skip() { echo -e "${GREEN}↷ $1 — already installed (--force-install to reinstall)${RESET}"; }

echo -e "
                         \033[1;37m██████╗  █████╗ ██╗████████╗██╗  ██╗███████╗\033[0m
                         \033[1;37m██╔══██╗██╔══██╗   ╚══██╔══╝██║  ██║██╔════╝\033[0m
                         \033[1;37m██████╔╝███████║██║   ██║   ███████║█████╗  \033[0m
                         \033[1;37m██╔══██╗██╔══██║██║   ██║   ██╔══██║██╔══╝  \033[0m
                         \033[1;37m██║  ██║██║  ██║██║   ██║   ██║  ██║███████╗\033[0m
                         \033[1;37m╚═╝  ╚═╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝\033[0m
                                         \033[0;36mSEARCH ENGINE\033[0m
                            \033[0;36m© RAiTHE INDUSTRIES INCORPORATED 2026\033[0m
"

# ── Argument parsing ──────────────────────────────────────────────────────────

INSTALL_ONLY=0
FORCE_INSTALL=0
RUN_ONLY=0
PASSTHROUGH_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --install-only)  INSTALL_ONLY=1 ;;
        --force-install) FORCE_INSTALL=1 ;;
        --run-only)      RUN_ONLY=1 ;;
        *)               PASSTHROUGH_ARGS+=("$arg") ;;
    esac
done

# ==============================================================================
# PREFLIGHT — system deps, pip, PATH
# ==============================================================================

step "Preflight checks"

# Ensure ~/.local/bin on PATH (pip --user installs land here)
export PATH="$HOME/.local/bin:$PATH"

# apt packages required on fresh Ubuntu 24.04
APT_NEEDED=()
for pkg in python3 python3-pip pkg-config build-essential wget; do
    command -v "$pkg" >/dev/null 2>&1 || dpkg -s "$pkg" >/dev/null 2>&1 || APT_NEEDED+=("$pkg")
done
# libssl-dev checked via dpkg
dpkg -s libssl-dev >/dev/null 2>&1 || APT_NEEDED+=("libssl-dev")

if [[ ${#APT_NEEDED[@]} -gt 0 ]]; then
    warn "Installing apt packages: ${APT_NEEDED[*]}"
    sudo apt-get install -y "${APT_NEEDED[@]}" \
        || fail "apt install failed for: ${APT_NEEDED[*]}"
fi
ok "System packages present"

# Ensure ~/.local/bin in ~/.bashrc for future shells
grep -q 'HOME/.local/bin' ~/.bashrc \
    || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

ok "PATH: $HOME/.local/bin included"

# ==============================================================================
# SYSTEM DETECTION
# ==============================================================================

step "Detecting system resources"

RAM_GB=$(free -g | awk '/Mem:/ {print $2}')
CPU_CORES=$(nproc)
FREE_DISK_GB=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {gsub("G",""); print $4}')

VRAM_MB=0
VRAM_GB=0
HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
        | head -n1 || echo "0")
    VRAM_GB=$((VRAM_MB / 1024))
    HAS_GPU=1
fi

ok "CPU CORES:   ${CPU_CORES}"
ok "RAM:         ${RAM_GB} GB"
ok "FREE DISK:   ${FREE_DISK_GB} GB"
ok "VRAM:        ${VRAM_GB} GB (GPU present: $( [[ $HAS_GPU -eq 1 ]] && echo yes || echo no ))"

# ── CUDA gate ─────────────────────────────────────────────────────────────────
# raithe-se is a GPU-first product. The generator (7B / 14B causal-LM) is
# unusable on CPU — a 2026 design choice. Fail loudly if the box is not
# CUDA-capable rather than silently falling back to a 60-minute CPU load.
if [[ $HAS_GPU -ne 1 ]]; then
    fail "No NVIDIA GPU detected (nvidia-smi absent). raithe-se requires \
≥4 GB of CUDA-capable VRAM. Install NVIDIA drivers or run on a GPU host."
fi
if [[ $VRAM_GB -lt 4 ]]; then
    fail "VRAM too small: ${VRAM_GB} GB detected, ≥4 GB required. \
Smallest supported tier is Qwen2.5-1.5B fp16 at ~3 GB VRAM."
fi

# Parse nvidia-smi's advertised CUDA version. Driver-bundled CUDA stubs
# from driver 580+ report CUDA 13.0; ort 2.0.0-rc.12 is built against
# CUDA ≥12.8, so 13.0 is a superset. We only warn on unexpected majors.
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader 2>/dev/null | head -n1 || echo "")
NV_CUDA=$(nvidia-smi 2>/dev/null | awk -F'CUDA Version: ' '/CUDA Version:/ {print $2}' | awk '{print $1}' || echo "")
if [[ -z "$NV_CUDA" ]]; then
    warn "Could not parse CUDA version from nvidia-smi; proceeding anyway"
else
    CUDA_MAJOR="${NV_CUDA%%.*}"
    if [[ "$CUDA_MAJOR" -lt 12 ]]; then
        fail "CUDA ${NV_CUDA} detected. ort 2.0.0-rc.12 requires CUDA ≥12.8. \
Upgrade NVIDIA driver to ≥535 (CUDA 12.8) or ≥575 (CUDA 13.0)."
    fi
    ok "CUDA:        ${NV_CUDA} (driver-bundled)"
fi

# ── Generator tier matrix ─────────────────────────────────────────────────────
# Keyed on (VRAM, RAM). int4 (MatMulNBits) fits a 7B model in ~5 GB VRAM
# including KV cache; used for 8–15 GB cards. fp16 for 16+ GB. All tiers
# are CUDA, no CPU fallback (by policy).
#
# Embedder + reranker always run on CPU (Option A): leaves the full
# generator budget for the generator, and embedder/reranker are fast
# enough on CPU that GPU time isn't the bottleneck on their path.
GENERATOR_PROVIDER="cuda"
EMBEDDER_PROVIDER="cpu"
RERANKER_PROVIDER="cpu"
GENERATOR_GPU_MEM_LIMIT_BYTES=$((6 * 1024 * 1024 * 1024))    # default 6 GB

if   [[ $VRAM_GB -ge 24 && $RAM_GB -ge 64 ]]; then
    GENERATOR_ID="Qwen/Qwen2.5-14B-Instruct"
    GENERATOR_DISPLAY="Qwen2.5-14B-Instruct"
    GENERATOR_QUANT="fp16"
    GENERATOR_GPU_MEM_LIMIT_BYTES=$((20 * 1024 * 1024 * 1024))
    TIER_LABEL="14B fp16 CUDA"
elif [[ $VRAM_GB -ge 16 && $RAM_GB -ge 32 ]]; then
    GENERATOR_ID="Qwen/Qwen2.5-7B-Instruct"
    GENERATOR_DISPLAY="Qwen2.5-7B-Instruct"
    GENERATOR_QUANT="fp16"
    GENERATOR_GPU_MEM_LIMIT_BYTES=$((14 * 1024 * 1024 * 1024))
    TIER_LABEL="7B fp16 CUDA"
elif [[ $VRAM_GB -ge 8  && $RAM_GB -ge 16 ]]; then
    GENERATOR_ID="Qwen/Qwen2.5-7B-Instruct"
    GENERATOR_DISPLAY="Qwen2.5-7B-Instruct"
    # Round 1: fp32 — matches existing on-disk generator; int4 conversion
    # lands in round 2 alongside the Stage 2 quantization pipeline. When
    # flipped to int4 here, also update Stage 2 export + add quant.txt.
    GENERATOR_QUANT="fp32"
    GENERATOR_GPU_MEM_LIMIT_BYTES=$((14 * 1024 * 1024 * 1024))
    TIER_LABEL="7B fp32 CUDA (int4 pending round 2)"
elif [[ $VRAM_GB -ge 6  && $RAM_GB -ge 16 ]]; then
    GENERATOR_ID="Qwen/Qwen2.5-3B-Instruct"
    GENERATOR_DISPLAY="Qwen2.5-3B-Instruct"
    GENERATOR_QUANT="fp16"
    GENERATOR_GPU_MEM_LIMIT_BYTES=$((5 * 1024 * 1024 * 1024))
    TIER_LABEL="3B fp16 CUDA"
else
    GENERATOR_ID="Qwen/Qwen2.5-1.5B-Instruct"
    GENERATOR_DISPLAY="Qwen2.5-1.5B-Instruct"
    GENERATOR_QUANT="fp16"
    GENERATOR_GPU_MEM_LIMIT_BYTES=$((3 * 1024 * 1024 * 1024))
    TIER_LABEL="1.5B fp16 CUDA"
fi

ok "AUTO-SELECT: ${TIER_LABEL}"
ok "CONFIRMED:   $GENERATOR_DISPLAY"
ok "VRAM LIMIT:  $((GENERATOR_GPU_MEM_LIMIT_BYTES / 1024 / 1024 / 1024)) GB (generator session)"

# Export the tier selection into the process env so raithe-app's config
# loader (figment, prefix RAITHE__) picks them up without requiring a
# config file rewrite on each run.
export RAITHE__NEURAL__GENERATOR_QUANTIZATION="$GENERATOR_QUANT"
export RAITHE__NEURAL__GENERATOR_PROVIDER="$GENERATOR_PROVIDER"
export RAITHE__NEURAL__EMBEDDER_PROVIDER="$EMBEDDER_PROVIDER"
export RAITHE__NEURAL__RERANKER_PROVIDER="$RERANKER_PROVIDER"
export RAITHE__NEURAL__GENERATOR_GPU_MEM_LIMIT_BYTES="$GENERATOR_GPU_MEM_LIMIT_BYTES"

# ==============================================================================
# ORT SHARED LIBRARY — auto-download if missing
# ==============================================================================

step "Locating ONNX Runtime shared library"

ORT_SO=$(find "$ORT_CACHE" -name "libonnxruntime.so*" -not -name "*.lock" 2>/dev/null \
    | sort -V | tail -n1 || true)

# Detect if the cached dylib is the CPU-only variant (from a previous
# raithe.sh run before we switched to the GPU tarball). Signal: CPU-only
# .so is ~15 MB; GPU .so is ~380 MB. Force re-download if the size is too
# small, otherwise the CUDA EP registration at runtime silently no-ops.
if [[ -n "$ORT_SO" && -f "$ORT_SO" ]]; then
    ORT_SO_MB=$(( $(stat -c%s "$ORT_SO") / 1024 / 1024 ))
    if [[ $ORT_SO_MB -lt 100 ]]; then
        warn "Cached $ORT_SO is ${ORT_SO_MB} MB — likely CPU-only variant; re-downloading GPU build"
        rm -f "$ORT_CACHE"/libonnxruntime.so*
        ORT_SO=""
    fi
fi

if [[ -z "$ORT_SO" ]]; then
    warn "libonnxruntime.so (GPU) not found — downloading ORT v${ORT_VERSION}"
    mkdir -p "$ORT_CACHE"
    TMP_TAR="/tmp/${ORT_TARBALL}"
    wget -q --show-progress "$ORT_URL" -O "$TMP_TAR" \
        || fail "Failed to download ORT from $ORT_URL"
    tar -xzf "$TMP_TAR" -C /tmp
    cp "/tmp/onnxruntime-linux-x64-gpu-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION}" \
        "$ORT_CACHE/"
    # GPU tarball also ships libonnxruntime_providers_cuda.so and
    # libonnxruntime_providers_shared.so — both must sit next to the main
    # .so so dlopen can find them at provider-registration time.
    cp "/tmp/onnxruntime-linux-x64-gpu-${ORT_VERSION}/lib/libonnxruntime_providers_cuda.so" \
        "$ORT_CACHE/" 2>/dev/null || true
    cp "/tmp/onnxruntime-linux-x64-gpu-${ORT_VERSION}/lib/libonnxruntime_providers_shared.so" \
        "$ORT_CACHE/" 2>/dev/null || true
    ln -sf "$ORT_CACHE/libonnxruntime.so.${ORT_VERSION}" \
           "$ORT_CACHE/libonnxruntime.so"
    rm -f "$TMP_TAR"
    rm -rf "/tmp/onnxruntime-linux-x64-gpu-${ORT_VERSION}"
    ORT_SO="$ORT_CACHE/libonnxruntime.so.${ORT_VERSION}"
    ok "ORT (GPU) downloaded and cached"
fi

export ORT_DYLIB_PATH="$ORT_SO"
ok "ORT: $ORT_DYLIB_PATH"

# ==============================================================================
# BINARY CHECK
# ==============================================================================

if [[ ! -x "$BINARY" ]]; then
    fail "raithe-se binary not found — run: cargo build --release"
fi

# ==============================================================================
# MODEL INSTALLATION / VALIDATION
# ==============================================================================

declare -A MODEL_MIN_BYTES=(
    [embedder]=$((1200  * 1024 * 1024))
    [reranker]=$((1800  * 1024 * 1024))
    # Generator floor sized for the 0.5B fp32 fallback. Primary (7B ~14 GB,
    # 14B ~28 GB) and 1.5B fallback (~6 GB) all clear comfortably.
    [generator]=$((1800  * 1024 * 1024))
)

get_task() {
    local model_id="$1"
    if [[ "$model_id" == *"bge-large"* || "$model_id" == *"bge-base"* \
       || "$model_id" == *"bge-small"* ]]; then
        echo "feature-extraction"; return
    fi
    if [[ "$model_id" == *"reranker"* || "$model_id" == *"cross-encoder"* ]]; then
        echo "text-classification"; return
    fi
    if [[ "$model_id" == *"flan"* || "$model_id" == *"t5"* ]]; then
        echo "text2text-generation"; return
    fi
    if [[ "$model_id" == *"Qwen"* || "$model_id" == *"Llama"* \
       || "$model_id" == *"Mistral"* ]]; then
        echo "text-generation"; return
    fi
    echo "text-classification"
}

get_model_class() {
    local task="$1"
    case "$task" in
        text-generation)      echo "AutoModelForCausalLM" ;;
        text2text-generation) echo "AutoModelForSeq2SeqLM" ;;
        text-classification)  echo "AutoModelForSequenceClassification" ;;
        feature-extraction)   echo "AutoModel" ;;
        *)                    echo "AutoModel" ;;
    esac
}

# External data for an ONNX model may be written as either "model.onnx_data"
# (underscore — onnx.save_model's default) or "model.onnx.data" (period —
# torch ONNXProgram.save's default). Both conventions are valid per the ONNX
# external-data spec; the exact filename lives inside the .onnx protobuf's
# TensorProto.external_data.location field. Renaming on disk would desync
# that reference and break ORT's loader, so validators below instead accept
# whichever name the exporter chose.
_model_data_file() {
    local dir="$1" f
    for f in "$dir/model.onnx_data" "$dir/model.onnx.data"; do
        [[ -f "$f" ]] && { echo "$f"; return 0; }
    done
    return 1
}

_model_total_bytes() {
    local dir="$1" onnx_b=0 data_b=0 data_path
    [[ -f "$dir/model.onnx" ]] && onnx_b=$(stat -c%s "$dir/model.onnx")
    data_path=$(_model_data_file "$dir") && data_b=$(stat -c%s "$data_path")
    echo $(( onnx_b + data_b ))
}

model_is_valid() {
    local subdir="$1"
    local dest="$MODEL_BASE/$subdir"
    [[ -f "$dest/model.onnx" && -f "$dest/tokenizer.json" ]] || return 1
    local total_b
    total_b=$(_model_total_bytes "$dest")
    [[ $total_b -ge ${MODEL_MIN_BYTES[$subdir]} ]] || return 1
    return 0
}

export_model() {
    local subdir="$1"
    local hf_id="$2"

    local dest="$MODEL_BASE/$subdir"
    local work="$dest/_work"
    local task model_class
    task=$(get_task "$hf_id")
    model_class=$(get_model_class "$task")

    if [[ "$FORCE_INSTALL" -eq 0 ]] && model_is_valid "$subdir"; then
        local total_mb
        total_mb=$(( $(_model_total_bytes "$dest") / 1024 / 1024 ))
        skip "$subdir ($hf_id) — ${total_mb} MB"
        return
    fi

    step "Exporting $subdir → $hf_id  [$task]  $model_class"

    rm -rf "$work"
    mkdir -p "$work" "$dest"

    # ── Stage 1 — download ──────────────────────────────────────────────────
    sub "Stage 1/3 — downloading weights + tokenizer via $model_class"
    sub "  CPU: ${CPU_CORES} cores  GPU: $( [[ $HAS_GPU -eq 1 ]] \
        && echo "${VRAM_GB} GB VRAM" || echo none )"

    set +e
    OMP_NUM_THREADS=$CPU_CORES MKL_NUM_THREADS=$CPU_CORES \
    python3 - >"$work/download.log" 2>&1 <<PYEOF
import warnings, logging, os, sys, torch

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["TQDM_DISABLE"]            = "1"
os.environ["OMP_NUM_THREADS"]         = "${CPU_CORES}"
os.environ["MKL_NUM_THREADS"]         = "${CPU_CORES}"

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

try:
    from transformers import AutoTokenizer, ${model_class} as ModelClass
except Exception as e:
    print(f"IMPORT_ERROR: {e}", file=sys.stderr); sys.exit(2)

model_id = "${hf_id}"
path     = "${work}"
task     = "${task}"

load_kwargs = {"trust_remote_code": True}
if task == "text-generation":
    load_kwargs["torch_dtype"] = torch.bfloat16

try:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tok.save_pretrained(path)
    model = ModelClass.from_pretrained(model_id, **load_kwargs)
    if task == "text-generation":
        model = model.to("cpu").float()
    model.save_pretrained(path)
    print(f"OK: {model_id}", flush=True)
except Exception as e:
    print(f"DOWNLOAD_ERROR: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(3)
PYEOF
    dl_rc=$?
    set -e

    if [[ $dl_rc -ne 0 ]]; then
        echo -e "${RED}✗ Stage 1 failed for $hf_id — exit $dl_rc${RESET}" >&2
        cat "$work/download.log" >&2
        if [[ "$task" == "text-generation" ]]; then
            warn "download failed for $hf_id — will try next candidate"
            rm -rf "$work" "$dest"
            return 1
        fi
        fail "download failed for $hf_id"
    fi
    ok "download complete"

    # ── Stage 2 — ONNX export ───────────────────────────────────────────────
    sub "Stage 2/3 — exporting to ONNX ($task)"

    rm -rf "$work/onnx"
    mkdir -p "$work/onnx"

    if [[ "$task" == "text-generation" ]]; then
        # Dynamo exporter path for causal-LM text-generation. The legacy
        # TorchScript exporter (used by optimum-cli) runs
        # _jit_pass_onnx_graph_shape_type_inference which serializes the
        # whole ModelProto into a single in-memory protobuf *before* writing
        # to disk — hard 2 GiB ceiling, fails for Qwen2.5-7B fp32 (~14 GB).
        # torch.onnx.export(dynamo=True) uses torch.export/FX instead and
        # emits external-data-format output natively.
        set +e
        OMP_NUM_THREADS=$CPU_CORES MKL_NUM_THREADS=$CPU_CORES \
        python3 - >"$work/export.log" 2>&1 <<PYEOF
import warnings, logging, os, sys
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"]           = "1"
os.environ["OMP_NUM_THREADS"]        = "${CPU_CORES}"
os.environ["MKL_NUM_THREADS"]        = "${CPU_CORES}"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torch.onnx").setLevel(logging.ERROR)

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM
except ImportError as e:
    print(f"IMPORT_ERROR: {e}", file=sys.stderr); sys.exit(2)

class CausalLMLogitsWrapper(nn.Module):
    """Strip HF's ModelOutput tree (past_key_values, hidden_states,
    attentions, loss) to the (input_ids, attention_mask) -> logits contract
    that generator_step in crates/neural/src/lib.rs expects. use_cache=False
    disables KV-cache side-outputs; the Rust path re-feeds the full token
    sequence each step."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, input_ids, attention_mask):
        return self.inner(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits

work_dir = "${work}"
out_dir  = os.path.join(work_dir, "onnx")
out_path = os.path.join(out_dir, "model.onnx")
os.makedirs(out_dir, exist_ok=True)

try:
    model = AutoModelForCausalLM.from_pretrained(
        work_dir, trust_remote_code=True, torch_dtype=torch.float32,
    ).eval()
    wrapped = CausalLMLogitsWrapper(model).eval()

    example_ids  = torch.zeros(1, 8, dtype=torch.long)
    example_mask = torch.ones(1, 8, dtype=torch.long)
    batch = torch.export.Dim("batch", min=1, max=8)
    seq   = torch.export.Dim("seq",   min=2, max=16384)

    with torch.inference_mode():
        onnx_program = torch.onnx.export(
            wrapped,
            (example_ids, example_mask),
            dynamo=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_shapes={
                "input_ids":      {0: batch, 1: seq},
                "attention_mask": {0: batch, 1: seq},
            },
        )

    # ONNXProgram.save() in torch>=2.5 writes external data adjacent to the
    # .onnx file when weights exceed the 2 GiB protobuf limit (produces
    # model.onnx + model.onnx_data). If save() cannot split, fall through to
    # onnx.save_model with explicit external-data config.
    try:
        onnx_program.save(out_path)
    except Exception:
        import onnx
        proto = None
        for attr in ("model_proto", "model", "_model_proto"):
            cand = getattr(onnx_program, attr, None)
            if cand is not None and hasattr(cand, "graph"):
                proto = cand
                break
        if proto is None:
            raise
        onnx.save_model(
            proto, out_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx_data",
            size_threshold=1024,
            convert_attribute=False,
        )
    print("OK", flush=True)
except Exception as e:
    import traceback; traceback.print_exc(file=sys.stderr)
    print(f"EXPORT_ERROR: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(3)
PYEOF
        export_rc=$?
        set -e
        if [[ $export_rc -ne 0 ]]; then
            warn "dynamo export failed for $hf_id (rc=$export_rc)"
            tail -n 30 "$work/export.log" >&2
            rm -rf "$work" "$dest"
            return 1
        fi
    else
        local export_flags="--framework pt --dtype fp32 --monolith"
        # shellcheck disable=SC2086
        OMP_NUM_THREADS=$CPU_CORES MKL_NUM_THREADS=$CPU_CORES \
        optimum-cli export onnx \
            -m "$work" --task "$task" $export_flags "$work/onnx" \
            >"$work/export.log" 2>&1 || {
            echo -e "${RED}✗ ONNX export failed for $hf_id${RESET}" >&2
            cat "$work/export.log" >&2
            fail "ONNX export failed for $hf_id ($task)"
        }
    fi

    local onnx_check
    onnx_check=$(find "$work/onnx" -name "*.onnx" -size +0c | head -n 1 || true)
    if [[ -z "$onnx_check" ]]; then
        echo -e "${RED}✗ optimum exited 0 but no .onnx files found${RESET}" >&2
        cat "$work/export.log" >&2
        ls -lah "$work/onnx/" >&2
        fail "optimum produced no output for $hf_id"
    fi
    ok "ONNX export complete"

    # ── Stage 3 — validate size, install ───────────────────────────────────
    sub "Stage 3/3 — validating and installing"

    local total_b total_mb min_mb
    total_b=$(find "$work/onnx" -type f | xargs stat -c%s 2>/dev/null | awk '{s+=$1} END{print s+0}')
    total_mb=$(( total_b / 1024 / 1024 ))
    min_mb=$(( ${MODEL_MIN_BYTES[$subdir]} / 1024 / 1024 ))

    if [[ $total_b -lt ${MODEL_MIN_BYTES[$subdir]} ]]; then
        echo -e "${RED}✗ $hf_id: ${total_mb} MB total < ${min_mb} MB minimum${RESET}" >&2
        if [[ "$task" == "text-generation" ]]; then
            warn "size-validate failed for $hf_id — will try next candidate"
            rm -rf "$work" "$dest"
            return 1
        fi
        fail "Corrupt ONNX export for $hf_id (${total_mb} MB < ${min_mb} MB)"
    fi
    ok "Validated: ${total_mb} MB ≥ ${min_mb} MB minimum"

    # Copy all exported files into dest
    cp -r "$work/onnx"/. "$dest/"

    # Ensure tokenizer.json present
    if [[ ! -f "$dest/tokenizer.json" ]]; then
        if [[ -f "$work/tokenizer.json" ]]; then
            cp "$work/tokenizer.json" "$dest/tokenizer.json"
        else
            fail "Missing tokenizer.json for $hf_id"
        fi
    fi

    # Encoders: ensure canonical model.onnx symlink if absent
    if [[ ! -f "$dest/model.onnx" ]]; then
        local first_onnx
        first_onnx=$(find "$dest" -name "*.onnx" | sort | head -n 1 || true)
        [[ -n "$first_onnx" ]] && ln -sf "$(basename "$first_onnx")" "$dest/model.onnx" || true
    fi

    rm -rf "$work"
    ok "$subdir installed → $dest"
    echo ""
}

if [[ $RUN_ONLY -eq 0 ]]; then
    step "Verifying model stack"

    if [[ "$FORCE_INSTALL" -eq 1 ]] \
        || ! model_is_valid embedder \
        || ! model_is_valid reranker \
        || ! model_is_valid generator; then

        [[ "$FORCE_INSTALL" -eq 0 ]] \
            && warn "Model stack incomplete or corrupt — installing"

        command -v python3 >/dev/null || fail "python3 required for model installation"

        python3 -c "import transformers, optimum, torch" 2>/dev/null || {
            warn "Installing Python dependencies..."
            pip3 install --quiet --break-system-packages \
                "optimum[exporters]==1.23.3" \
                "transformers==4.46.3" \
                "torch==2.5.1" \
                "onnxscript" \
                "onnx" \
                "onnxruntime" \
                huggingface_hub \
                || fail "pip install failed"
        }

        # Re-export PATH after pip install in case optimum-cli just landed
        export PATH="$HOME/.local/bin:$PATH"
        command -v optimum-cli >/dev/null 2>&1 \
            || fail "optimum-cli not found — ensure ~/.local/bin is on PATH"

        mkdir -p "$MODEL_BASE"

        export_model "embedder"  "BAAI/bge-large-en-v1.5"
        export_model "reranker"  "BAAI/bge-reranker-large"

        # Generator fallback chain. The primary candidate is auto-selected
        # upstream by RAM/VRAM tier; on export failure, cascade through
        # progressively smaller Qwen2.5 Instruct variants. The list below
        # the primary is kept monotonically decreasing so a retry never
        # re-attempts the primary or anything larger than it.
        case "$GENERATOR_ID" in
            "Qwen/Qwen2.5-14B-Instruct")
                GENERATOR_FALLBACKS=(
                    "Qwen/Qwen2.5-7B-Instruct:Qwen2.5-7B-Instruct"
                    "Qwen/Qwen2.5-1.5B-Instruct:Qwen2.5-1.5B-Instruct"
                    "Qwen/Qwen2.5-0.5B-Instruct:Qwen2.5-0.5B-Instruct"
                ) ;;
            "Qwen/Qwen2.5-7B-Instruct")
                GENERATOR_FALLBACKS=(
                    "Qwen/Qwen2.5-1.5B-Instruct:Qwen2.5-1.5B-Instruct"
                    "Qwen/Qwen2.5-0.5B-Instruct:Qwen2.5-0.5B-Instruct"
                ) ;;
            *)
                GENERATOR_FALLBACKS=() ;;
        esac

        generator_installed=0
        GENERATOR_CANDIDATES=(
            "$GENERATOR_ID:$GENERATOR_DISPLAY"
            "${GENERATOR_FALLBACKS[@]}"
        )
        for spec in "${GENERATOR_CANDIDATES[@]}"; do
            cand_id="${spec%%:*}"
            cand_display="${spec##*:}"
            if export_model "generator" "$cand_id"; then
                GENERATOR_ID="$cand_id"
                GENERATOR_DISPLAY="$cand_display"
                generator_installed=1
                break
            fi
            warn "generator candidate $cand_display failed — trying next"
        done
        [[ $generator_installed -eq 1 ]] \
            || fail "All generator fallback candidates exhausted"

        step "Final verification"
        ALL_OK=1
        for subdir in embedder reranker generator; do
            if model_is_valid "$subdir"; then
                total_mb=$(( $(_model_total_bytes "$MODEL_BASE/$subdir") / 1024 / 1024 ))
                if data_path=$(_model_data_file "$MODEL_BASE/$subdir"); then
                    ok "$subdir: model.onnx + $(basename "$data_path") (${total_mb} MB)"
                else
                    ok "$subdir: model.onnx (${total_mb} MB)"
                fi
            else
                echo -e "${RED}✗ $subdir failed validation${RESET}" >&2
                ALL_OK=0
            fi
        done
        [[ $ALL_OK -eq 1 ]] || fail "Model stack verification failed"
        echo ""
        ok "Model stack ready  ·  Generator: $GENERATOR_DISPLAY"
        echo ""
    else
        for subdir in embedder reranker generator; do
            total_mb=$(( $(_model_total_bytes "$MODEL_BASE/$subdir") / 1024 / 1024 ))
            ok "$subdir: ${total_mb} MB"
        done
        ok "Model stack verified  ·  Generator: $GENERATOR_DISPLAY"
        echo ""
    fi
fi

[[ $INSTALL_ONLY -eq 1 ]] && { ok "Install complete."; exit 0; }

# ==============================================================================
# LAUNCH
# ==============================================================================

step "Launching raithe-se"
exec "$BINARY" "${PASSTHROUGH_ARGS[@]}"
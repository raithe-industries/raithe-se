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
#   1. Hardware detection and generator selection
#   2. ONNX model installation and validation
#   3. ORT shared library location and environment setup
#   4. Binary launch
#
# The raithe-se binary prints its own banner at startup.
# This script prints only operational status lines.
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BINARY="$SCRIPT_DIR/target/release/raithe-se"
MODEL_BASE="$SCRIPT_DIR/data/models"

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

# Banner — printed once, first, before all operational output.
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

# ── Generator selection ───────────────────────────────────────────────────────
# (high-accuracy — RAM ≥ 64 GB, VRAM ≥ 16 GB)

GENERATOR_ID="Qwen/Qwen2.5-7B-Instruct"
GENERATOR_DISPLAY="Qwen2.5-7B-Instruct"
if [[ "$RAM_GB" -ge 64 && "$VRAM_GB" -ge 16 ]]; then
    GENERATOR_ID="Qwen/Qwen2.5-14B-Instruct"
    GENERATOR_DISPLAY="Qwen2.5-14B-Instruct"
    ok "AUTO-SELECT: Qwen2.5-14B (high-accuracy)"
else
    ok "AUTO-SELECT: Qwen2.5-7B (balanced)"
fi
ok "CONFIRMED:   $GENERATOR_DISPLAY"

# ==============================================================================
# ORT SHARED LIBRARY
# ==============================================================================

step "Locating ONNX Runtime shared library"

ORT_CACHE="${HOME}/.cache/ort.pyke.io"
ORT_SO=$(find "$ORT_CACHE" -name "libonnxruntime.so*" -not -name "*.lock" 2>/dev/null \
    | sort -V | tail -n1 || true)

if [[ -z "$ORT_SO" ]]; then
    warn "libonnxruntime.so not found under $ORT_CACHE"
    fail "Run 'cargo build --release' first — ort downloads the library automatically."
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

# Minimum combined (model.onnx + model.onnx_data) sizes in bytes.
declare -A MODEL_MIN_BYTES=(
    [embedder]=$((1200  * 1024 * 1024))
    [reranker]=$((1800  * 1024 * 1024))
    [generator]=$((10000 * 1024 * 1024))
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

model_is_valid() {
    local subdir="$1"
    local dest="$MODEL_BASE/$subdir"
    [[ -f "$dest/model.onnx" && -f "$dest/tokenizer.json" ]] || return 1
    local onnx_b data_b total_b
    onnx_b=$(stat -c%s "$dest/model.onnx")
    data_b=0
    [[ -f "$dest/model.onnx_data" ]] && data_b=$(stat -c%s "$dest/model.onnx_data")
    total_b=$(( onnx_b + data_b ))
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
        local onnx_b data_b total_mb
        onnx_b=$(stat -c%s "$dest/model.onnx")
        data_b=0
        [[ -f "$dest/model.onnx_data" ]] && data_b=$(stat -c%s "$dest/model.onnx_data")
        total_mb=$(( (onnx_b + data_b) / 1024 / 1024 ))
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
    # Load in bfloat16 to halve peak RAM (~14 GB for 7B).
    # No device_map — accelerate meta device prevents model.to("cpu").float().
    load_kwargs["dtype"] = torch.bfloat16

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
        fail "download failed for $hf_id"
    fi
    ok "download complete"

    # ── Stage 2 — ONNX export ───────────────────────────────────────────────
    sub "Stage 2/3 — exporting to ONNX ($task)"

    rm -rf "$work/onnx"
    mkdir -p "$work/onnx"

    # --monolith for encoders only.
    # LLMs use ONNX external data format (model.onnx + model.onnx_data) —
    # --monolith on a 7B model doubles peak RAM and produces corrupt output.
    local export_flags="--framework pt --dtype fp32"
    [[ "$task" != "text-generation" ]] && export_flags="$export_flags --monolith"

    # shellcheck disable=SC2086
    OMP_NUM_THREADS=$CPU_CORES MKL_NUM_THREADS=$CPU_CORES \
    optimum-cli export onnx \
        -m "$work" --task "$task" $export_flags "$work/onnx" \
        >"$work/export.log" 2>&1 || {
        echo -e "${RED}✗ ONNX export failed for $hf_id${RESET}" >&2
        cat "$work/export.log" >&2
        fail "ONNX export failed for $hf_id ($task)"
    }

    local onnx_check
    onnx_check=$(find "$work/onnx" -name "model.onnx" -size +0c | head -n 1 || true)
    if [[ -z "$onnx_check" ]]; then
        echo -e "${RED}✗ optimum exited 0 but model.onnx is missing or empty${RESET}" >&2
        cat "$work/export.log" >&2
        ls -lah "$work/onnx/" >&2
        fail "optimum produced no output for $hf_id"
    fi
    ok "ONNX export complete"

    # ── Stage 3 — validate size, install ───────────────────────────────────
    sub "Stage 3/3 — validating and installing"

    local onnx_file onnx_b data_b data_file total_b total_mb min_mb
    onnx_file=$(find "$work/onnx" -name "model.onnx" | head -n 1)
    onnx_b=$(stat -c%s "$onnx_file")
    data_b=0
    data_file=$(find "$work/onnx" \
        \( -name "model.onnx_data" -o -name "*.onnx_data" \) 2>/dev/null \
        | head -n 1 || true)
    [[ -n "$data_file" ]] && data_b=$(stat -c%s "$data_file")
    total_b=$(( onnx_b + data_b ))
    total_mb=$(( total_b / 1024 / 1024 ))
    min_mb=$(( ${MODEL_MIN_BYTES[$subdir]} / 1024 / 1024 ))

    if [[ $total_b -lt ${MODEL_MIN_BYTES[$subdir]} ]]; then
        echo -e "${RED}✗ $hf_id: ${total_mb} MB total < ${min_mb} MB minimum${RESET}" >&2
        cat "$work/export.log" >&2
        fail "Corrupt ONNX export for $hf_id (${total_mb} MB < ${min_mb} MB)"
    fi
    ok "Validated: ${total_mb} MB ≥ ${min_mb} MB minimum"

    cp "$onnx_file" "$dest/model.onnx"
    if [[ -n "$data_file" ]]; then
        cp "$data_file" "$dest/model.onnx_data"
        ok "Installed: model.onnx_data ($(( data_b / 1024 / 1024 )) MB)"
    fi

    if [[ -f "$work/onnx/tokenizer.json" ]]; then
        cp "$work/onnx/tokenizer.json" "$dest/tokenizer.json"
    elif [[ -f "$work/tokenizer.json" ]]; then
        cp "$work/tokenizer.json" "$dest/tokenizer.json"
    else
        fail "Missing tokenizer.json for $hf_id"
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
        python3 -c "import transformers" 2>/dev/null || {
            warn "Installing Python dependencies..."
            pip install -U transformers torch huggingface_hub "optimum[exporters]" \
                || fail "pip install failed"
        }
        command -v optimum-cli >/dev/null || {
            pip install -U "optimum[exporters]" || fail "optimum install failed"
        }
        mkdir -p "$MODEL_BASE"

        export_model "embedder"  "BAAI/bge-large-en-v1.5"
        export_model "reranker"  "BAAI/bge-reranker-large"
        export_model "generator" "$GENERATOR_ID"

        step "Final verification"
        ALL_OK=1
        for subdir in embedder reranker generator; do
            if model_is_valid "$subdir"; then
                onnx_b=$(stat -c%s "$MODEL_BASE/$subdir/model.onnx")
                data_b=0
                [[ -f "$MODEL_BASE/$subdir/model.onnx_data" ]] \
                    && data_b=$(stat -c%s "$MODEL_BASE/$subdir/model.onnx_data")
                total_mb=$(( (onnx_b + data_b) / 1024 / 1024 ))
                if [[ -f "$MODEL_BASE/$subdir/model.onnx_data" ]]; then
                    ok "$subdir: model.onnx + model.onnx_data (${total_mb} MB)"
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
            onnx_b=$(stat -c%s "$MODEL_BASE/$subdir/model.onnx")
            data_b=0
            [[ -f "$MODEL_BASE/$subdir/model.onnx_data" ]] \
                && data_b=$(stat -c%s "$MODEL_BASE/$subdir/model.onnx_data")
            total_mb=$(( (onnx_b + data_b) / 1024 / 1024 ))
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

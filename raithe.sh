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
ORT_TARBALL="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
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
# ORT SHARED LIBRARY — auto-download if missing
# ==============================================================================

step "Locating ONNX Runtime shared library"

ORT_SO=$(find "$ORT_CACHE" -name "libonnxruntime.so*" -not -name "*.lock" 2>/dev/null \
    | sort -V | tail -n1 || true)

if [[ -z "$ORT_SO" ]]; then
    warn "libonnxruntime.so not found — downloading ORT v${ORT_VERSION}"
    mkdir -p "$ORT_CACHE"
    TMP_TAR="/tmp/${ORT_TARBALL}"
    wget -q --show-progress "$ORT_URL" -O "$TMP_TAR" \
        || fail "Failed to download ORT from $ORT_URL"
    tar -xzf "$TMP_TAR" -C /tmp
    cp "/tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so.${ORT_VERSION}" \
        "$ORT_CACHE/"
    ln -sf "$ORT_CACHE/libonnxruntime.so.${ORT_VERSION}" \
           "$ORT_CACHE/libonnxruntime.so"
    rm -f "$TMP_TAR"
    ORT_SO="$ORT_CACHE/libonnxruntime.so.${ORT_VERSION}"
    ok "ORT downloaded and cached"
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
        text-classification)  echo "AutoModelForSequenceClassification" ;;# BINARY CHECK
170
# ==============================================================================
171
​
172
if [[ ! -x "$BINARY" ]]; then
173
    fail "raithe-se binary not found — run: cargo build --release"
174
fi
175
​
176
# ==============================================================================
177
# MODEL INSTALLATION / VALIDATION
178
# ==============================================================================
179
​
180
declare -A MODEL_MIN_BYTES=(
181
    [embedder]=$((1200  * 1024 * 1024))
182
    [reranker]=$((1800  * 1024 * 1024))
183
    [generator]=$((10000 * 1024 * 1024))
184
)
185
​
186
get_task() {
187
    local model_id="$1"
188
    if [[ "$model_id" == *"bge-large"* || "$model_id" == *"bge-base"* \
189
       || "$model_id" == *"bge-small"* ]]; then
190
        echo "feature-extraction"; return
191
    fi
192
    if [[ "$model_id" == *"reranker"* || "$model_id" == *"cross-encoder"* ]]; then
193
        echo "text-classification"; return
194
    fi
195
    if [[ "$model_id" == *"flan"* || "$model_id" == *"t5"* ]]; then
196
        echo "text2text-generation"; return
197
    fi
198
    if [[ "$model_id" == *"Qwen"* || "$model_id" == *"Llama"* \
199
       || "$model_id" == *"Mistral"* ]]; then
200
        echo "text-generation"; return
201
    fi
202
    echo "text-classification"
203
}
204
​
205
get_model_class() {
206
    local task="$1"
207
    case "$task" in
208
        text-generation)      echo "AutoModelForCausalLM" ;;
209
        text2text-generation) echo "AutoModelForSeq2SeqLM" ;;
210
        text-classification)  echo "AutoModelForSequenceClassification" ;;
211
        feature-extraction)   echo "AutoModel" ;;
212
        *)                    echo "AutoModel" ;;
213
    esac
214
}
215
​
216
model_is_valid() {
217
    local subdir="$1"
218
    local dest="$MODEL_BASE/$subdir"
219
    [[ -f "$dest/model.onnx" && -f "$dest/tokenizer.json" ]] || return 1
220
    local onnx_b data_b total_b
221
    onnx_b=$(stat -c%s "$dest/model.onnx")
222
    data_b=0
223
    [[ -f "$dest/model.onnx_data" ]] && data_b=$(stat -c%s "$dest/model.onnx_data")
224
    total_b=$(( onnx_b + data_b ))
225
    [[ $total_b -ge ${MODEL_MIN_BYTES[$subdir]} ]] || return 1
226
    return 0
227
}
228
​
229
export_model() {
230
    local subdir="$1"
231
    local hf_id="$2"
232
​
233
    local dest="$MODEL_BASE/$subdir"
234
    local work="$dest/_work"
235
    local task model_class
236
    task=$(get_task "$hf_id")
237
    model_class=$(get_model_class "$task")
238
​
239
    if [[ "$FORCE_INSTALL" -eq 0 ]] && model_is_valid "$subdir"; then
240
        local onnx_b data_b total_mb
241
        onnx_b=$(stat -c%s "$dest/model.onnx")
242
        data_b=0
243
        [[ -f "$dest/model.onnx_data" ]] && data_b=$(stat -c%s "$dest/model.onnx_data")
244
        total_mb=$(( (onnx_b + data_b) / 1024 / 1024 ))
245
        skip "$subdir ($hf_id) — ${total_mb} MB"
246
        return
247
    fi
248
​
249
    step "Exporting $subdir → $hf_id  [$task]  $model_class"
250
​
251
    rm -rf "$work"
252
    mkdir -p "$work" "$dest"
253

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
        fail "download failed for $hf_id"
    fi
    ok "download complete"

    # ── Stage 2 — ONNX export ───────────────────────────────────────────────
    sub "Stage 2/3 — exporting to ONNX ($task)"

    rm -rf "$work/onnx"
    mkdir -p "$work/onnx"

    if [[ "$task" == "text-generation" ]]; then
        # >2GB protobuf wall is hit during torch.onnx shape inference pass,
        # BEFORE file write — so use_external_data_format=True alone fails.
        # Fix: disable shape inference globally, then export to file path,
        # then re-save with proper external data split via onnx.save_model.
        set +e
        OMP_NUM_THREADS=$CPU_CORES MKL_NUM_THREADS=$CPU_CORES \
        python3 - >"$work/export.log" 2>&1 <<PYEOF
import os, sys, warnings, logging
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"]           = "1"
os.environ["OMP_NUM_THREADS"]        = "${CPU_CORES}"
os.environ["MKL_NUM_THREADS"]        = "${CPU_CORES}"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("optimum").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

try:
    import torch
    import torch.onnx._globals as onnx_globals
    import onnx
    from transformers import AutoModelForCausalLM, AutoConfig
    from optimum.exporters.tasks import TasksManager
except ImportError as e:
    print(f"IMPORT_ERROR: {e}", file=sys.stderr); sys.exit(2)

work = Path("${work}")
out_dir = work / "onnx"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "model.onnx"

# Kill the shape inference pass — this is what triggers the 2GB protobuf serialize
onnx_globals.GLOBALS.onnx_shape_inference = False

try:
    hf_config = AutoConfig.from_pretrained(str(work), trust_remote_code=True)
    onnx_config_ctor = TasksManager._SUPPORTED_MODEL_TYPE["qwen2"]["onnx"]["text-generation-with-past"]
    onnx_config = onnx_config_ctor(hf_config, use_past=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(work), trust_remote_code=True, torch_dtype=torch.float32,
    )
    model.eval()

    dummy_inputs = onnx_config.generate_dummy_inputs(framework="pt")
    input_names = list(dummy_inputs.keys())
    output_names = list(onnx_config.outputs.keys())
    dynamic_axes = {**onnx_config.inputs, **onnx_config.outputs}

    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            str(out_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=False,
            onnx_shape_inference=False,
            use_external_data_format=True,
        )

    # Re-save to guarantee proper external data layout (single file, no threshold)
    m = onnx.load(str(out_path), load_external_data=True)
    onnx.save_model(
        m,
        str(out_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=0,
        convert_attribute=False,
    )
    print("OK", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"EXPORT_ERROR: {type(e).__name__}: {e}", file=sys.stderr); sys.exit(3)
PYEOF
        export_rc=$?
        set -e
        if [[ $export_rc -ne 0 ]]; then
            echo -e "${RED}✗ ONNX export failed for $hf_id — exit $export_rc${RESET}" >&2
            cat "$work/export.log" >&2
            fail "ONNX export failed for $hf_id ($task)"
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

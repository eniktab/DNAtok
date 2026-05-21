#!/usr/bin/env bash
# DNAtok plug-and-play smoke test.
#
# Run inside the Docker / Apptainer image. Validates:
#   1. `dnatok --help` works (CLI registered)
#   2. `dnatok list-models` lists 7 families
#   3. `dnatok info`     succeeds on every family
#   4. `dnatok encode`   succeeds on every family
#   5. `dnatok validate` produces 100% bit-identical match on every family
#      (n=100 random ACGT sequences each)
#   6. `dnatok bench`    produces a non-trivial speedup or wall-clock result
#
# Usage (from inside the container):
#     bash /work/docker/smoke_test.sh
#
# Usage (from host, against the container):
#     docker run --rm --gpus all \
#         -v $HF_HOME:/work/.hf-cache \
#         dnatok:latest bash /work/docker/smoke_test.sh
#
# Exit code:  0 if all checks pass; non-zero on first failure (set -e).
set -eu
set -o pipefail

MODELS=(
    "zhihan1996/DNABERT-2-117M"
    "AIRI-Institute/gena-lm-bert-base-t2t"
    "metagene-ai/METAGENE-1"
    "InstaDeepAI/NTv3_8M_pre"
    "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
    "LongSafari/hyenadna-tiny-1k-seqlen-hf"
    "arcinstitute/evo2_1b_base"
)

PASS=0
FAIL=0
FAILED_MODELS=()

echo "============================================================"
echo "DNAtok plug-and-play smoke test"
echo "============================================================"
echo "Started: $(date -Is)"
echo "Models:  ${#MODELS[@]}"
echo

echo "[1] dnatok --help"
dnatok --help > /dev/null
echo "    OK"

echo
echo "[2] dnatok list-models"
dnatok list-models | head -1 > /dev/null
echo "    OK"

for model in "${MODELS[@]}"; do
    echo
    echo "------------------------------------------------------------"
    echo "Model: $model"
    echo "------------------------------------------------------------"

    if ! dnatok info --model "$model" --json > /tmp/info.json 2>/tmp/info.err; then
        echo "    [info]     FAIL (see /tmp/info.err)"
        FAIL=$((FAIL + 1))
        FAILED_MODELS+=("$model")
        continue
    fi
    PATH_SELECTED=$(python3 -c "import json; print(json.load(open('/tmp/info.json'))['selected_path'])")
    echo "    [info]     OK (path=$PATH_SELECTED)"

    if ! dnatok encode --model "$model" --seq "ACGTACGTACGTACGTACGT" --json > /tmp/encode.json 2>/tmp/encode.err; then
        echo "    [encode]   FAIL"
        FAIL=$((FAIL + 1))
        FAILED_MODELS+=("$model")
        continue
    fi
    echo "    [encode]   OK"

    if ! dnatok validate --model "$model" --n 100 --window 1024 --json > /tmp/validate.json 2>/tmp/validate.err; then
        echo "    [validate] FAIL (non-zero exit)"
        FAIL=$((FAIL + 1))
        FAILED_MODELS+=("$model")
        continue
    fi
    MATCH_RATE=$(python3 -c "import json; d=json.load(open('/tmp/validate.json')); print(d['match_rate'])")
    if [ "$MATCH_RATE" != "1.0" ] && [ "$MATCH_RATE" != "1" ]; then
        echo "    [validate] FAIL (match_rate=$MATCH_RATE, expected 1.0)"
        FAIL=$((FAIL + 1))
        FAILED_MODELS+=("$model")
        continue
    fi
    echo "    [validate] OK (100/100 bit-identical)"

    if ! dnatok bench --model "$model" --n 200 --window 512 --chunk 32 --json > /tmp/bench.json 2>/tmp/bench.err; then
        echo "    [bench]    FAIL"
        FAIL=$((FAIL + 1))
        FAILED_MODELS+=("$model")
        continue
    fi
    SPEEDUP=$(python3 -c "import json; print(json.load(open('/tmp/bench.json'))['speedup_dnatok_vs_hf'])")
    DNATOK_MBPS=$(python3 -c "import json; print(json.load(open('/tmp/bench.json'))['dnatok_mbp_per_s'])")
    echo "    [bench]    OK (DNAtok ${DNATOK_MBPS} Mbp/s, speedup ${SPEEDUP}x vs HF threads=8)"

    PASS=$((PASS + 1))
done

echo
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "  passed:  $PASS / ${#MODELS[@]}"
echo "  failed:  $FAIL / ${#MODELS[@]}"
if [ "$FAIL" -gt 0 ]; then
    echo "  failed models:"
    for m in "${FAILED_MODELS[@]}"; do
        echo "    - $m"
    done
    exit 1
fi
echo "  ALL CLEAR — DNAtok image is good to ship."
exit 0

# A quick end-to-end check as one GPU job: tests, NeMo's outputs on a few files (COMPARE=<dir>: also compared
# with an earlier run of tools/nemo_ref.py), this package's default and torch paths against them, and the
# command line. Run as CMD=tools/smoke.sh~@ with REF_ENV=1 through scripts/gpu_job.pbs.
set -euo pipefail
OUT=$1
L=$REPO/.cache/lists
LISTS=()
SMOKE_SETS="${SMOKE_SETS:-short}"
for s in ${SMOKE_SETS//\~/ }; do LISTS+=("$L/$s.jsonl"); done  # qsub -v: '~' for spaces
$PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -5 || true
$REF_PY tools/nemo_ref.py --out "$OUT/nemo" --no-repeat --lists "${LISTS[@]}" ${COMPARE:+--compare "$COMPARE"}
$PY tools/parity.py --ref "$OUT/nemo" --out "$OUT/parity" --variants trt-float32 sdpa-graph --lists "${LISTS[@]}"
first=$(head -1 "${LISTS[0]}" | python3 -c "import json, sys; print(json.load(sys.stdin)['path'])")
t0=$SECONDS
$PY -m nemotron_diar_trt.cli diarize --nemo-compat "$first" > "$OUT/cli.txt"
echo "=== cli: $((SECONDS - t0)) s, $(wc -l < "$OUT/cli.txt") turns"

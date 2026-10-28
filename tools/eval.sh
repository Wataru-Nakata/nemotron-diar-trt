# The full evaluation as one GPU job: NeMo's outputs, this package's variants against them (agreement and turns
# for tools/der.py), and the speed of both on the same node. Run as CMD=tools/eval.sh~@ with REF_ENV=1 through
# scripts/gpu_job.pbs ($PY: the TensorRT env, $REF_PY: the NeMo env). Lists come from tools/prepare_data.py.
set -euo pipefail
OUT=$1
L=$REPO/.cache/lists
lists() { local out=(); for s in $1; do [ -s "$L/$s.jsonl" ] && out+=("$L/$s.jsonl"); done; echo "${out[@]}"; }
EVAL_SETS="${EVAL_SETS:-ami ami_long csj voxconverse_test voxconverse_dev}" BENCH_SETS="${BENCH_SETS:-bench50 ami_long}"
EVAL=$(lists "${EVAL_SETS//\~/ }")  # qsub -v: '~' for spaces
BENCH=$(lists "${BENCH_SETS//\~/ }")
$PY -m pytest -q -p no:cacheprovider tests 2>&1 | tail -5 || true
# shellcheck disable=SC2086
$REF_PY tools/nemo_ref.py --out "$OUT/nemo" --no-repeat --lists $EVAL
# shellcheck disable=SC2086
$PY tools/parity.py --ref "$OUT/nemo" --out "$OUT/parity" --no-probs --variants trt-float32 trt-float16 sdpa-graph \
  --lists $EVAL || echo "=== parity failed"
# shellcheck disable=SC2086
$PY tools/bench.py --lists $BENCH --out "$OUT/bench/bench.json" --variants trt-float32 trt-float16 torch \
  || echo "=== bench failed"
# shellcheck disable=SC2086
$REF_PY tools/nemo_ref.py --out "$OUT/nemo_bench" --lists $BENCH || echo "=== NeMo bench failed"

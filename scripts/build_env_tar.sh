#!/bin/bash
# Build a self-contained Python environment on node-local disk and pack it into one tar.zst. Shared
# filesystems are slow with the ~100k small files of torch/NeMo, so jobs unpack these archives into
# $PBS_LOCALDIR instead of running from a .venv (scripts/gpu_job.pbs).
#
#   scripts/build_env_tar.sh main    # this package and its deps -> .cache/envs/nemotron-env.tar.zst
#   scripts/build_env_tar.sh ref     # NeMo reference (tools/nemo_env) -> .cache/envs/nemo-ref.tar.zst
#
# The env is a relocatable python-build-standalone interpreter with packages installed directly into it:
# <dir>/bin/python3 runs from wherever it is extracted.
# The build dir (~13 GB with the uv cache; /tmp on a login node is shared) is removed once the tar is written;
# KEEP_BUILD=1 keeps it.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
KIND="${1:-main}"
BUILD_ROOT="${BUILD_ROOT:-${TMPDIR:-/tmp}/nemotron-env-build-$USER}"
OUT_DIR="$REPO/.cache/envs"
ZSTD="$(command -v zstd || echo "$HOME/miniconda3/bin/zstd")"

case $KIND in
  ref) NAME=nemo-ref; PYVER=3.12.9 ;;  # the interpreter the reference numbers were measured with
  main) NAME=nemotron-env; PYVER=3.12 ;;
  *) echo "usage: $0 [ref|main]" >&2; exit 2 ;;
esac

export UV_CACHE_DIR="$BUILD_ROOT/uv-cache"  # keep the cache off Lustre too
export UV_PYTHON_INSTALL_DIR="$BUILD_ROOT/pythons"
ENV="$BUILD_ROOT/$NAME"
rm -rf "$ENV"
mkdir -p "$BUILD_ROOT" "$OUT_DIR"

uv python install "$PYVER"
# resolve symlinks: uv also creates a minor-version alias dir that is a symlink to the real install
cp -a "$(dirname "$(dirname "$(readlink -f "$(uv python find "$PYVER" --managed-python)")")")" "$ENV"
test ! -L "$ENV"
rm -f "$ENV"/lib/python3.12/EXTERNALLY-MANAGED
PY="$ENV/bin/python3"

if [ "$KIND" = ref ]; then
  (cd "$REPO/tools/nemo_env" && uv export --frozen --no-hashes --no-emit-project -o "$BUILD_ROOT/$NAME.txt")
  uv pip install --python "$PY" -r "$BUILD_ROOT/$NAME.txt"
else
  (cd "$REPO" && uv lock && uv export --frozen --no-hashes --no-emit-project --all-groups -o "$BUILD_ROOT/$NAME.txt")
  uv pip install --python "$PY" -r "$BUILD_ROOT/$NAME.txt"
  uv pip install --python "$PY" --no-deps -e "$REPO"
fi
"$PY" -c "import torch, sys; print(sys.version.split()[0], 'torch', torch.__version__, torch.version.cuda)"

tar -C "$BUILD_ROOT" -cf - "$NAME" | "$ZSTD" -T16 -3 -q -o "$OUT_DIR/$NAME.tar.zst.tmp" -f
mv "$OUT_DIR/$NAME.tar.zst.tmp" "$OUT_DIR/$NAME.tar.zst"
ls -lh "$OUT_DIR/$NAME.tar.zst"
if [ "${KEEP_BUILD:-0}" = 1 ]; then
  echo "local copy (usable on this node): $PY"
else
  rm -rf "$BUILD_ROOT"
fi

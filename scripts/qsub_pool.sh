#!/bin/bash
# Submit scripts/gpu_job.pbs from the current directory on one of N single-GPU lanes (default 8,
# NEMOTRON_GPU_LANES): an idle lane starts the job at once, otherwise it queues behind the job of the lane
# picked in turn. At most N jobs run at the same time, however many people/agents submit.
#   scripts/qsub_pool.sh "CMD=tools/x.py~--out~@,REF_CMD=..." [walltime, default 01:00:00]
# Prints the job id. Wait for it with: until ! qstat <id> >/dev/null 2>&1; do sleep 30; done
# Two jobs from one checkout that build the same engines must not run at once: let the first one build.
# The cluster account comes from the environment, or else from the main checkout's .cache/qsub.env (not in git):
#   NEMOTRON_QSUB_GROUP (qsub -P), NEMOTRON_QSUB_QUEUE (qsub -q), NEMOTRON_QSUB_RTYPE (the job's RTYPE resource type)
# The lanes and qsub.env are the main checkout's, so every git worktree of it shares one pool.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
common="$(git -C "$root" rev-parse --git-common-dir 2>/dev/null || true)"  # worktrees: the main checkout's .git
[ -n "$common" ] && root="$(cd "$root" && cd "$common/.." && pwd)"
env_file="${NEMOTRON_QSUB_ENV:-$root/.cache/qsub.env}"
if [ -f "$env_file" ]; then  # only for what the environment leaves unset
  for v in NEMOTRON_QSUB_GROUP NEMOTRON_QSUB_QUEUE NEMOTRON_QSUB_RTYPE; do
    [ -n "${!v:-}" ] || printf -v "$v" '%s' "$(. "$env_file" && echo "${!v:-}")"
  done
fi
: "${NEMOTRON_QSUB_GROUP:?set NEMOTRON_QSUB_GROUP (qsub -P), e.g. in .cache/qsub.env}"
: "${NEMOTRON_QSUB_QUEUE:?set NEMOTRON_QSUB_QUEUE (qsub -q)}"
: "${NEMOTRON_QSUB_RTYPE:?set NEMOTRON_QSUB_RTYPE (resource type)}"
CHAIN_DIR="${NEMOTRON_GPU_CHAIN:-$root/.cache/gpu_chain}"  # point several checkouts at one dir to share a cap
LANES="${NEMOTRON_GPU_LANES:-8}"
mkdir -p "$CHAIN_DIR"
exec 9>"$CHAIN_DIR/lock"
flock 9
lane="" dep=()
for ((i = 0; i < LANES; i++)); do
  last=$(cat "$CHAIN_DIR/lane$i" 2>/dev/null || true)
  if [ -z "$last" ] || ! qstat "$last" >/dev/null 2>&1; then
    lane=$i
    break
  fi
done
if [ -z "$lane" ]; then  # every lane busy: queue behind the lanes in turn
  lane=$(($(cat "$CHAIN_DIR/turn" 2>/dev/null || echo 0) % LANES))
  echo $((lane + 1)) > "$CHAIN_DIR/turn"
  dep=(-W "depend=afterany:$(cat "$CHAIN_DIR/lane$lane")")
fi
id=$(qsub -P "$NEMOTRON_QSUB_GROUP" -q "$NEMOTRON_QSUB_QUEUE" "${dep[@]}" -v "RTYPE=$NEMOTRON_QSUB_RTYPE,$1" -l select=1 \
  -l "walltime=${2:-01:00:00}" \
  scripts/gpu_job.pbs)
echo "$id" > "$CHAIN_DIR/lane$lane"
echo "$(date '+%F %T') lane$lane $id $PWD $1" >> "$CHAIN_DIR/log"
echo "$id"

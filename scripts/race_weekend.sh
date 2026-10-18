#!/usr/bin/env bash
# One race-weekend run (P8-06, P8-07): sync, forecast, score, push.
#
# Every step is idempotent, so the schedule can fire this as often as it likes:
#   - predict --latest is a no-op when no weekend is past qualifying, or the forecast is
#     locked after lights out (ADR-0011);
#   - report score --latest is a no-op when the newest race has no published forecast or is
#     already scored;
#   - nothing is pushed when nothing was committed.
#
# Usage: scripts/race_weekend.sh [all|predict|score] [season] [round]
#   With a round, that race is forecast or scored instead of the newest one.
# Environment: F1SIM_MODEL (default simulator), LOG_DIR (default logs).
set -euo pipefail

STAGE="${1:-all}"
SEASON="${2:-$(date -u +%Y)}"
ROUND="${3:-}"
MODEL="${F1SIM_MODEL:-simulator}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/race-weekend-$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ -n "$ROUND" ]]; then
  TARGET=(--round "$ROUND")
else
  TARGET=(--latest)
fi

run() {
  echo "::group::$*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  echo "::endgroup::" | tee -a "$LOG"
}

echo "race weekend: stage=$STAGE season=$SEASON round=${ROUND:-latest} model=$MODEL" \
  | tee -a "$LOG"

# The previous season feeds the models' priors early in the year
run uv run --frozen f1sim data sync --season "$((SEASON - 1))"
run uv run --frozen f1sim data sync --season "$SEASON"

if [[ "$STAGE" == "all" || "$STAGE" == "predict" ]]; then
  run uv run --frozen f1sim predict --season "$SEASON" "${TARGET[@]}" --model "$MODEL"
fi
if [[ "$STAGE" == "all" || "$STAGE" == "score" ]]; then
  run uv run --frozen f1sim report score --season "$SEASON" "${TARGET[@]}"
fi

if [[ -n "$(git log --oneline '@{upstream}..HEAD' 2>/dev/null)" ]]; then
  # Another run may have pushed meanwhile; prediction commits touch only predictions/
  run git pull --rebase --quiet
  run git push --quiet
  # Tells the workflow to redeploy the site (pushes made with its token trigger nothing)
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then echo "pushed=true" >> "$GITHUB_OUTPUT"; fi
else
  echo "nothing new to push" | tee -a "$LOG"
fi

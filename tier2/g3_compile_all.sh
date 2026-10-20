#!/usr/bin/env bash
# Runs inside the pod from tier2/. Compiles the resident-cache stage graphs, up to MAX_JOBS at a time (default 4).
# The stage split comes from LIVEVID_STAGES (default 0-8,8-16,16-23,23-30); see g3_stage.py for the other options.
set -uo pipefail
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
L=/workspace/livevid/logs/tier2
MAX_JOBS=${MAX_JOBS:-4}
mkdir -p "$L"
logs=()
while read -r a b; do
  while [[ $(jobs -rp | wc -l) -ge $MAX_JOBS ]]; do wait -n; done
  log=$L/g3_stage_${a}_${b}.log
  logs+=("$log")
  python -u g3_stage.py "$a" "$b" > "$log" 2>&1 &
done < <(python g3_stage.py list)
wait
grep -h "COMPILE OK\|skip\|Error" "${logs[@]}"
echo STAGES DONE

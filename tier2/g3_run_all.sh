#!/usr/bin/env bash
# Runs inside the pod from tier2/. Waits for the stage graphs, warms the caches, then runs the pipeline with one
# core per stage and the same graphs sequentially in one process on 1 core. Uses the same LIVEVID_* settings as
# g3_compile_all.sh.
set -uo pipefail
n=$(python g3_stage.py list | wc -l)
paths=$(python -c 'from g3_stage import STAGES, stage_path; print(" ".join(str(stage_path(a, b)) for a, b in STAGES))')
until ls $paths > /dev/null 2>&1; do sleep 5; done
sleep 3
NEURON_RT_VISIBLE_CORES=0 python -u g3_pipeline.py warm 2>&1 | grep -v -i warn
python -u g3_pipeline.py run --inflight 6 --chunks 120 --name ${n}core 2>&1 | grep -v -i warn
python -u g3_pipeline.py run1 --chunks 40 --name 1core 2>&1 | grep -v -i warn
echo G3 DONE

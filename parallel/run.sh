#!/usr/bin/env bash
set -euo pipefail

start_ns=$(date +%s%N)

python setup.py -q build_ext --inplace
python -m parallel.run

end_ns=$(date +%s%N)
elapsed_ms=$(((end_ns - start_ns) / 1000000))
printf 'elapsed: %d.%03ds\n' "$((elapsed_ms / 1000))" "$((elapsed_ms % 1000))"

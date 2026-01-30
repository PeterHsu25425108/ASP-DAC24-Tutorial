#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <input_dir> <platform_dir> <output_dir> <top_module>" >&2
  exit 2
fi

input_dir=$1
platform_dir=$2
output_dir=$3
top_module=$4

source /opt/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
openroad -exit -python src/demo2_gate_sizing.py "$input_dir" "$platform_dir" "$output_dir" "$top_module"

# for profiling, uncomment the following line and comment out the above line
# openroad -exit -python -m cProfile -o program.prof src/demo2_gate_sizing.py "$input_dir" "$platform_dir" "$output_dir" "$top_module"
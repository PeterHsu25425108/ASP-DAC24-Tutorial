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

openroad -exit -python session1/demo2_gate_sizing.py "$input_dir" "$platform_dir" "$output_dir" "$top_module"
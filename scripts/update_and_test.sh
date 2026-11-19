#!/bin/bash
set -euo pipefail

MODE="${1:-1}"
if [[ ! "$MODE" =~ ^[1-4]$ ]]; then
  echo "invalid mode: $MODE (expected 1-4)" >&2
  exit 1
fi

printf '%s\n' "$MODE" | nc 127.0.0.1 30725

echo "blabla" >> /root/cool-file.txt

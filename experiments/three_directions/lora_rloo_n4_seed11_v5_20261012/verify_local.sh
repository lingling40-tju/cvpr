#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 sources/verify_positive_compact.py --manifest development256.json --compact evaluation/sft_paired_episodes.jsonl --report evaluation/sft_paired_report.json --validators validation --output validation/local_sft_recount.json
python3 sources/verify_positive_compact.py --manifest development256.json --compact evaluation/grpo_paired_episodes.jsonl --report evaluation/grpo_paired_report.json --validators validation --output validation/local_grpo_recount.json

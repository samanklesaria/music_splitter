#!/usr/bin/env bash
# Download the JaCappella dataset from Hugging Face.
#
# Prerequisites:
#   huggingface-cli login
#   (accept dataset terms at https://huggingface.co/datasets/jaCappella/jaCappella)

set -euo pipefail

OUTPUT_DIR="data/jacappella"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "Downloading JaCappella (~4.3 GB) to $OUTPUT_DIR ..."
echo "Note: you must be logged in and have accepted the dataset terms."
echo "  https://huggingface.co/datasets/jaCappella/jaCappella"
echo ""

huggingface-cli download jaCappella/jaCappella \
    --repo-type dataset \
    --local-dir "$OUTPUT_DIR"

echo "Done. Dataset saved to $OUTPUT_DIR"

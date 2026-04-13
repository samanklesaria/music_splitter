#!/usr/bin/env bash
# Download all datasets for vocal harmony separation training.
#
# Usage:
#   ./scripts/download_all.sh                        # download everything to ./data/
#   ./scripts/download_all.sh --output-dir /my/data  # download everything to /my/data/
#
# Prerequisites:
#   huggingface-cli login  (for JaCappella)
#   pip install acappella_info  (for Acappella)
#   yt-dlp must be installed (for Acappella)

set -euo pipefail
cd "$(dirname "$0")/.."

OUTPUT_DIR="/space/samanklesaria/data"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

echo "=== Downloading JaCappella (4.3 GB) ==="
bash scripts/download_jacappella.sh --output-dir "$OUTPUT_DIR/jacappella"

echo ""
echo "=== Downloading Dagstuhl ChoirSet (5.1 GB) ==="
bash scripts/download_dagstuhl.sh --output-dir "$OUTPUT_DIR/dagstuhl_choirset"

echo ""
echo "=== Downloading MUSDB18-HQ (22.7 GB) ==="
bash scripts/download_musdb18hq.sh --output-dir "$OUTPUT_DIR/musdb18hq"

echo ""
echo "=== Downloading Acappella from YouTube ==="
bash scripts/download_acappella.sh --output-dir "$OUTPUT_DIR/acappella"

echo ""
echo "All downloads complete. Data is in $OUTPUT_DIR/"

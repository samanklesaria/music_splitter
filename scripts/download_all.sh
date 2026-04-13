#!/usr/bin/env bash
# Download all datasets for vocal harmony separation training.
#
# Usage:
#   ./scripts/download_all.sh          # download everything
#
# Prerequisites:
#   huggingface-cli login  (for JaCappella)
#   pip install acappella_info  (for Acappella)
#   yt-dlp must be installed (for Acappella)

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Downloading JaCappella (4.3 GB) ==="
bash scripts/download_jacappella.sh

echo ""
echo "=== Downloading Dagstuhl ChoirSet (5.1 GB) ==="
bash scripts/download_dagstuhl.sh

echo ""
echo "=== Downloading MUSDB18-HQ (22.7 GB) ==="
bash scripts/download_musdb18hq.sh

echo ""
echo "=== Downloading Acappella from YouTube ==="
bash scripts/download_acappella.sh

echo ""
echo "All downloads complete. Data is in ./data/"

#!/usr/bin/env bash
# Download MUSDB18-HQ from Zenodo for pretraining.
# No authentication required. License: non-commercial/academic use only.
# Source: https://zenodo.org/records/3338373

set -euo pipefail

OUTPUT_DIR="data/musdb18hq"
KEEP_ZIP=false
URL="https://zenodo.org/records/3338373/files/musdb18hq.zip"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --keep-zip)   KEEP_ZIP=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
ZIP="$OUTPUT_DIR/musdb18hq.zip"

echo "Downloading MUSDB18-HQ (~22.7 GB) — this will take a while ..."
echo "  Source: $URL"
curl -L -o "$ZIP" --progress-bar "$URL"

echo "Extracting ..."
unzip -q "$ZIP" -d "$OUTPUT_DIR"

if [ "$KEEP_ZIP" = false ]; then
    rm "$ZIP"
    echo "Removed zip file."
fi

echo "Done. Dataset saved to $OUTPUT_DIR"

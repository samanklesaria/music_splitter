#!/usr/bin/env bash
# Download the Dagstuhl ChoirSet (DCS) from Zenodo.
# No authentication required. License: CC BY 4.0.
# Source: https://zenodo.org/records/3897182

set -euo pipefail

OUTPUT_DIR="data/dagstuhl_choirset"
KEEP_ZIP=false
URL="https://zenodo.org/records/3897182/files/DagstuhlChoirSet.zip"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --keep-zip)   KEEP_ZIP=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
ZIP="$OUTPUT_DIR/DagstuhlChoirSet.zip"

echo "Downloading Dagstuhl ChoirSet (~5.1 GB) ..."
echo "  Source: $URL"
curl -L -o "$ZIP" --progress-bar "$URL"

echo "Extracting ..."
unzip -q "$ZIP" -d "$OUTPUT_DIR"

if [ "$KEEP_ZIP" = false ]; then
    rm "$ZIP"
    echo "Removed zip file."
fi

echo "Done. Dataset saved to $OUTPUT_DIR"

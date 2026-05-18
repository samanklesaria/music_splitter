#!/usr/bin/env bash
# Download the Acappella dataset from YouTube.
#
# Reads video IDs and timestamps from csv_files/full_dataset.csv,
# then downloads each clip as WAV using yt-dlp.
#
# Prerequisites:
#   yt-dlp (https://github.com/yt-dlp/yt-dlp)
#   miller / mlr (https://miller.readthedocs.io/)
#
# Source: https://ipcv.github.io/Acappella/
# License: CC BY 4.0 (metadata); original video copyright applies.

set -euo pipefail

OUTPUT_DIR="data/acappella"
PKG_DIR=$(uv run python -c "import acappella_info, os; print(os.path.dirname(acappella_info.__file__))" 2>/dev/null)
if [[ -z "$PKG_DIR" ]]; then
    echo "Error: acappella_info not found."
    exit 1
fi
CSV_FILE="$PKG_DIR/csv_files/full_dataset.csv"
MAX_LINES=""  # empty = no limit

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --csv)        CSV_FILE="$2"; shift 2 ;;
        --lines)      MAX_LINES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if ! command -v yt-dlp &>/dev/null; then
    echo "Error: yt-dlp not found."
    exit 1
fi

if ! command -v mlr &>/dev/null; then
    echo "Error: miller (mlr) not found. See https://miller.readthedocs.io/"
    exit 1
fi

if [[ ! -f "$CSV_FILE" ]]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

download_video() {
    local video_id="$1"
    local start="$2"
    local end="$3"
    local out="$OUTPUT_DIR/$video_id.wav"

    [[ -f "$out" ]] && return 0

    if yt-dlp -x --audio-format wav --output "$out" --quiet --download-sections "*${start}-${end}" -- "$video_id"; then
        return 0
    else
        return 1
    fi
}

mlr_cmd=(mlr --csv --otsv --headerless-csv-output cut -f ID,Init,Fin)
if [[ -n "$MAX_LINES" ]]; then
    mlr_cmd+=(then head -n "$MAX_LINES")
fi
mlr_cmd+=("$CSV_FILE")

succeeded=0
failed=0
count=0

while IFS=$'\t' read -r video_id start end; do
    [[ -z "$video_id" ]] && continue
    count=$((count + 1))

    # echo "$video_id" "$start" "$end"
    if download_video "$video_id" "$start" "$end"; then
        succeeded=$((succeeded + 1))
    else
        failed=$((failed + 1))
    fi

    if (( count % 10 == 0 )); then
        echo "$count processed  (ok=$succeeded, fail=$failed)"
    fi
done < <("${mlr_cmd[@]}")

echo "Done: $succeeded ok, $failed failed. Dataset saved to $OUTPUT_DIR"

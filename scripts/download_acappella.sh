#!/usr/bin/env bash
# Download the Acappella dataset from YouTube.
#
# Reads video IDs and timestamps from the acappella_info pip package,
# then downloads each clip as WAV using yt-dlp.
#
# Prerequisites:
#   pip install acappella_info
#   yt-dlp (https://github.com/yt-dlp/yt-dlp)
#
# Source: https://ipcv.github.io/Acappella/
# License: CC BY 4.0 (metadata); original video copyright applies.

set -euo pipefail

OUTPUT_DIR="data/acappella"
SPLITS="train val_seen test_seen test_unseen"
MAX_PER_SPLIT=""  # empty = no limit

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --splits)        SPLITS="$2"; shift 2 ;;
        --max-per-split) MAX_PER_SPLIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Locate the acappella_info package data directory via pip
PKG_LOCATION=$(pip show acappella_info 2>/dev/null | awk '/^Location:/ {print $2}')
if [[ -z "$PKG_LOCATION" ]]; then
    echo "Error: acappella_info not found."
    exit 1
fi
PKG_DIR="$PKG_LOCATION/acappella_info"

if ! command -v yt-dlp &>/dev/null; then
    echo "Error: yt-dlp not found."
    exit 1
fi

download_video() {
    local video_id="$1"
    local start="$2"
    local end="$3"
    local split_dir="$4"
    local out="$split_dir/$video_id.wav"

    [[ -f "$out" ]] && return 0

    local cmd=(yt-dlp -x --audio-format wav --output "$out" --quiet)

    if [[ -n "$start" && -n "$end" ]]; then
        cmd+=(--download-sections "*${start}-${end}")
    fi

    cmd+=("https://www.youtube.com/watch?v=$video_id")

    if timeout 120 "${cmd[@]}" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

parse_and_download_csv() {
    local csv="$1"
    local split_dir="$2"
    local max="$3"
    local succeeded=0 failed=0 count=0

    # Use awk to parse CSV (skip header, extract id/start/end columns)
    # Tries common column names: video_id/YouTube_ID/id, start/start_time, end/end_time
    while IFS=',' read -r video_id start end; do
        [[ "$video_id" == "video_id" || "$video_id" == "YouTube_ID" || "$video_id" == "id" ]] && continue
        [[ -z "$video_id" ]] && continue
        [[ -n "$max" && $count -ge $max ]] && break
        count=$((count + 1))

        if download_video "$video_id" "$start" "$end" "$split_dir"; then
            succeeded=$((succeeded + 1))
        else
            failed=$((failed + 1))
        fi

        if (( count % 10 == 0 )); then
            echo "  $count processed  (ok=$succeeded, fail=$failed)"
        fi
    done < <(awk -F',' 'NR==1 {
        for (i=1; i<=NF; i++) {
            if ($i=="video_id" || $i=="YouTube_ID" || $i=="id") vid=i
            if ($i=="start" || $i=="start_time") st=i
            if ($i=="end" || $i=="end_time") en=i
        }
        next
    } { print $vid "," $st "," $en }' "$csv")

    echo "  Done: $succeeded ok, $failed failed"
}

parse_and_download_json() {
    local json="$1"
    local split_dir="$2"
    local max="$3"

    if ! command -v jq &>/dev/null; then
        echo "  Warning: jq not found, skipping $json"
        return
    fi

    local succeeded=0 failed=0 count=0

    while IFS=$'\t' read -r video_id start end; do
        [[ -z "$video_id" ]] && continue
        [[ -n "$max" && $count -ge $max ]] && break
        count=$((count + 1))

        if download_video "$video_id" "$start" "$end" "$split_dir"; then
            succeeded=$((succeeded + 1))
        else
            failed=$((failed + 1))
        fi

        if (( count % 10 == 0 )); then
            echo "  $count processed  (ok=$succeeded, fail=$failed)"
        fi
    done < <(jq -r '.[] | [(.video_id // .YouTube_ID // .id), (.start // .start_time // ""), (.end // .end_time // "")] | @tsv' "$json")

    echo "  Done: $succeeded ok, $failed failed"
}

for split in $SPLITS; do
    split_dir="$OUTPUT_DIR/$split"
    mkdir -p "$split_dir"

    echo ""
    echo "=== Split: $split ==="

    if [[ -f "$PKG_DIR/$split.csv" ]]; then
        parse_and_download_csv "$PKG_DIR/$split.csv" "$split_dir" "$MAX_PER_SPLIT"
    elif [[ -f "$PKG_DIR/$split.json" ]]; then
        parse_and_download_json "$PKG_DIR/$split.json" "$split_dir" "$MAX_PER_SPLIT"
    else
        echo "  Warning: no data file found for split '$split' in $PKG_DIR"
    fi
done

echo ""
echo "Done. Dataset saved to $OUTPUT_DIR"

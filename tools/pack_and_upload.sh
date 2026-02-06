#!/bin/bash
# Pack and upload workspace components to Google Drive.
# Usage: bash tools/pack_and_upload.sh /NVMe3/wicompass_workspace
#
# Strategy:
#   1. Compress 5 small components in parallel
#   2. Upload each as soon as it finishes
#   3. Compress and upload 2 large components sequentially
#
# Archives are stored alongside the workspace to avoid filling /home.

set -e

WORKSPACE="${1:?Usage: $0 /path/to/wicompass_workspace}"
ARCHIVE_DIR="${WORKSPACE}_archives"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ARCHIVE_DIR"

echo "============================================"
echo "Workspace:   $WORKSPACE"
echo "Archive dir: $ARCHIVE_DIR"
echo "============================================"

compress_and_upload() {
    local name="$1"
    local src="$WORKSPACE/$name"
    local archive="$ARCHIVE_DIR/${name}.tar.gz"

    if [ ! -d "$src" ]; then
        echo "[SKIP] $name: not found"
        return
    fi
    if [ -f "$archive" ]; then
        echo "[SKIP] $name: archive exists"
    else
        echo "[PACK] $name ..."
        tar -czf "$archive" -C "$WORKSPACE" "$name"
        local size=$(du -sh "$archive" | cut -f1)
        echo "[DONE] $name -> $size"
    fi

    echo "[UPLOAD] $name ..."
    python "$REPO_ROOT/tools/sync_gdrive.py" upload --file "$archive"
    echo "[UPLOADED] $name"
}

# --- Phase 1: small components in parallel ---
echo ""
echo "=== Phase 1: Small components (parallel) ==="
SMALL_COMPONENTS="model_zoo mmfi AMASS_preproc real_world simulation_datasets"

for name in $SMALL_COMPONENTS; do
    compress_and_upload "$name" &
done
wait
echo "=== Phase 1 done ==="

# --- Phase 2: large components (sequential, to avoid disk pressure) ---
echo ""
echo "=== Phase 2: Large components (sequential) ==="
for name in wicompass_logs mmbody; do
    compress_and_upload "$name"
done
echo "=== Phase 2 done ==="

echo ""
echo "============================================"
echo "All components uploaded."
echo "Archive dir: $ARCHIVE_DIR"
ls -lh "$ARCHIVE_DIR"
echo ""
echo "Next: share the Google Drive 'WiCompass' folder and update README."
echo "============================================"

#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"

echo "Trajecta release (Linux)"
echo "BUILD_DIR: $BUILD_DIR"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake ..
cmake --build .

cpack -G TGZ

archive=$(ls -t Trajecta-*.tar.gz 2>/dev/null | head -n 1 || true)
if [[ -z "$archive" ]]; then
  echo "ERROR: release archive not found"
  exit 1
fi

sha256sum "$archive" > "$archive.sha256"

echo "Archive: $archive"
cat "$archive.sha256"
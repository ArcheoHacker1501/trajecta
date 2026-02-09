#!/usr/bin/env bash
set -e

echo "GDAL check (Linux)"

if command -v gdalinfo >/dev/null 2>&1; then
  gdalinfo --version
else
  echo "ERROR: gdalinfo not found in PATH"
  exit 1
fi

if command -v gdal-config >/dev/null 2>&1; then
  gdal-config --version
  echo "GDAL prefix: $(gdal-config --prefix)"
else
  echo "WARNING: gdal-config not found in PATH"
fi

echo "GDAL check OK"
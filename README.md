# Trajecta

CPU and GPU-accelerated least-cost path analysis on terrain surfaces.

## Overview

Trajecta provides two complementary workflows for movement modeling:
- **FETE** (From-Everywhere-To-Everywhere): accessibility and path density from many sources.
- **LCPA** (Least-Cost Path Analysis): optimal routes from one origin to multiple destinations.

Both modes use anisotropic cost functions and support cost surface modifiers from polyline shapefiles.

## Platform Support

- **Windows 10/11**: Supported (CPU and GPU)
- **Linux**: Experimental (CPU and GPU with CUDA). Some Windows-specific code paths still need portability updates.
- **macOS**: Not currently supported (CUDA not available and Windows-specific code paths). CPU-only port is planned.

## Requirements

- CMake 3.24+
- C++17 compiler
- GDAL 3.x (external dependency)
- CUDA Toolkit 11.0+ (required for current build due to .cu sources)
- NVIDIA GPU (for GPU mode)

Windows uses **OSGeo4W** for GDAL. Linux uses system packages. macOS is not supported yet.

## Quick Start

1. Follow the installation guide: `INSTALL.md`
2. Build the project with CMake
3. Run the executable from the build output

Optional: run `scripts/check_gdal.ps1` (Windows) or `scripts/check_gdal.sh` (Linux) to validate GDAL.

## Documentation

- `INSTALL.md` - Installation and build instructions
- `USAGE.md` - Detailed usage guide
- `CONTRIBUTING.md` - Contributing guidelines
- `THIRD_PARTY_NOTICES.md` - Third-party licenses
- `RELEASE_CHECKLIST.md` - Steps to prepare a release
- `RELEASE_NOTES_TEMPLATE.md` - Release notes template

## Release (Maintainers)

- Windows: `scripts/release.ps1`
- Linux: `scripts/release.sh`

## Citation

If you use Trajecta in your research, please cite:

```
[Add citation information here]
```

## License

GPL-3.0. See `LICENSE` for details.

## Acknowledgments

- Gunrock GPU graph library
- GDAL for geospatial data I/O
- Tobler walking function (White 2015 variant [The Basics of Least Cost Analysis for Archaeological Applications])

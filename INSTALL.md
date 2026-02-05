# Trajecta Installation Guide

This guide covers prerequisites, build steps, and packaging with NSIS. 

## Prerequisites

### Windows

1. **NVIDIA GPU and driver** (for GPU mode)
2. **CUDA Toolkit 11.0+** (required for current build due to .cu sources)
3. **GDAL via OSGeo4W** (external dependency)
4. **CMake 3.24+**
5. **Visual Studio 2019/2022** (Desktop C++ workload)

#### OSGeo4W (GDAL)
1. Download OSGeo4W from https://trac.osgeo.org/osgeo4w/
2. Choose **Express Install**
3. Select **GDAL** (and required dependencies)
4. Default install path: `C:\OSGeo4W64`

### Linux (Ubuntu/Debian example)

Linux support is currently **experimental**. Some Windows-specific code paths may require portability updates.

1. **NVIDIA GPU + driver** (for GPU mode)
2. **CUDA Toolkit** (required for current build due to .cu sources)
3. **GDAL**
4. **CMake 3.24+**
5. **C++ build tools**

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libgdal-dev gdal-bin
```

### macOS

macOS is **not supported** at the moment due to CUDA unavailability and Windows-specific code paths.
A CPU-only port is planned.

## Getting the Source

```bash
git clone https://github.com/<your-org>/trajecta.git
cd trajecta
```

### Gunrock

Trajecta supports two ways of obtaining Gunrock:

- **Submodule (recommended for reproducibility)**
  ```bash
  git submodule update --init --recursive
  ```

- **FetchContent (automatic download)**
  No manual steps are required. CMake will download Gunrock if it is not present.

You can disable the automatic download:
```bash
cmake .. -DTRAJECTA_FETCH_GUNROCK=OFF
```

## GDAL Check Script (Optional)

Use the helper script to verify GDAL is installed and accessible.

Windows:
```bash
powershell -ExecutionPolicy Bypass -File scripts/check_gdal.ps1
```

Linux:
```bash
chmod +x scripts/check_gdal.sh
./scripts/check_gdal.sh
```

## Build (Windows)

From a Visual Studio Developer Command Prompt:

```bash
mkdir build
cd build
cmake .. -DGDAL_ROOT="C:/OSGeo4W64" -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build . --config Release
```

**CUDA architectures:**
- RTX 30xx (Ampere): `86`
- RTX 40xx (Ada): `89`
- V100 (Volta): `70`
- A100 (Ampere): `80`

### GDAL DLLs (Windows)

Trajecta expects GDAL runtime DLLs to be next to the executable.
CMake can copy them automatically:

```bash
cmake .. -DTRAJECTA_COPY_GDAL_RUNTIME=ON
```

If the DLLs are **not** copied automatically, you must copy them manually:

1. Go to `C:\OSGeo4W64\bin`
2. Copy all DLLs into `build\Release`

This step is required for `trajecta.exe` to start correctly.

## Build (Linux)

```bash
mkdir build
cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build .
```

## Running

Windows:
```bash
cd build\Release
trajecta.exe
```

Linux:
```bash
cd build
./trajecta
```

## Packaging (NSIS Installer - Windows)

NSIS is only required on the machine that builds the installer.
End users do **not** need NSIS.

1. Install NSIS: https://nsis.sourceforge.io/Download
2. Verify:
   ```bash
   makensis /VERSION
   ```
3. Build the installer:
   ```bash
   cpack -G NSIS
   ```

The installer will be created in the build directory.

## Troubleshooting

- **GDAL not found:** check `GDAL_ROOT` and ensure OSGeo4W is installed.
- **Gunrock not found:** use submodule or allow FetchContent download.
- **CUDA architecture mismatch:** set `CMAKE_CUDA_ARCHITECTURES` for your GPU.

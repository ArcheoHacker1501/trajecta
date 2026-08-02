# Trajecta Studio

Trajecta Studio is the graphical interface for **Trajecta**. It exposes every
parameter of the interactive `trajecta.exe` console program through a modern
Qt desktop application, so end users never have to touch PowerShell.

![architecture](../../TrajectaHD.ico)

## How it works

`trajecta.exe` is an interactive console program: it prints one question at a
time and reads the answer from `stdin`. Trajecta Studio does **not**
re-implement the engine — it launches the very same `trajecta.exe` as a child
process and answers its questions automatically:

1. The user fills a validated form (mode, input files, cost modifiers,
   algorithm settings, performance limits, output names).
2. `TrajectaRunner` starts `trajecta.exe` with `QProcess`, accumulates its
   stdout and, every time the stream ends with a `> ` prompt, recognises the
   question from its wording and writes the matching answer to stdin.
   (Because `std::cin` is tied to `std::cout`, the full question text is
   always flushed through the pipe before the engine blocks on input.)
3. Progress bars (`\r\033[K\033[1m 45.3%` …) are parsed into a real progress
   bar; the raw console stream is shown live in an ANSI-aware log view.
4. If the engine repeats a question, its validation rejected our input: the
   run is aborted and the `ERROR:` lines from the transcript are shown.
5. On success the *Output Summary* is extracted and displayed, with a button
   that opens the output folder.

The GUI also locates the GDAL runtime automatically (OSGeo4W in standard
locations, DLLs bundled next to the engine, or a user-selected folder) and
injects `PATH`, `PROJ_LIB` and `GDAL_DATA` into the child process — no manual
environment configuration needed.

## Source layout

| File | Role |
|---|---|
| `main.cpp` | Application bootstrap: Fusion style, dark palette, stylesheet, icon |
| `mainwindow.*` | The whole UI: sidebar, Setup / Run / Guide / About pages, validation, settings persistence, GDAL & engine discovery |
| `trajectarunner.*` | The QProcess driver: prompt/answer state machine, progress parsing, error extraction |
| `consoleview.*` | ANSI-aware live log (SGR colors, `\r` progress rewrites) |
| `pathpicker.*` | File/folder selector with drag & drop and validity indicator |
| `systeminfo.h` | Detected RAM for sensible defaults |
| `theme.qss` | Dark theme (slate / green / orange, matching the Trajecta logo) |
| `runner_probe.cpp` | Headless test tool that drives a full analysis without the GUI |

A complete architecture and code walkthrough (design rationale, the
question/answer state machine, every file explained block by block) is in
[`CODE_EXPLANATION.md`](CODE_EXPLANATION.md).

## Building (development)

Requirements: Qt 6.x (Widgets), CMake 3.19+, a C++17 compiler
(MinGW 13 from the Qt installer works out of the box).

Open `CMakeLists.txt` in **Qt Creator** and press Run, or from a shell:

```powershell
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DCMAKE_PREFIX_PATH=C:/Qt/6.10.2/mingw_64
cmake --build build-release
```

At startup the GUI looks for `trajecta.exe` in its own folder; during
development set the `TRAJECTA_ENGINE` environment variable to your engine
build (e.g. `build\Release\trajecta.exe`) or use *Locate engine…* in the
sidebar.

### Headless state-machine test

```powershell
cmake -S . -B build-release -DTRAJECTA_STUDIO_BUILD_TESTTOOL=ON  # + generator flags
build-release\runner_probe.exe --exe ...\trajecta.exe --mode fete `
    --dem DEM.tif --points Points.shp --out C:\tmp\out --echo
```

## Packaging into the installer

`scripts/release.ps1` (repo root) does everything:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\release.ps1
```

It builds the GUI, runs `windeployqt` into `build\gui_deploy`, then builds
the engine and packages both with CPack/NSIS. The resulting
`build\Trajecta-<version>.exe` installer contains `trajecta.exe`,
`TrajectaStudio.exe` with all Qt DLLs, and creates Start Menu and Desktop
shortcuts for **Trajecta Studio**.

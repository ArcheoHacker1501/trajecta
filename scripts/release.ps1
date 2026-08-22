param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$CudaArch = "86",
    [string]$GdalRoot = "C:\OSGeo4W64",
    # Where the GDAL runtime bundled into the package is taken from. Passed
    # explicitly so a release never depends on whatever is left in the CMake
    # cache. QGIS is used rather than OSGeo4W because its proj.db is current;
    # the OSGeo4W copy is years older than its own DLLs.
    [string]$RuntimeDir = "C:\Program Files\QGIS 3.44.10",
    [string]$QtDir = "C:\Qt\6.10.2\mingw_64",
    [string]$MinGWDir = "C:\Qt\Tools\mingw1310_64",
    [string]$NinjaExe = "C:\Qt\Tools\Ninja\ninja.exe",
    [switch]$SkipBuild,
    [switch]$SkipGui,
    [switch]$SkipPackage
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot

Write-Host "Trajecta release (Windows)"
Write-Host "RepoRoot: $RepoRoot"
Write-Host "BuildDir: $BuildDir"
Write-Host "Config: $Config"
Write-Host "GDAL_ROOT: $GdalRoot"
Write-Host "QtDir: $QtDir"

if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Force $BuildDir | Out-Null
}

# ---------------------------------------------------------------------------
# 1) Build Trajecta Studio (the Qt GUI) and deploy it with windeployqt
# ---------------------------------------------------------------------------
$GuiDeployDir = ""
if (-not $SkipGui) {
    $GuiSource = Join-Path $RepoRoot "gui\TrajectaStudio"
    $GuiBuild = Join-Path $RepoRoot "gui\TrajectaStudio\build-release"
    # Forward slashes: backslashes in -D cache values can be mis-read by CMake.
    $GuiDeployDir = (Join-Path (Resolve-Path $BuildDir).Path "gui_deploy") -replace '\\', '/'

    if (-not (Test-Path "$QtDir\bin\windeployqt.exe")) {
        Write-Host "ERROR: windeployqt not found in $QtDir\bin. Pass -QtDir or use -SkipGui."
        exit 1
    }

    Write-Host "`n=== Building Trajecta Studio (GUI) ==="
    $env:PATH = "$MinGWDir\bin;" + $env:PATH

    $generatorArgs = @()
    if (Test-Path $NinjaExe) {
        $generatorArgs = @("-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=$NinjaExe")
    } else {
        $generatorArgs = @("-G", "MinGW Makefiles")
    }

    cmake -S $GuiSource -B $GuiBuild @generatorArgs `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_PREFIX_PATH="$QtDir" `
        -DCMAKE_CXX_COMPILER="$MinGWDir\bin\g++.exe"
    if ($LASTEXITCODE -ne 0) { exit 1 }
    cmake --build $GuiBuild
    if ($LASTEXITCODE -ne 0) { exit 1 }

    Write-Host "`n=== Deploying Qt runtime (windeployqt) ==="
    if (Test-Path $GuiDeployDir) {
        Remove-Item -Recurse -Force $GuiDeployDir
    }
    New-Item -ItemType Directory -Force $GuiDeployDir | Out-Null
    Copy-Item "$GuiBuild\TrajectaStudio.exe" $GuiDeployDir

    # windeployqt writes a harmless "Cannot find any version of the
    # dxcompiler.dll and dxil.dll" warning to stderr on some Qt installs (the
    # app never needs DirectX shader compilation) — PowerShell's default
    # ErrorActionPreference of "Stop" turns any stderr write from a native
    # command into a terminating error, aborting the script over a warning
    # $LASTEXITCODE below would have ignored anyway. Relaxed to "Continue"
    # for just this call, which is what actually lets the exit-code check do
    # its job.
    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & "$QtDir\bin\windeployqt.exe" --release --compiler-runtime --no-translations `
        --no-system-d3d-compiler --no-opengl-sw `
        "$GuiDeployDir\TrajectaStudio.exe"
    $windeployExit = $LASTEXITCODE
    $ErrorActionPreference = $previousEap
    if ($windeployExit -ne 0) { exit 1 }

    Write-Host "GUI deployed to: $GuiDeployDir"
}

# ---------------------------------------------------------------------------
# 2) Build the Trajecta engine and package everything with CPack/NSIS
# ---------------------------------------------------------------------------
Push-Location $BuildDir
try {
    if (-not $SkipBuild) {
        if (Test-Path "CMakeCache.txt") {
            $cache = Get-Content -Raw "CMakeCache.txt"
            if ($cache -match "CMAKE_CUDA_ARCHITECTURES:.*=\\$CudaArch") {
                Remove-Item -Force "CMakeCache.txt"
                if (Test-Path "CMakeFiles") {
                    Remove-Item -Recurse -Force "CMakeFiles"
                }
            }
        }
        $cmakeArgs = @("-DGDAL_ROOT=$GdalRoot",
                       "-DTRAJECTA_RUNTIME_DIR=$($RuntimeDir -replace '\\','/')")
        if ($GuiDeployDir) {
            $cmakeArgs += "-DTRAJECTA_GUI_DIR=$GuiDeployDir"
        }
        cmake .. @cmakeArgs
        if ($LASTEXITCODE -ne 0) { exit 1 }
        cmake --build . --config $Config
        if ($LASTEXITCODE -ne 0) { exit 1 }
    } elseif ($GuiDeployDir) {
        # Refresh the GUI path in the existing cache even when skipping the build.
        cmake .. -DTRAJECTA_GUI_DIR="$GuiDeployDir"
        if ($LASTEXITCODE -ne 0) { exit 1 }
    }

    if (-not $SkipPackage) {
        cpack -G NSIS
        if ($LASTEXITCODE -ne 0) { exit 1 }
    }

    $installer = Get-ChildItem -File -Filter "Trajecta-*.exe" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $installer) {
        Write-Host "ERROR: Installer not found. Expected Trajecta-*.exe in $BuildDir"
        exit 1
    }

    $hash = Get-FileHash -Algorithm SHA256 $installer.FullName
    $hashLine = "$($hash.Hash)  $($installer.Name)"
    $hashLine | Set-Content -NoNewline "$($installer.FullName).sha256"

    Write-Host "Installer: $($installer.FullName)"
    Write-Host "SHA256: $($hash.Hash)"
} finally {
    Pop-Location
}

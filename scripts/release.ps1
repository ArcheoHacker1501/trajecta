param(
    [string]$BuildDir = "build",
    [string]$Config = "Release",
    [string]$CudaArch = "86",
    [string]$GdalRoot = "C:\OSGeo4W64",
    [switch]$SkipBuild,
    [switch]$SkipPackage
)

$ErrorActionPreference = "Stop"

Write-Host "Trajecta release (Windows)"
Write-Host "BuildDir: $BuildDir"
Write-Host "Config: $Config"
Write-Host "CUDA Arch: $CudaArch"
Write-Host "GDAL_ROOT: $GdalRoot"

if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Force $BuildDir | Out-Null
}

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
        $archArg = "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
        cmake .. -DGDAL_ROOT="$GdalRoot" $archArg
        cmake --build . --config $Config
    }

    if (-not $SkipPackage) {
        cpack -G NSIS
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

param(
    [string]$GdalRoot = $env:GDAL_ROOT
)

if ([string]::IsNullOrWhiteSpace($GdalRoot)) {
    $GdalRoot = "C:\OSGeo4W64"
}

Write-Host "GDAL check (Windows / OSGeo4W)"
Write-Host "GDAL_ROOT: $GdalRoot"

$errors = 0

$gdalInfoCmd = Get-Command gdalinfo -ErrorAction SilentlyContinue
if ($gdalInfoCmd) {
    Write-Host "gdalinfo found in PATH: $($gdalInfoCmd.Source)"
} else {
    $gdalInfoPath = Join-Path $GdalRoot "bin\gdalinfo.exe"
    if (Test-Path $gdalInfoPath) {
        Write-Host "gdalinfo found at: $gdalInfoPath"
    } else {
        Write-Host "ERROR: gdalinfo not found in PATH or $gdalInfoPath"
        $errors++
    }
}

$gdalLib = Join-Path $GdalRoot "lib\gdal_i.lib"
if (Test-Path $gdalLib) {
    Write-Host "GDAL import lib found: $gdalLib"
} else {
    Write-Host "ERROR: GDAL import lib not found: $gdalLib"
    $errors++
}

$binDir = Join-Path $GdalRoot "bin"
if (Test-Path $binDir) {
    $dlls = Get-ChildItem -Path $binDir -Filter "gdal*.dll" -ErrorAction SilentlyContinue
    if ($dlls.Count -gt 0) {
        Write-Host "GDAL runtime DLLs found in $binDir ($($dlls.Count) files)"
    } else {
        Write-Host "WARNING: No GDAL runtime DLLs found in $binDir"
    }
} else {
    Write-Host "ERROR: GDAL bin directory not found: $binDir"
    $errors++
}

$projDb = Join-Path $GdalRoot "share\proj\proj.db"
if (Test-Path $projDb) {
    Write-Host "PROJ database found: $projDb"
} else {
    Write-Host "WARNING: PROJ database not found: $projDb"
}

if ($errors -gt 0) {
    Write-Host "GDAL check FAILED ($errors error(s))"
    exit 1
}

Write-Host "GDAL check OK"
exit 0
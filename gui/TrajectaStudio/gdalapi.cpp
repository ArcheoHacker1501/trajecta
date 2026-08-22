#include "gdalapi.h"

#include <QByteArray>
#include <QDir>
#include <QFileInfo>
#include <QRegularExpression>

#include <algorithm>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

GdalApi &GdalApi::instance()
{
    static GdalApi api;
    return api;
}

namespace {

#ifdef Q_OS_WIN
HMODULE g_gdalModule = nullptr;

template <typename T>
bool resolve(T &fn, const char *name)
{
    fn = reinterpret_cast<T>(
        reinterpret_cast<void *>(GetProcAddress(g_gdalModule, name)));
    return fn != nullptr;
}
#endif

} // namespace

bool GdalApi::tryLoadFrom(const QString &dir)
{
#ifdef Q_OS_WIN
    // An OSGeo4W root keeps every GDAL it has ever installed side by side
    // (gdal301.dll ... gdal312.dll). Alphabetical order would pick the oldest,
    // which pairs with a PROJ that predates the installed proj.db; sort by the
    // version number in the name and try the newest first.
    QStringList dlls = QDir(dir).entryList({QStringLiteral("gdal*.dll")}, QDir::Files);
    static const QRegularExpression versionRe(QStringLiteral("(\\d+)"));
    auto versionOf = [](const QString &name) {
        const QRegularExpressionMatch m = versionRe.match(name);
        return m.hasMatch() ? m.captured(1).toInt() : -1;
    };
    std::sort(dlls.begin(), dlls.end(), [&](const QString &a, const QString &b) {
        return versionOf(a) > versionOf(b);
    });
    for (const QString &dll : dlls) {
        const QString full = QDir(dir).absoluteFilePath(dll);
        // SetDllDirectory + ALTERED_SEARCH_PATH so gdal's own dependency DLLs
        // (proj, geos, sqlite, ...) resolve from the same OSGeo4W bin folder.
        SetDllDirectoryW(reinterpret_cast<const wchar_t *>(dir.utf16()));
        HMODULE mod = LoadLibraryExW(
            reinterpret_cast<const wchar_t *>(full.utf16()), nullptr,
            LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!mod)
            continue;
        // Not every gdal*.dll is the library (e.g. plugins); check a symbol.
        if (!GetProcAddress(mod, "GDALAllRegister")) {
            FreeLibrary(mod);
            continue;
        }
        // Keep the successful directory as the DLL search path: GDAL may
        // still load driver dependencies lazily from the same folder.
        g_gdalModule = mod;
        return true;
    }
    // Nothing usable here: restore the default search path instead of
    // leaving the process pointed at a folder that failed to deliver.
    SetDllDirectoryW(nullptr);
#else
    Q_UNUSED(dir);
#endif
    return false;
}

bool GdalApi::load(const QStringList &candidateDirs,
                   const QString &projDataDir, const QString &gdalDataDir)
{
    if (m_loaded)
        return true;

#ifdef Q_OS_WIN
    for (const QString &dir : candidateDirs) {
        if (!dir.isEmpty() && QDir(dir).exists() && tryLoadFrom(dir))
            break;
    }
    if (!g_gdalModule) {
        m_error = QStringLiteral("gdal*.dll not found in any candidate folder");
        return false;
    }

    const bool ok =
        resolve(AllRegister, "GDALAllRegister")
        && resolve(OpenEx, "GDALOpenEx")
        && resolve(Close, "GDALClose")
        && resolve(GetRasterXSize, "GDALGetRasterXSize")
        && resolve(GetRasterYSize, "GDALGetRasterYSize")
        && resolve(GetRasterBand, "GDALGetRasterBand")
        && resolve(GetGeoTransform, "GDALGetGeoTransform")
        && resolve(GetProjectionRef, "GDALGetProjectionRef")
        && resolve(RasterIO, "GDALRasterIO")
        && resolve(GetRasterNoDataValue, "GDALGetRasterNoDataValue")
        && resolve(OSRNewSpatialReference, "OSRNewSpatialReference")
        && resolve(OSRDestroySpatialReference, "OSRDestroySpatialReference")
        && resolve(OSRGetName, "OSRGetName")
        && resolve(OSRGetAuthorityName, "OSRGetAuthorityName")
        && resolve(OSRGetAuthorityCode, "OSRGetAuthorityCode")
        && resolve(OSRIsGeographic, "OSRIsGeographic")
        && resolve(OSRSetPROJSearchPaths, "OSRSetPROJSearchPaths")
        && resolve(CPLSetConfigOption, "CPLSetConfigOption")
        && resolve(OSRImportFromEPSG, "OSRImportFromEPSG")
        && resolve(OSRSetAxisMappingStrategy, "OSRSetAxisMappingStrategy")
        && resolve(OCTNewCoordinateTransformation, "OCTNewCoordinateTransformation")
        && resolve(OCTDestroyCoordinateTransformation, "OCTDestroyCoordinateTransformation")
        && resolve(OCTTransform, "OCTTransform")
        && resolve(DatasetGetLayerCount, "GDALDatasetGetLayerCount")
        && resolve(DatasetGetLayer, "GDALDatasetGetLayer")
        && resolve(L_ResetReading, "OGR_L_ResetReading")
        && resolve(L_GetNextFeature, "OGR_L_GetNextFeature")
        && resolve(F_GetGeometryRef, "OGR_F_GetGeometryRef")
        && resolve(F_Destroy, "OGR_F_Destroy")
        && resolve(G_GetGeometryType, "OGR_G_GetGeometryType")
        && resolve(G_GetGeometryCount, "OGR_G_GetGeometryCount")
        && resolve(G_GetGeometryRef, "OGR_G_GetGeometryRef")
        && resolve(G_GetPointCount, "OGR_G_GetPointCount")
        && resolve(G_GetPoint, "OGR_G_GetPoint");

    // Optional, and deliberately outside the chain above: these three refuse a
    // route comparison whose layers are in degrees, and reproject an imported
    // vector onto the raster beneath it. A GDAL build that somehow lacked them
    // should still load and run everything else — a vector then draws in its
    // own coordinates, which is right whenever it already matches the raster.
    resolve(L_GetSpatialRef, "OGR_L_GetSpatialRef");
    resolve(OSRExportToWkt, "OSRExportToWkt");
    resolve(VSIFree, "VSIFree");

    // The attributes of a feature, and the whole of the writing side. Optional
    // for the same reason and in the same way: the site-coherence tool asks
    // canWriteVector() / canWriteRaster() before it promises an output, and
    // every other part of the application is unaffected by their absence.
    resolve(F_GetFieldCount, "OGR_F_GetFieldCount");
    resolve(F_GetFieldDefnRef, "OGR_F_GetFieldDefnRef");
    resolve(Fld_GetNameRef, "OGR_Fld_GetNameRef");
    resolve(F_GetFieldAsString, "OGR_F_GetFieldAsString");

    resolve(GetDriverByName, "GDALGetDriverByName");
    resolve(Create, "GDALCreate");
    resolve(SetGeoTransform, "GDALSetGeoTransform");
    resolve(SetProjection, "GDALSetProjection");
    resolve(SetRasterNoDataValue, "GDALSetRasterNoDataValue");
    resolve(DatasetCreateLayer, "GDALDatasetCreateLayer");
    resolve(Fld_Create, "OGR_Fld_Create");
    resolve(Fld_Destroy, "OGR_Fld_Destroy");
    resolve(Fld_SetWidth, "OGR_Fld_SetWidth");
    resolve(Fld_SetPrecision, "OGR_Fld_SetPrecision");
    resolve(L_CreateField, "OGR_L_CreateField");
    resolve(L_GetLayerDefn, "OGR_L_GetLayerDefn");
    resolve(F_Create, "OGR_F_Create");
    resolve(F_SetFieldDouble, "OGR_F_SetFieldDouble");
    resolve(F_SetFieldInteger, "OGR_F_SetFieldInteger");
    resolve(F_SetFieldString, "OGR_F_SetFieldString");
    resolve(F_SetGeometryDirectly, "OGR_F_SetGeometryDirectly");
    resolve(L_CreateFeature, "OGR_L_CreateFeature");
    resolve(G_CreateGeometry, "OGR_G_CreateGeometry");
    resolve(G_SetPoint_2D, "OGR_G_SetPoint_2D");
    resolve(G_DestroyGeometry, "OGR_G_DestroyGeometry");
    resolve(OSRSetFromUserInput, "OSRSetFromUserInput");
    if (!ok) {
        m_error = QStringLiteral("a required GDAL C API symbol is missing");
        FreeLibrary(g_gdalModule);
        g_gdalModule = nullptr;
        SetDllDirectoryW(nullptr);
        return false;
    }

    if (!projDataDir.isEmpty()) {
        const QByteArray projUtf8 = QDir::toNativeSeparators(projDataDir).toUtf8();
        const char *paths[] = {projUtf8.constData(), nullptr};
        OSRSetPROJSearchPaths(paths);
    }
    if (!gdalDataDir.isEmpty()) {
        const QByteArray dataUtf8 = QDir::toNativeSeparators(gdalDataDir).toUtf8();
        CPLSetConfigOption("GDAL_DATA", dataUtf8.constData());
    }
    AllRegister();
    m_loaded = true;
    return true;
#else
    Q_UNUSED(candidateDirs);
    Q_UNUSED(projDataDir);
    Q_UNUSED(gdalDataDir);
    m_error = QStringLiteral("runtime GDAL loading is only implemented on Windows");
    return false;
#endif
}

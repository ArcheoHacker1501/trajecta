#pragma once

#include <QString>
#include <QStringList>

// Minimal dynamic binding of the GDAL C API, loaded at runtime from the same
// OSGeo4W installation the engine uses. The GUI is built with MinGW while
// OSGeo4W ships MSVC binaries; the plain-C ABI is the one interface that is
// safe to share between the two, so only C entry points are bound here.
//
// Opaque GDAL handles.
using GDALDatasetH = void *;
using GDALRasterBandH = void *;
using OGRSpatialReferenceH = void *;
using OGRCoordinateTransformationH = void *;
using OGRLayerH = void *;
using OGRFeatureH = void *;
using OGRGeometryH = void *;

class GdalApi
{
public:
    // GDAL enum values used through the C API (stable since GDAL 2.x).
    static constexpr unsigned int OF_Raster = 0x02;
    static constexpr unsigned int OF_Vector = 0x04;
    static constexpr int ReadFlag = 0;      // GF_Read
    static constexpr int Float32 = 6;       // GDT_Float32
    static constexpr int Float64 = 7;       // GDT_Float64
    static constexpr int WkbPoint = 1;
    static constexpr int WkbLineString = 2;
    static constexpr int WkbMultiPoint = 4;
    static constexpr int WkbMultiLineString = 5;

    static GdalApi &instance();

    // Idempotent. Tries each candidate directory until a gdal*.dll loads and
    // every needed symbol resolves. projData/gdalData configure PROJ and GDAL
    // support-file lookup (empty = leave defaults).
    bool load(const QStringList &candidateDirs,
              const QString &projDataDir, const QString &gdalDataDir);
    bool isLoaded() const { return m_loaded; }
    QString loadError() const { return m_error; }

    // --- Bound C entry points (valid only when isLoaded()) ---
    void (*AllRegister)() = nullptr;
    GDALDatasetH (*OpenEx)(const char *, unsigned int, const char *const *,
                           const char *const *, const char *const *) = nullptr;
    void (*Close)(GDALDatasetH) = nullptr;
    int (*GetRasterXSize)(GDALDatasetH) = nullptr;
    int (*GetRasterYSize)(GDALDatasetH) = nullptr;
    GDALRasterBandH (*GetRasterBand)(GDALDatasetH, int) = nullptr;
    int (*GetGeoTransform)(GDALDatasetH, double *) = nullptr;
    const char *(*GetProjectionRef)(GDALDatasetH) = nullptr;
    int (*RasterIO)(GDALRasterBandH, int, int, int, int, int,
                    void *, int, int, int, int, int) = nullptr;
    double (*GetRasterNoDataValue)(GDALRasterBandH, int *) = nullptr;

    OGRSpatialReferenceH (*OSRNewSpatialReference)(const char *) = nullptr;
    void (*OSRDestroySpatialReference)(OGRSpatialReferenceH) = nullptr;
    const char *(*OSRGetName)(OGRSpatialReferenceH) = nullptr;
    const char *(*OSRGetAuthorityName)(OGRSpatialReferenceH, const char *) = nullptr;
    const char *(*OSRGetAuthorityCode)(OGRSpatialReferenceH, const char *) = nullptr;
    int (*OSRIsGeographic)(OGRSpatialReferenceH) = nullptr;
    void (*OSRSetPROJSearchPaths)(const char *const *) = nullptr;
    void (*CPLSetConfigOption)(const char *, const char *) = nullptr;

    // Coordinate transformation (satellite basemap reprojection).
    int (*OSRImportFromEPSG)(OGRSpatialReferenceH, int) = nullptr;
    void (*OSRSetAxisMappingStrategy)(OGRSpatialReferenceH, int) = nullptr;
    OGRCoordinateTransformationH (*OCTNewCoordinateTransformation)(
        OGRSpatialReferenceH, OGRSpatialReferenceH) = nullptr;
    void (*OCTDestroyCoordinateTransformation)(OGRCoordinateTransformationH) = nullptr;
    int (*OCTTransform)(OGRCoordinateTransformationH, int,
                        double *, double *, double *) = nullptr;

    int (*DatasetGetLayerCount)(GDALDatasetH) = nullptr;
    OGRLayerH (*DatasetGetLayer)(GDALDatasetH, int) = nullptr;
    void (*L_ResetReading)(OGRLayerH) = nullptr;
    OGRFeatureH (*L_GetNextFeature)(OGRLayerH) = nullptr;
    OGRGeometryH (*F_GetGeometryRef)(OGRFeatureH) = nullptr;
    void (*F_Destroy)(OGRFeatureH) = nullptr;
    int (*G_GetGeometryType)(OGRGeometryH) = nullptr;
    int (*G_GetGeometryCount)(OGRGeometryH) = nullptr;
    OGRGeometryH (*G_GetGeometryRef)(OGRGeometryH, int) = nullptr;
    int (*G_GetPointCount)(OGRGeometryH) = nullptr;
    void (*G_GetPoint)(OGRGeometryH, int, double *, double *, double *) = nullptr;

    // wkbLineString25D/Z/M etc. all flatten to the base 2D type.
    static int flattenGeomType(int t)
    {
        t &= 0x7fffffff;          // strip the legacy 2.5D flag
        if (t >= 1000)
            t %= 1000;            // strip Z/M/ZM offsets
        return t;
    }

private:
    GdalApi() = default;
    bool tryLoadFrom(const QString &dir);

    bool m_loaded = false;
    QString m_error;
};

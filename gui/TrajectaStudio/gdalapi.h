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
using GDALDriverH = void *;
using OGRFieldDefnH = void *;
using OGRFeatureDefnH = void *;

class GdalApi
{
public:
    // GDAL enum values used through the C API (stable since GDAL 2.x).
    static constexpr unsigned int OF_Raster = 0x02;
    static constexpr unsigned int OF_Vector = 0x04;
    static constexpr int ReadFlag = 0;      // GF_Read
    static constexpr int WriteFlag = 1;     // GF_Write
    static constexpr int Float32 = 6;       // GDT_Float32
    static constexpr int Float64 = 7;       // GDT_Float64
    // OGR field types, for the layers this application writes.
    static constexpr int OftInteger = 0;
    static constexpr int OftReal = 2;
    static constexpr int OftString = 4;
    static constexpr int WkbPoint = 1;
    static constexpr int WkbLineString = 2;
    static constexpr int WkbPolygon = 3;
    static constexpr int WkbMultiPoint = 4;
    static constexpr int WkbMultiLineString = 5;
    static constexpr int WkbMultiPolygon = 6;
    static constexpr int WkbGeometryCollection = 7;
    // A polygon's parts are rings; a ring is a closed line string.
    static constexpr int WkbLinearRing = 101;

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
    // Optional: only used to refuse a comparison in degrees, so a GDAL
    // build without it still loads and everything else keeps working.
    OGRSpatialReferenceH (*L_GetSpatialRef)(OGRLayerH) = nullptr;
    // Also optional, and used together with the one above: an imported vector
    // has to be reprojected onto the raster under it, and the layer's spatial
    // reference belongs to a dataset that is closed as soon as it is read, so
    // the definition is copied out as text while it is still valid.
    int (*OSRExportToWkt)(OGRSpatialReferenceH, char **) = nullptr;
    void (*VSIFree)(void *) = nullptr;
    OGRFeatureH (*L_GetNextFeature)(OGRLayerH) = nullptr;
    OGRGeometryH (*F_GetGeometryRef)(OGRFeatureH) = nullptr;
    void (*F_Destroy)(OGRFeatureH) = nullptr;
    int (*G_GetGeometryType)(OGRGeometryH) = nullptr;
    int (*G_GetGeometryCount)(OGRGeometryH) = nullptr;
    OGRGeometryH (*G_GetGeometryRef)(OGRGeometryH, int) = nullptr;
    int (*G_GetPointCount)(OGRGeometryH) = nullptr;
    void (*G_GetPoint)(OGRGeometryH, int, double *, double *, double *) = nullptr;

    // --- Attributes of a read feature ---
    //
    // Optional, like everything below: the site-coherence tool copies the
    // input layer's own fields into its output so a result can be read without
    // going back to the source, and does without them if this GDAL cannot say
    // what they are.
    int (*F_GetFieldCount)(OGRFeatureH) = nullptr;
    OGRFieldDefnH (*F_GetFieldDefnRef)(OGRFeatureH, int) = nullptr;
    const char *(*Fld_GetNameRef)(OGRFieldDefnH) = nullptr;
    const char *(*F_GetFieldAsString)(OGRFeatureH, int) = nullptr;

    // --- Writing ---
    //
    // Also optional, and for one reason: a GDAL build that is missing any of
    // these must still load, so that everything the application already does
    // keeps working. What depends on them checks canWriteVector() /
    // canWriteRaster() first and says plainly what cannot be produced.
    GDALDriverH (*GetDriverByName)(const char *) = nullptr;
    GDALDatasetH (*Create)(GDALDriverH, const char *, int, int, int, int,
                           const char *const *) = nullptr;
    int (*SetGeoTransform)(GDALDatasetH, double *) = nullptr;
    int (*SetProjection)(GDALDatasetH, const char *) = nullptr;
    int (*SetRasterNoDataValue)(GDALRasterBandH, double) = nullptr;

    OGRLayerH (*DatasetCreateLayer)(GDALDatasetH, const char *, OGRSpatialReferenceH,
                                    int, const char *const *) = nullptr;
    OGRFieldDefnH (*Fld_Create)(const char *, int) = nullptr;
    void (*Fld_Destroy)(OGRFieldDefnH) = nullptr;
    void (*Fld_SetWidth)(OGRFieldDefnH, int) = nullptr;
    void (*Fld_SetPrecision)(OGRFieldDefnH, int) = nullptr;
    int (*L_CreateField)(OGRLayerH, OGRFieldDefnH, int) = nullptr;
    OGRFeatureDefnH (*L_GetLayerDefn)(OGRLayerH) = nullptr;
    OGRFeatureH (*F_Create)(OGRFeatureDefnH) = nullptr;
    void (*F_SetFieldDouble)(OGRFeatureH, int, double) = nullptr;
    void (*F_SetFieldInteger)(OGRFeatureH, int, int) = nullptr;
    void (*F_SetFieldString)(OGRFeatureH, int, const char *) = nullptr;
    int (*F_SetGeometryDirectly)(OGRFeatureH, OGRGeometryH) = nullptr;
    int (*L_CreateFeature)(OGRLayerH, OGRFeatureH) = nullptr;
    OGRGeometryH (*G_CreateGeometry)(int) = nullptr;
    void (*G_SetPoint_2D)(OGRGeometryH, int, double, double) = nullptr;
    void (*G_DestroyGeometry)(OGRGeometryH) = nullptr;
    int (*OSRSetFromUserInput)(OGRSpatialReferenceH, const char *) = nullptr;

    bool canReadFields() const
    {
        return F_GetFieldCount && F_GetFieldDefnRef && Fld_GetNameRef
               && F_GetFieldAsString;
    }
    bool canWriteRaster() const
    {
        return GetDriverByName && Create && SetGeoTransform && SetProjection
               && SetRasterNoDataValue && RasterIO && Close;
    }
    bool canWriteVector() const
    {
        return GetDriverByName && Create && DatasetCreateLayer && Fld_Create
               && Fld_Destroy && L_CreateField && L_GetLayerDefn && F_Create
               && F_SetFieldDouble && F_SetFieldInteger && F_SetFieldString
               && F_SetGeometryDirectly && L_CreateFeature && G_CreateGeometry
               && G_SetPoint_2D && F_Destroy && Close;
    }

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

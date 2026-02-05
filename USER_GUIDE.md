# Trajecta Usage Guide

This guide explains how to use Trajecta in FETE and LCPA modes.

## Quick Start

Run the program using the **x64 native tools command prompt for vs 2022**:

```bash
# Windows
cd build/Release
trajecta.exe

# Linux
cd build
./trajecta
```

On start, choose the analysis mode:
```
Select mode:
1. FETE (From-Everywhere-To-Everywhere)
2. LCPA (Least Cost Path Analysis)
```

If you want, you can enable verbose output when prompted. This will print out detailed information about completed processes allowing for advanced debug.

## Input Data Requirements

### Digital Elevation Model (DEM)
- Format: GeoTIFF or any GDAL readable raster
- Projection: projected CRS (UTM recommended)
- Units: meters

### Points

**FETE mode (sample points):**
- Format: Shapefile (.shp)
- Geometry: Point

**LCPA mode (origin and destinations):**
- Origin: single Point shapefile
- Destinations: multiple Point features

### Optional Cost Modifiers
You can modify the cost surface with a polyline shapefile:
- Geometry: LineString or MultiLineString
- Required field: `cost` (float multiplier)
  - < 1.0 makes travel faster (roads)
  - = 1.0 no change
  - > 1.0 makes travel slower (barriers)

## FETE Mode

FETE computes accessibility and path density from many sources.

### Workflow
1. Provide DEM path
2. Provide sample points shapefile
3. Choose output directory
4. Optionally add cost modifiers
5. Choose parameters

### Key Parameters
- **Connectivity:** 8, 16, 24, 32, or 64 neighbors
- **Slope units:** degrees or percent
- **Buffer radius:** smoothing around paths
- **Max RAM:** memory limit for processing

### Output Files
- `slope_*.tif` - slope raster
- `cost_surface_*.tif` - base cost surface
- `density_*.tif` - path density raster
- If modifiers are used:
  - `additional_cost_*.tif`
  - `total_cost_*.tif`

## LCPA Mode

LCPA computes least-cost paths from one origin to multiple destinations.

### Workflow
1. Provide DEM path
2. Provide origin shapefile (single point)
3. Provide destinations shapefile (one or more points)
4. Choose output directory
5. Optionally add cost modifiers

### Output Files
- `slope_*.tif`
- `cost_surface_*.tif`
- `path_raster_*.tif`
- `path_lines_*.shp`

## Cost Modifiers Details

When a polyline shapefile is provided, Trajecta rasterizes the features and
applies the `cost` multiplier to the base cost surface. Buffer radius
controls how many cells around the line are affected.

Example multipliers:
- Road: `0.4`
- River crossing: `2.0`
- Wall/barrier: `10.0`

## Tips

- Ensure all inputs share the same CRS.
- Start with a small DEM to validate the workflow.
- Use verbose mode on first runs to confirm parameters.
- Keep buffer radius small unless you need strong smoothing.

## Troubleshooting

- **Points outside DEM:** reproject or verify bounds
- **No paths found:** check inputs and NoData areas
- **GDAL errors on Windows:** confirm DLLs are in `build/Release`

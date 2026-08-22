# Overview


> [!IMPORTANT]
> - To install the latest version of Trajecta (v1.0.1) [download the installer](https://github.com/ArcheoHacker1501/trajecta/releases/tag/v1.0.1) and click on it once the download is done. Then follow the instructions.


Trajecta is a least-cost analysis software specifically developed for users with only a basic computer science background. **Be patient, this software is currently under development and can contain bugs or errors**. Please, contact me for bug reporting, problems during the installation, improvements or additional features you would like to see developed and included in future releases.

## Core Functions of Trajecta

Currently, Trajecta provides two complementary workflows for movement modeling (FETE and LCPA, see below). Both modes use anisotropic cost functions (e.g. Modified Tobler's Hiking Function) and support cost surface modifiers (e.g. waterbodies).

### FETE — From Everywhere To Everywhere

As analysis model, FETE was originally conceptualized by White & Barber (2012). The FETE algorithm implemented by Trajecta allows to calculate a high number of least-cost paths connecting every point to every other point of a regular or randomly scattered point grid. To compute LCPs, DEM or other elevation based data can be used to calculate slope which can then be transformed using cost functions (e.g. Tobler's Hiking Function). Additional costs can be added as for waterbodies or terrain indexes.

Based on all the LCPs calculated, overlap analysis is then computed and a density raster is created, with the same resolution as the input DEM. Different color gradients can be used to display most probable paths among all calculated LCPs.

|![](gui/TrajectaStudio/assets/guide/Grid_FETE.jpg)|![](gui/TrajectaStudio/assets/guide/unfiltered_FETE.jpg)|
|:-:|:-:|
|*Example of regular point grid and SRTM 30m DEM used as input for FETE computation.*|*Unfiltered FETE raster resulting from computation with Trajecta.*|

|![](gui/TrajectaStudio/assets/guide/filtered_FETE.jpg)|
|:-:|
|*Filtered FETE raster using only top 20% results.*|

### LCPA — Least-Cost Path Analysis

For an introduction to Least-Cost Path Analysis (LCPA), see White (2015). Trajecta allows high-speed computation of Least-Cost Paths (LCPs) between a single origin and one or more destinations. To compute LCPs, DEM or other elevation based data can be used to calculate slope which can then be transformed using cost functions (e.g. Modified Tobler's Hiking Function). Additional costs can be added as for waterbodies or terrain indexes.

|![](gui/TrajectaStudio/assets/guide/LCPA.jpg)|
|:-:|
|*Least-Cost Paths from single origin to multiple destinations calculated using Trajecta and SRTM 30m DEM.*|

## Post-processing: NNI — Natural Neighbour Interpolation

The **Post-processing** page turns a FETE density raster into a smooth, continuous surface using **discrete Sibson (natural neighbour) interpolation (Park et al. 2006)**. Cells at or above the **sample threshold** act as sample points; every other cell receives the average of the samples whose influence area it would claim. Sample values are preserved exactly. The optional **max search radius** caps how far the interpolation reaches into empty areas (beyond it, cells take the value of their nearest sample), which keeps large rasters fast. After a successful FETE run the density raster is filled in automatically, allowing for direct post-processing.

|![](gui/TrajectaStudio/assets/guide/FETE_density.jpg)|![](gui/TrajectaStudio/assets/guide/FETE_density_NNI.jpg)|
|:-:|:-:|
|*FETE density raster generated with Trajecta.*|*The same FETE density raster after NNI.*|

## Sample points: imported or generated

In FETE mode the sample points can either be **imported from a file** (e.g. .shp, .geojson, .csv), or **directly generated from the DEM**.

Generation takes two parameters. The **density** is expressed either as a **point spacing** (one point every N rows and every N columns, so the count falls with the square of N) or as a **target number of points**, from which the spacing is derived using the number of usable DEM cells. The **arrangement** is either a **regular grid**, which puts every point at the same offset inside its block, or **stratified random**, which picks one random cell per block: same density, none of the regularity a grid imposes on the result. A stratified random layer is reproducible from its **seed**.

Points are only placed on cells a path can actually cross, so NoData areas stay empty. The layer is written to the output folder as a shapefile *before* the analysis starts and is then read back as its input: what the run consumed is always on disk, and it appears in the **Viewer** as a selectable overlay. The setup page shows the resulting point count while you type, with a warning above roughly 50,000 points — FETE cost grows with the square of the number of points.

## Batch processing

A batch runs many analyses one after another, unattended. It is organised in two
levels: a **row** is a single engine run — the smallest thing that can succeed or
fail on its own — and a **chunk** groups the rows that share an algorithm and a
set of cost modifiers, which is exactly the pair of settings that is changed
rarely and would otherwise have to be repeated on every row. The mode, the
hardware limits and the output folder are fixed for the whole batch.

Rows can be typed in, duplicated, or created in bulk from a folder of DEMs. A
batch is saved to and loaded from a `.trjbatch` file, so a set of runs can be
kept or shared. Each row reports its own outcome, and a row that fails does not
stop the ones after it.

## Auto-save and resume

Long analyses are checkpointed: at a chosen interval the engine writes the state
of the propagation to disk, and the interface writes alongside it what it would
need to start the run again. Auto-save is on by default, every thirty minutes.

If Trajecta is closed, interrupted, or stopped by a power cut, the next start
offers to resume the analysis from its last saved point rather than from the
beginning; declining deletes the saved state, and there is an option to keep a
copy of it elsewhere first. A batch resumes at the row it stopped on, with the
rows already finished left alone. On an analysis that runs for days, this is the
difference between losing an afternoon and losing the week.

## Comparing a computed route against a known one

The **known-route comparison** measures a computed least-cost path against a
route that is actually attested — a Roman road, a drover's track, a surveyed
path. It is what turns a modelled route from an illustration into a claim that
can be shown to be wrong.

The distances between the two lines are reported as a distribution — median,
90th percentile and maximum — rather than as a single average, because a route
may follow the real one closely for nine kilometres and then take the wrong side
of a hill for one, and a mean hides precisely that. Both directions are
measured, since a short computed path lying on top of a long known one is close
in one direction and far in the other; their maximum is the Hausdorff distance,
the worst disagreement anywhere. Finally, the share of each line running within
a chosen tolerance of the other answers the question most often actually asked:
how much of it did the model get right.

Both layers must be projected and in the same coordinate system. Geographic
coordinates are refused rather than silently reported in meaningless units.

## Input requirements

Trajecta allows for different types of inputs and file formats:

| Input | Requirements |
|---|---|
| **DEM** | GeoTIFF (.tif/.tiff), georeferenced, with a CRS. |
| **Points** | .shp, .geojson/.json, .kml, .gml/.xml or .csv (coordinate columns named x/y, lon/lat or easting/northing). In FETE mode they can be generated from the DEM instead. |
| **Vector modifiers** | Polylines with a float **cost** field holding the multiplier; for .csv the geometry must be in a WKT column. |
| **Raster modifiers** | GeoTIFF with the same dimensions as the DEM; cell values are multipliers (1.0 = unchanged, 2.0 = double cost). |

## Cost modifiers & barriers

Modifiers let you make specific features (water bodies, restricted areas, specific types of terrain) more expensive to cross. The **polyline buffer** widens rasterized lines so paths cannot slip diagonally across them. The **barrier threshold** turns extreme multipliers (e.g. 999999) into hard barriers: cells at or above the threshold are excluded from movement, which also keeps the computation fast.

## Algorithm parameters

**Neighbours** — connectivity of the search grid (8, 16, 24, 32, 64). Higher values allow finer path angles at the price of speed. A connectivity radius of 16 (Knigth's Move) is the usual choice.

**Cost function** — the anisotropic hiking model applied to slope. Currently, the following cost function have been implemented in Trajecta:
- Modified Tobler hiking function (White 2015);
- Márquez-Pérez et al. (2017);
- Irmischer & Clarke (2017).

**Path smoothing buffer** — buffer in cells applied around each computed path when accumulating results. This function allows to calculate wider paths, thus maximising the possibility that actual, historical path(s) were located inside the modeled high-traversability areas.

## Outputs

**Both modes:** slope raster, base cost surface, and (with modifiers) the additional and total cost surfaces.  
**FETE:** the path-density raster, plus the sample points shapefile when the points were generated from the DEM.  
**LCPA:** the paths raster and the paths polyline shapefile.

## Finding your way around

Trajecta Studio carries a **guided tour** of its own interface: it walks through
the pages in order, lighting up one control at a time and saying what it does
and why it is there. It changes nothing that has been set up, and it can be
started again at any point from the **tutorial** link at the top of the Guide
page — which is the first thing to try on a new installation.

The **Viewer** tab displays the results as they are produced: rasters with a
choice of colour ramps, an optional hillshade, vector overlays, and image
export. It is meant for checking work as it goes, not for producing final maps.

## GDAL

The Trajecta engine relies on the **GDAL** geospatial libraries. Since v1.0.0 they are installed together with Trajecta and sit next to the engine: there is nothing to install separately, and no PATH to configure. The status at the bottom of the sidebar should read **GDAL ready** from the first launch.

Trajecta looks beside its own engine before it looks anywhere else, which is what makes this dependable — an installed copy always uses the libraries it shipped with, and cannot be disturbed by any other GDAL on the machine, a QGIS or OSGeo4W install included. If the status is not green the installation is incomplete, or the program was moved by hand rather than installed; reinstalling is the proper remedy, and **Locate GDAL folder** in the sidebar is the stopgap.

## Currently supported Platforms

- **Windows 10/11**: Supported (CPU only)
- **Linux**: Experimental (CPU only). Some Windows-specific code paths still need portability updates.

## Citation

If you use Trajecta in your research, please cite:

```
Trajecta, developed by Stefano Aprà, Ph.D. Candidate - Institute for the Study of the Ancient World (NYU)
```

## License

GPL-3.0. See `LICENSE` for details.

## Acknowledgments

- GDAL for managing geospatial data I/O
- Qt6 for the Trajecta Studio graphical interface

## References

Irmischer, I. J., & Clarke, K. C. (2017). Measuring and modeling the speed of human navigation. *Cartography and Geographic Information Science*, 45(2), 177–186. [doi:10.1080/15230406.2017.1292150](https://doi.org/10.1080/15230406.2017.1292150)

Márquez-Pérez, J., Vallejo-Villalta, I., & Álvarez-Francoso, J. I. (2017). Estimated travel time for walking trails in natural areas. *Geografisk Tidsskrift–Danish Journal of Geography*, 117(1), 53–62. [doi:10.1080/00167223.2017.1316212](https://doi.org/10.1080/00167223.2017.1316212)

Park, S. W., Linsen, L., Kreylos, O., Owens, J. D., & Hamann, B. (2006). Discrete Sibson interpolation. *IEEE Transactions on Visualization and Computer Graphics*, 12(2), 243–253. [doi:10.1109/TVCG.2006.27](https://doi.org/10.1109/TVCG.2006.27)

White, D. A. (2015). The Basics of Least Cost Analysis for Archaeological Applications. *Advances in Archaeological Practice*, 3(4), 407–414. [doi:10.7183/2326-3768.3.4.407](https://doi.org/10.7183/2326-3768.3.4.407)

White, D. A., & Barber, S. B. (2012). Geospatial modeling of pedestrian transportation networks: A case study from precolumbian Oaxaca, Mexico. *Journal of Archaeological Science*, 39(8), 2684–2696. [doi:10.1016/j.jas.2012.04.017](https://doi.org/10.1016/j.jas.2012.04.017)
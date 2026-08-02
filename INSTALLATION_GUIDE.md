# TRAJECTA — Installation Guide

Installing Trajecta takes one step: run the installer. Everything the program
needs is inside it, including the geospatial libraries.

> **Changed in version 1.0.0.** Earlier releases required you to install GDAL
> separately through OSGeo4W and to add it to the system PATH. **That is no
> longer necessary.** The package now carries its own copy of GDAL and of the
> Microsoft C++ runtime, so Trajecta no longer depends on what else is
> installed on the computer. If you followed the old guide, nothing breaks —
> the bundled copy is used regardless.

---

## Step 1 — Download and install

1. Go to the [Releases page](https://github.com/ArcheoHacker1501/trajecta/releases).
2. Download the installer, named `Trajecta-1.0.0.exe`.
3. Double-click it and follow the on-screen instructions.
4. When it finishes, launch **Trajecta Studio** from the Start Menu or the
   desktop shortcut.

That is the whole installation.

### System requirements

| | |
|---|---|
| Operating system | Windows 10 or Windows 11, 64-bit |
| Disk space | about 350 MB |
| Memory | 8 GB minimum; 16 GB or more for large DEMs |
| Administrator rights | needed **only** to install, not to use the program |

Nothing else has to be installed first: no GDAL, no Python, no Visual C++
redistributable.

### Windows SmartScreen

The installer is not code-signed, so on first run Windows may show a blue
"Windows protected your PC" panel. Click **More info**, then **Run anyway**.
This is the normal warning for software from an individual developer rather
than an indication of a problem.

---

## Step 2 — Check that everything was found

Open Trajecta Studio and look at the status bar at the bottom of the window.
Two indicators should be green:

- **✓ Engine ready** — the computation program was found.
- **✓ GDAL ready** — the geospatial libraries were found.

Both refer to files inside the installation folder, so they should be green
immediately after a normal installation. If either is red, use the
**Locate engine…** or **Locate GDAL folder…** buttons next to them.

---

## Do I still need QGIS?

**Not for running Trajecta.** It is, however, still strongly recommended as a
companion, because Trajecta computes results but does not prepare or map data:

- to prepare inputs (clip a DEM, digitise sample points, reproject a layer);
- to make presentation maps from the rasters Trajecta produces.

Trajecta's own **Viewer** tab is enough to check results as you work — display
a raster, apply a hillshade, overlay the sample points, export an image — but
it is not a substitute for a GIS when preparing data or producing figures.

QGIS is free: <https://qgis.org>.

> **A note if you already have QGIS or OSGeo4W.** Trajecta always uses its own
> bundled libraries and ignores whatever GDAL is installed elsewhere. This is
> deliberate: it means an update to QGIS can never break Trajecta, and the two
> cannot interfere with each other.

---

## Preparing your data

Trajecta expects:

| Input | Format |
|---|---|
| DEM | GeoTIFF (`.tif`), in a **projected** CRS with metres as units (e.g. UTM) |
| Sample points / origin / destinations | `.shp`, `.geojson`, `.kml`, `.gml` or `.csv` |
| Cost modifiers (optional) | polyline vector with a `cost` field, or a multiplier raster |

Two rules matter, and the interface checks both before starting:

1. **Every layer must use the same coordinate reference system as the DEM.**
2. **All points must fall inside the DEM extent.**

A DEM in geographic coordinates (degrees) will produce meaningless distances:
reproject it to a metric CRS first.

---

## Running your first analysis

1. Choose the mode: **FETE** (movement across a whole landscape) or **LCPA**
   (routes from one origin to one or more destinations).
2. Select the DEM, the point layers and an output folder.
3. Leave the algorithm settings at their defaults for a first run —
   16-connectivity and the Modified Tobler function are sensible choices.
4. Under **Hardware resources**, set the number of CPU threads. About 12 is a
   good compromise on a modern laptop: more threads add little because the
   computation is limited by memory speed, not by processor count.
5. Press **Run analysis**.

Results appear in the output folder and are loaded automatically into the
**Viewer** tab.

---

## Uninstalling

Use *Settings → Apps → Installed apps → Trajecta → Uninstall*, or the
`Uninstall.exe` in the installation folder. Your analysis outputs are never
placed inside the installation folder, so they are not affected.

---

## Troubleshooting

**The program will not start at all.**
Check that you are on 64-bit Windows 10 or later. If you extracted the files
manually instead of running the installer, install it properly: the libraries
must sit next to `trajecta.exe`.

**"GDAL not found" in the status bar.**
This should not happen with a normal installation. It usually means the
installation folder is incomplete — reinstall. As a workaround, click
**Locate GDAL folder…** and select the folder containing `trajecta.exe`.

**"X of Y points are outside DEM extent".**
The points and the DEM do not overlap, or they are in different coordinate
systems. Open both in QGIS and check.

**An analysis is taking a very long time.**
FETE's cost grows with the number of sample points *and* with how widely they
are spread. A sample covering the whole DEM forces every search to cross the
entire raster. Reducing the number of points is the most effective remedy: the
run time is proportional to it.

**Large memory pages are shown as unavailable.**
This is an optional performance setting and is off by default; it never changes
results. See the `?` next to it for how to enable it.

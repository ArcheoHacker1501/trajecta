# Trajecta

> [!IMPORTANT]
> - To install the latest version of Trajecta (v1.0.1) [download the installer](https://github.com/ArcheoHacker1501/trajecta/releases/tag/v1.0.1) and click on it once the download is done. Then follow the instructions.

Trajecta is a free, open-source least-cost analysis (LCA) software distributed under the GNU General Public License 3.0 and specifically developed to provide a seamless and user-friendly experience to every type of user, even without prior experience. It is primarily designed to be used by archaeologists, historians, geographers, and other researchers who need to model movement across landscapes to investigate spatial patterns in the Ancient World. Trajecta is completely written in C++ and Qt for fast and efficient computation. The source code of Trajecta is available on [GitHub](https://github.com/ArcheoHacker1501/trajecta).

For an introductory walkthrough on how to use Trajecta, you can launch this in-app **tutorial**. You can also click on the **?** badge beside any field to get information about the selected parameter or function. For additional details on Trajecta's features, you can also refer to the specific pages in this **Guide** section.

At its core, Trajecta models movement across a landscape with **FETE** and **LCPA**, refines and checks the result with **NNI interpolation** and **route comparison**, and tests it against real settlement patterns with **site–corridor coherence**. Finally, the Viewer offers a simple platform to visualize the results of the computations.

The list on the left opens the rest of the **Guide**, and each tool is described in detail. For contacts and information about the author, please refer to the **About** section.

Trajecta was inspired and made possible thanks to the previous work of many scholars from different fields. All the references and sources used to develop Trajecta are listed in the **References** section of this Guide.

**IMPORTANT**: Be patient, this software is currently under development and can contain bugs or errors! For bug reporting, problems during the installation, or to suggest improvements or additional features to be included in Trajecta, please use this **report form**.

# FETE — From Everywhere To Everywhere

Trajecta provides two complementary workflows for modeling movement across terrain: FETE, described here, and LCPA on the next page. Both use anisotropic cost functions (e.g. Modified Tobler's Hiking Function, see Algorithm parameters) and support cost surface modifiers (e.g. waterbodies, terrain indexes).

From-Everywhere-To-Everywhere (FETE) is a GIS-based method initially conceptualized by White and Barber (2012). FETE allows to model probable movement corridors across a landscape without requiring predetermined origin and destination points as, instead, in Least-Cost Path Analysis (see next section). In this way, instead of calculating single paths between pre-selected points, FETE allows to model the general mobility characterizing a region. This is done by using a grid containing hundreds, thousands or even hundred of thousands regularly or randomly scattered points. The FETE algorithm implemented by Trajecta then calculates all the least-cost paths connecting every point to every other point of the grid. Based on all the LCPs calculated, overlap analysis is then computed and a density raster is created, with the same resolution as the input DEM. Each cell of the density raster contains a number. This number is the arithmetical sum of all the LCPs that cross that specific cell. The most crossed cells (i.e. those with highest values) represent the busiest and most travelled routes. Different color gradients can be used to display most probable paths among all calculated LCPs corridors. To compute all these LCPs, DEM or other elevation based data can be used to calculate slope which can then be transformed using cost functions (e.g. Tobler's Hiking Function). Additional costs can be added as for waterbodies or different types of terrain via raster or vector inputs.

The density raster generated can then be used in different ways. For example, it can be compared to known routes or settlements in order to assess possible relationships between mobility across a region and settlement patterns.

## Sample points

In FETE mode the sample points can either be **imported from a file** (e.g. .shp, .geojson, .csv), or **directly generated from the DEM**.

Generation takes two parameters. The **density** is expressed either as a
**point spacing** (one point every N rows and every N columns, so the count falls
with the square of N) or as a **target number of points**, from which the spacing
is derived using the number of usable DEM cells. The **arrangement** is either a
**regular grid**, which puts every point at the same offset inside its block, or
**stratified random**, which picks one random cell per block: same density, none
of the regularity a grid imposes on the result. A stratified random layer is
reproducible from its **seed**.

Points are only placed on cells a path can actually cross, so NoData areas stay
empty. The layer is written to the output folder as a shapefile *before* the
analysis starts and is then read back as its input: what the run consumed is always
on disk, and it appears in the **Viewer** as a selectable overlay. The setup page
shows the resulting point count while you type, with a warning above roughly 50,000
points — FETE cost grows with the square of the number of points.

## Cost modifiers

Modifiers let you make specific features (water bodies, restricted areas, specific types of terrain) more
expensive to cross. Vector modifiers are rasterized onto the DEM grid: every segment is
first clipped to the raster's bounds (Liang & Barsky 1984), then walked cell by cell
with **Bresenham's line algorithm** (Bresenham 1965). The clipping is what keeps a layer
delivered in the wrong CRS from being merely wrong instead of also unbearably slow.
The **polyline buffer** widens rasterized lines so paths cannot
slip diagonally across them. The **barrier threshold** turns extreme multipliers
(e.g. 999999) into hard barriers: cells at or above the threshold are excluded from
movement, which also keeps the computation fast.

## Algorithm parameters

Most of the following algorithm parameters are shared by both FETE and LCPA modes.

**Neighbours** — connectivity of the search grid (8, 16, 24, 32, 64, or any
admissible number through *Custom*). Higher values allow finer path angles at the
price of speed. A connectivity radius of 16 (Knight's Move) is the usual choice.
See the **?** next to the field for which totals are admissible and why.

**The search** — least-cost paths are found with **Dijkstra's algorithm**
(Dijkstra 1959) over the grid that connectivity defines, with no heuristic: what comes back
is the cheapest path that exists in that grid, not a good approximation of it. Because every
move is priced on its own and the price depends on the direction of travel, the graph is
**directed** — which is why A→B and B→A need not follow the same cells.

## How the cost of one cell-to-cell move is computed

Every cost function in Trajecta is applied **to a single move between two cell
centres**, not to a cell in isolation. For each move the engine computes:

<table>
<tr><td>`dh`</td><td>horizontal distance between the two cell centres, in metres —
from the neighbour offset and the DEM cell size, so a diagonal move is longer than an
orthogonal one</td></tr>
<tr><td>`dz`</td><td>elevation difference, `z(to) − z(from)`, in metres —
**signed**: positive uphill, negative downhill</td></tr>
<tr><td>`S`</td><td>the slope of that move, `S = dz / dh` (a tangent, not an angle).
Where a formula below uses a percentage the engine passes `S × 100`</td></tr>
</table>

The cost function converts `S` into a walking speed `v`, and the cost of
the move is the time it takes:

> `cost = (dh / 1000) / v`   →   **hours**, when
> `v` is in km/h

Because `S` keeps its sign, all three functions are **anisotropic**: going up a
slope and coming back down it cost different amounts, and A→B is not the same as B→A.
Any cost modifiers you supply multiply this base cost afterwards.

The **base cost surface** raster is *not* what the path search uses. It is a
summary for inspection: the mean of the move costs from each cell to all of its
neighbours. The search itself always uses the individual move costs.

## The cost functions, exactly as implemented

**1 — Modified Tobler's Hiking Function** (Tobler 1993; inverted into time as
described by White 2015)

> `v = 6 · e^(−3.5 · |S + 0.05|)`   km/h  → 
> `cost = (dh/1000) / v` hours

> Fastest at `S = −0.05`, i.e. a 5% downhill, where `v = 6` km/h;
> on the flat 5.4 km/h. This is Tobler's **on-path** form: the × 0.6 factor that
> Tobler suggests for **off-path** travel is **not** applied. If your route is
> cross-country, expect this function to be optimistic by roughly that factor.

**2 — Márquez-Pérez et al. (2017)**

> `v = 4.8 · e^(−5.3 · |(0.7 · S) + 0.03|)`   km/h

> A recalibration of Tobler on GPS tracks from marked trails in Spanish natural
> parks. Slower overall (4.8 instead of 6) and it penalises slope more sharply.
> Fastest at `S ≈ −0.043`.

**3 — Irmischer & Clarke (2017)** — **on-path, male** variant

> `v = 0.11 + e^(−(S% + 5)2 / 1800)`   m/s,
>   with `S% = 100 · S`   and   `1800 = 2 · 30²`

> The paper publishes four variants (male/female × on-path/off-path);
> Trajecta implements the **on-path male** one. The others differ by a 0.67 factor on
> the exponential and a +2 rather than +5 shift (off-path), and by an overall × 0.95
> (female). Derived from GPS tracks of 200 cadets, so it includes way-finding time, which
> is why it is slower than Tobler on the flat. The constant 0.11 m/s is a floor: this
> function never reaches zero speed, however steep the ground.

> *Note.* Trajecta feeds this function the **signed** slope, so
> the +5 shift makes it anisotropic with its peak at a 5% downhill, as the shift is meant
> to express. Some other implementations pass `|S|` instead, which makes the function
> symmetric and moves the peak to the flat; results are therefore not directly comparable
> with those packages.

**4 — Herzog (2013)**, fitted to Minetti et al. (2002) — **energy, not time**

> `C(S) = 1337.8·S⁶ + 278.19·S⁵ − 517.39·S⁴
> − 78.199·S³ + 93.419·S² + 19.825·S + 1.64`

> `cost = C(S) · dh`   →   **kilojoules per kilogram** of walker

> The only function here that measures **effort** rather than duration.
> Herzog fitted this sixth-degree polynomial to the treadmill measurements of Minetti et al.,
> and it has the shape the data show and every speed model misses: the minimum sits at about a
> **10.5% downhill**, and the curve rises on *both* sides — because braking down a
> steep slope costs energy too. Tobler and the others simply get faster and faster downhill.

> **Read the units.** Every cost in a Herzog run — the cost
> surfaces, the accumulated cost behind each path — is in kJ/kg, not hours. Those rasters
> cannot be compared with, or added to, the output of any other function. Trajecta says so in the
> run summary, in the manifest and under the selector, but the file itself carries no unit.

> **Range.** Minetti's data span roughly ±45% slope
> (about ±24°). Beyond that the polynomial is extrapolation: it stays positive and climbs
> steeply, which is right in direction but is no longer a measurement. Use the slope cut-off below
> to keep a run inside the calibrated range.

**5, 6 — Campbell et al. (2019)**, asymmetric Lorentz, 5th and 50th percentile

> `v = c / (π·b·(1 + ((θ − a)/b)²)) + d + e·θ`
>   m/s, with `θ` the slope in **degrees**

<table border="1" cellspacing="0" style="margin-left:20px">
<tr><th>Percentile</th><th>c</th><th>b</th><th>a</th><th>d</th><th>e</th></tr>
<tr><td>5th</td><td>36.813</td><td>14.041</td><td>−1.527</td><td>0.320</td><td>−0.00273</td></tr>
<tr><td>50th</td><td>63.660</td><td>10.064</td><td>−2.171</td><td>0.628</td><td>−0.00463</td></tr>
</table>

> Fitted to **421,247 GPS activities** from 29,928 people recorded through
> Strava — by far the largest empirical basis of any function here. The dataset mixes walking,
> jogging and running, so the paper publishes one parameter set per percentile of the population
> rather than a single average.

> **Which percentile to choose.** The authors recommend the
> **5th** as representative of ordinary hiking: on the flat it gives about 1.15 m/s
> (4.1 km/h), a normal walking pace. The **50th** is the median of the whole dataset and
> reaches about 2.55 m/s (9.2 km/h) on the flat — that is a run, not a walk. Use it
> only if fast movement is what you mean to model.

> **Range.** The fit is calibrated for slopes below 30°; the paper
> discarded steeper segments. The other percentiles (1st, 25th, 75th, 95th and the rest) exist in the
> paper's supplementary material and can be added on request.

## Slope cut-off

Off by default. When it is on, a move steeper than the limit you set is not expensive —
it is **impossible**, and the engine removes it from the graph. The limit applies to the
**move**, not to the cell: a terrace can still be entered from the side when the approach
from below is too steep, which is how real terrain behaves. Uphill and downhill are set
separately, because a slope that can be climbed slowly is often refused on the way down.

Two uses: keeping routes out of ground nobody would walk, and keeping a cost function inside
the range it was measured in (see Herzog and Campbell above). Set it too tight and a destination
can become unreachable — the run then reports the paths it could not compute rather than
inventing one.

## Units — what the numbers in the outputs actually mean

<table border="1" cellspacing="0">
<tr><th>Quantity</th><th>Unit</th><th>Notes</th></tr>
<tr><td>DEM elevation `z`</td><td>metres</td><td>assumed; a DEM in feet gives slopes too small by 3.28×</td></tr>
<tr><td>Cell size, `dh`</td><td>metres</td><td>taken from the DEM geotransform, so the CRS must be **projected**, never geographic degrees</td></tr>
<tr><td>Slope `S`</td><td>dimensionless (m/m)</td><td>a tangent. `S = 1` is 45°, not 100°</td></tr>
<tr><td>Slope raster output</td><td>**degrees** or **percent**</td><td>degrees with Tobler and Campbell, percent with the others; stated in the run summary and in the manifest. This affects the *exported raster only* — the cost functions always receive `S = dz/dh`</td></tr>
<tr><td>Speed `v`</td><td>km/h (1–2), m/s (3, 5, 6)</td><td>converted internally; the m/s functions are multiplied by 3.6</td></tr>
<tr><td>Cost of one move</td><td>**hours**, except Herzog: **kJ/kg**</td><td>printed in every run summary and written into the manifest as *cost units*</td></tr>
<tr><td>Base / additional / total cost surface</td><td>same as the move</td><td>mean over the neighbours of a cell — a summary, not what the search uses</td></tr>
<tr><td>Accumulated cost (internal)</td><td>same as the move</td><td>sum of move costs along the cheapest route found</td></tr>
<tr><td>FETE density raster</td><td>**count** of paths</td><td>a pure integer count, not a cost and not a time</td></tr>
<tr><td>Cost modifiers</td><td>**dimensionless multiplier**</td><td>multiplies the base cost, so 2.0 means "twice as slow here"</td></tr>
</table>

The five time-based functions return hours, so their cost surfaces are **numerically
comparable**: a cell at 0.5 means half an hour in all of them. **Herzog is not** — its
rasters are in kJ/kg and must never be compared with, subtracted from, or added to the
others. What is comparable to nothing at all is a cost surface against a density raster:
they measure different things.

None of the six models load carriage or ground surface. Herzog is the only one that
represents effort; the other five represent duration, and should not be described as a
measure of effort however intuitive that reading is.

**Path smoothing buffer** — buffer in cells applied around each computed path
when accumulating results. This function allows to calculate wider paths, thus maximising the possibility that actual, historical path(s) were located inside the modeled high-traversability areas.

## Input requirements

<table border="0" cellspacing="0">
<tr><th align="left">Input</th><th align="left">Requirements</th></tr>
<tr><td>**DEM**</td><td>GeoTIFF (.tif/.tiff), georeferenced, with a CRS.</td></tr>
<tr><td>**Sample points**</td><td>.shp, .geojson/.json, .kml, .gml/.xml or .csv
    (coordinate columns named x/y, lon/lat or easting/northing) if imported; or
    generated directly from the DEM instead.</td></tr>
<tr><td>**Vector modifiers** (optional)</td><td>Polylines with a float **cost** field
    holding the multiplier; for .csv the geometry must be in a WKT column.</td></tr>
<tr><td>**Raster modifiers** (optional)</td><td>GeoTIFF with the same dimensions as the DEM;
    cell values are multipliers (1.0 = unchanged, 2.0 = double cost).</td></tr>
</table>

## Outputs

Slope raster, base cost surface, and — with modifiers — the additional and total
cost surfaces; the **path-density raster**; and the sample points shapefile when
the points were generated from the DEM.

Every run also writes a **run manifest** next to its results, unless the
option is turned off: a plain text record of the version, the inputs with their
content hashes, every setting, the hardware and the files produced.

# LCPA — Least-Cost Path Analysis

For a detailed introduction to Least-Cost Path Analysis (LCPA), see White (2015). LCPA is a spatial analysis method, typically implemented in GIS environments, that identifies the minimum cumulative-cost route between two points across a cost surface. Each cell of the raster grid represents the cost of traversing it – expressed in terms of physical effort, time, energy expenditure, or resistance to movement – calculated as a function of variables such as slope, land cover, hydrography, or other environmental and cultural factors relevant to the study context.

Algorithmically, the raster surface is treated as a weighted graph (cells as nodes, adjacencies as edges), and the problem is solved with shortest-path algorithms such as Dijkstra's or A* (A-star), which compute both the accumulated cost surface from the source and the optimal path to one or more destinations. A key distinction is between isotropic cost (equal in every direction) and anisotropic cost (direction-dependent, as with slope varying between ascent and descent – Tobler's Hiking Function being the classic example for pedestrian movement).

In archaeology, LCPA is widely used to reconstruct probable movement corridors, ancient route networks, or trade paths from digital elevation models, on the assumption that human movement tends to minimize effort. It should nonetheless be used cautiously when investigating ancient routes: LCPA inherently introduces a strong selection bias as it necessarily needs the user to select at least two points (one origin and at least one destination) to be connected. Importantly, the two points might have never been actually connected in ancient times. Consequently, this selection bias must always be taken into account and additional proofing of the results should be always provided.

To compute LCPs in Trajecta, DEM or other elevation based data can be used to calculate slope which can then be transformed using different cost functions (e.g. Modified Tobler's Hiking Function, Irmischer and Clarke 2017, Herzog 2013). Additional costs can be added as for waterbodies or terrain indexes using raster or vector input layers.

## Cost modifiers

Modifiers let you make specific features (water bodies, restricted areas, specific types of terrain) more
expensive to cross. Vector modifiers are rasterized onto the DEM grid: every segment is
first clipped to the raster's bounds (Liang & Barsky 1984), then walked cell by cell
with **Bresenham's line algorithm** (Bresenham 1965). The clipping is what keeps a layer
delivered in the wrong CRS from being merely wrong instead of also unbearably slow.
The **polyline buffer** widens rasterized lines so paths cannot
slip diagonally across them. The **barrier threshold** turns extreme multipliers
(e.g. 999999) into hard barriers: cells at or above the threshold are excluded from
movement, which also keeps the computation fast.

## Algorithm parameters

Most of the following algorithm parameters are shared by both FETE and LCPA modes.

**Neighbours** — connectivity of the search grid (8, 16, 24, 32, 64, or any
admissible number through *Custom*). Higher values allow finer path angles at the
price of speed. A connectivity radius of 16 (Knight's Move) is the usual choice.
See the **?** next to the field for which totals are admissible and why.

**The search** — least-cost paths are found with **Dijkstra's algorithm**
(Dijkstra 1959) over the grid that connectivity defines, with no heuristic: what comes back
is the cheapest path that exists in that grid, not a good approximation of it. Because every
move is priced on its own and the price depends on the direction of travel, the graph is
**directed** — which is why A→B and B→A need not follow the same cells.

## How the cost of one cell-to-cell move is computed

Every cost function in Trajecta is applied **to a single move between two cell
centres**, not to a cell in isolation. For each move the engine computes:

<table>
<tr><td>`dh`</td><td>horizontal distance between the two cell centres, in metres —
from the neighbour offset and the DEM cell size, so a diagonal move is longer than an
orthogonal one</td></tr>
<tr><td>`dz`</td><td>elevation difference, `z(to) − z(from)`, in metres —
**signed**: positive uphill, negative downhill</td></tr>
<tr><td>`S`</td><td>the slope of that move, `S = dz / dh` (a tangent, not an angle).
Where a formula below uses a percentage the engine passes `S × 100`</td></tr>
</table>

The cost function converts `S` into a walking speed `v`, and the cost of
the move is the time it takes:

> `cost = (dh / 1000) / v`   →   **hours**, when
> `v` is in km/h

Because `S` keeps its sign, all three functions are **anisotropic**: going up a
slope and coming back down it cost different amounts, and A→B is not the same as B→A.
Any cost modifiers you supply multiply this base cost afterwards.

The **base cost surface** raster is *not* what the path search uses. It is a
summary for inspection: the mean of the move costs from each cell to all of its
neighbours. The search itself always uses the individual move costs.

## The cost functions, exactly as implemented

**1 — Modified Tobler's Hiking Function** (Tobler 1993; inverted into time as
described by White 2015)

> `v = 6 · e^(−3.5 · |S + 0.05|)`   km/h  → 
> `cost = (dh/1000) / v` hours

> Fastest at `S = −0.05`, i.e. a 5% downhill, where `v = 6` km/h;
> on the flat 5.4 km/h. This is Tobler's **on-path** form: the × 0.6 factor that
> Tobler suggests for **off-path** travel is **not** applied. If your route is
> cross-country, expect this function to be optimistic by roughly that factor.

**2 — Márquez-Pérez et al. (2017)**

> `v = 4.8 · e^(−5.3 · |(0.7 · S) + 0.03|)`   km/h

> A recalibration of Tobler on GPS tracks from marked trails in Spanish natural
> parks. Slower overall (4.8 instead of 6) and it penalises slope more sharply.
> Fastest at `S ≈ −0.043`.

**3 — Irmischer & Clarke (2017)** — **on-path, male** variant

> `v = 0.11 + e^(−(S% + 5)2 / 1800)`   m/s,
>   with `S% = 100 · S`   and   `1800 = 2 · 30²`

> The paper publishes four variants (male/female × on-path/off-path);
> Trajecta implements the **on-path male** one. The others differ by a 0.67 factor on
> the exponential and a +2 rather than +5 shift (off-path), and by an overall × 0.95
> (female). Derived from GPS tracks of 200 cadets, so it includes way-finding time, which
> is why it is slower than Tobler on the flat. The constant 0.11 m/s is a floor: this
> function never reaches zero speed, however steep the ground.

> *Note.* Trajecta feeds this function the **signed** slope, so
> the +5 shift makes it anisotropic with its peak at a 5% downhill, as the shift is meant
> to express. Some other implementations pass `|S|` instead, which makes the function
> symmetric and moves the peak to the flat; results are therefore not directly comparable
> with those packages.

**4 — Herzog (2013)**, fitted to Minetti et al. (2002) — **energy, not time**

> `C(S) = 1337.8·S⁶ + 278.19·S⁵ − 517.39·S⁴
> − 78.199·S³ + 93.419·S² + 19.825·S + 1.64`

> `cost = C(S) · dh`   →   **kilojoules per kilogram** of walker

> The only function here that measures **effort** rather than duration.
> Herzog fitted this sixth-degree polynomial to the treadmill measurements of Minetti et al.,
> and it has the shape the data show and every speed model misses: the minimum sits at about a
> **10.5% downhill**, and the curve rises on *both* sides — because braking down a
> steep slope costs energy too. Tobler and the others simply get faster and faster downhill.

> **Read the units.** Every cost in a Herzog run — the cost
> surfaces, the accumulated cost behind each path — is in kJ/kg, not hours. Those rasters
> cannot be compared with, or added to, the output of any other function. Trajecta says so in the
> run summary, in the manifest and under the selector, but the file itself carries no unit.

> **Range.** Minetti's data span roughly ±45% slope
> (about ±24°). Beyond that the polynomial is extrapolation: it stays positive and climbs
> steeply, which is right in direction but is no longer a measurement. Use the slope cut-off below
> to keep a run inside the calibrated range.

**5, 6 — Campbell et al. (2019)**, asymmetric Lorentz, 5th and 50th percentile

> `v = c / (π·b·(1 + ((θ − a)/b)²)) + d + e·θ`
>   m/s, with `θ` the slope in **degrees**

<table border="1" cellspacing="0" style="margin-left:20px">
<tr><th>Percentile</th><th>c</th><th>b</th><th>a</th><th>d</th><th>e</th></tr>
<tr><td>5th</td><td>36.813</td><td>14.041</td><td>−1.527</td><td>0.320</td><td>−0.00273</td></tr>
<tr><td>50th</td><td>63.660</td><td>10.064</td><td>−2.171</td><td>0.628</td><td>−0.00463</td></tr>
</table>

> Fitted to **421,247 GPS activities** from 29,928 people recorded through
> Strava — by far the largest empirical basis of any function here. The dataset mixes walking,
> jogging and running, so the paper publishes one parameter set per percentile of the population
> rather than a single average.

> **Which percentile to choose.** The authors recommend the
> **5th** as representative of ordinary hiking: on the flat it gives about 1.15 m/s
> (4.1 km/h), a normal walking pace. The **50th** is the median of the whole dataset and
> reaches about 2.55 m/s (9.2 km/h) on the flat — that is a run, not a walk. Use it
> only if fast movement is what you mean to model.

> **Range.** The fit is calibrated for slopes below 30°; the paper
> discarded steeper segments. The other percentiles (1st, 25th, 75th, 95th and the rest) exist in the
> paper's supplementary material and can be added on request.

## Slope cut-off

Off by default. When it is on, a move steeper than the limit you set is not expensive —
it is **impossible**, and the engine removes it from the graph. The limit applies to the
**move**, not to the cell: a terrace can still be entered from the side when the approach
from below is too steep, which is how real terrain behaves. Uphill and downhill are set
separately, because a slope that can be climbed slowly is often refused on the way down.

Two uses: keeping routes out of ground nobody would walk, and keeping a cost function inside
the range it was measured in (see Herzog and Campbell above). Set it too tight and a destination
can become unreachable — the run then reports the paths it could not compute rather than
inventing one.

## Units — what the numbers in the outputs actually mean

<table border="1" cellspacing="0">
<tr><th>Quantity</th><th>Unit</th><th>Notes</th></tr>
<tr><td>DEM elevation `z`</td><td>metres</td><td>assumed; a DEM in feet gives slopes too small by 3.28×</td></tr>
<tr><td>Cell size, `dh`</td><td>metres</td><td>taken from the DEM geotransform, so the CRS must be **projected**, never geographic degrees</td></tr>
<tr><td>Slope `S`</td><td>dimensionless (m/m)</td><td>a tangent. `S = 1` is 45°, not 100°</td></tr>
<tr><td>Slope raster output</td><td>**degrees** or **percent**</td><td>degrees with Tobler and Campbell, percent with the others; stated in the run summary and in the manifest. This affects the *exported raster only* — the cost functions always receive `S = dz/dh`</td></tr>
<tr><td>Speed `v`</td><td>km/h (1–2), m/s (3, 5, 6)</td><td>converted internally; the m/s functions are multiplied by 3.6</td></tr>
<tr><td>Cost of one move</td><td>**hours**, except Herzog: **kJ/kg**</td><td>printed in every run summary and written into the manifest as *cost units*</td></tr>
<tr><td>Base / additional / total cost surface</td><td>same as the move</td><td>mean over the neighbours of a cell — a summary, not what the search uses</td></tr>
<tr><td>Accumulated cost (internal)</td><td>same as the move</td><td>sum of move costs along the cheapest route found</td></tr>
<tr><td>FETE density raster</td><td>**count** of paths</td><td>a pure integer count, not a cost and not a time</td></tr>
<tr><td>Cost modifiers</td><td>**dimensionless multiplier**</td><td>multiplies the base cost, so 2.0 means "twice as slow here"</td></tr>
</table>

The five time-based functions return hours, so their cost surfaces are **numerically
comparable**: a cell at 0.5 means half an hour in all of them. **Herzog is not** — its
rasters are in kJ/kg and must never be compared with, subtracted from, or added to the
others. What is comparable to nothing at all is a cost surface against a density raster:
they measure different things.

None of the six models load carriage or ground surface. Herzog is the only one that
represents effort; the other five represent duration, and should not be described as a
measure of effort however intuitive that reading is.

**Path smoothing buffer** — buffer in cells applied around each computed path
when accumulating results. This function allows to calculate wider paths, thus maximising the possibility that actual, historical path(s) were located inside the modeled high-traversability areas.

## Input requirements

<table border="0" cellspacing="0">
<tr><th align="left">Input</th><th align="left">Requirements</th></tr>
<tr><td>**DEM**</td><td>GeoTIFF (.tif/.tiff), georeferenced, with a CRS.</td></tr>
<tr><td>**Origin**</td><td>Vector file with exactly one point (.shp, .geojson/.json,
    .kml, .gml/.xml or .csv): the starting location of the least-cost routes.</td></tr>
<tr><td>**Destinations**</td><td>Vector file with one or more points, same formats: the
    target(s) the optimal route(s) is/are computed to.</td></tr>
<tr><td>**Vector modifiers** (optional)</td><td>Polylines with a float **cost** field
    holding the multiplier; for .csv the geometry must be in a WKT column.</td></tr>
<tr><td>**Raster modifiers** (optional)</td><td>GeoTIFF with the same dimensions as the DEM;
    cell values are multipliers (1.0 = unchanged, 2.0 = double cost).</td></tr>
</table>

## Outputs

Slope raster, base cost surface, and — with modifiers — the additional and total
cost surfaces; the **paths raster** and the **paths polyline shapefile**, plus
the **cost corridor raster** when one was asked for.

Every run also writes a **run manifest** next to its results, unless the
option is turned off: a plain text record of the version, the inputs with their
content hashes, every setting, the hardware and the files produced.

# Post-processing: NNI — Natural Neighbour Interpolation

The **Post-processing** page turns a FETE density raster into a smooth,
continuous surface using **discrete Sibson (natural neighbour) interpolation
(Sibson 1981; Park et al. 2006)**.
Cells at or above the **sample threshold** act as sample points; every other
cell receives the average of the samples whose influence area it would claim.
Sample values are preserved exactly. The optional **max search radius** caps
how far the interpolation reaches into empty areas (beyond it, cells take the
value of their nearest sample), which keeps large rasters fast. After a
successful FETE run the density raster is filled in automatically, allowing for direct post-processing.

## Input requirements

A **density raster** (GeoTIFF), typically the FETE output — the interface fills
this in automatically after a successful FETE run.

## Outputs

The **interpolated raster**, written next to the density raster it was made
from and named after it. A run manifest is written alongside it too, unless the
option is turned off.

# Post-processing: comparison with a known route

The second tool on the **Post-processing** page does not compute anything new: it
**measures a computed route against a route that is actually known** — a Roman road, a
drover's track, a surveyed path. This is the step that turns a least-cost path from an
illustration into a claim that can be wrong; without it a model can only ever agree with
itself.

It takes two vector layers of lines — normally the LCPA paths shapefile and the known
route — and a **tolerance** in metres, which is what you consider "close". Both layers
must be **projected and in the same CRS**: distances in degrees would be meaningless, so
they are refused rather than silently reported.

The report gives, in both directions, the **median**, the **90th percentile** and the
**maximum** distance from one line to the other, and the **share of each line that runs
within the tolerance** of the other. A distribution rather than a single number, because a
route can follow the real one closely for 9 km and then take the wrong side of a hill for
1 km — and an average hides exactly that. Both directions are needed too: a short computed
path lying on top of a long known one is close in one direction and far in the other. The
worst disagreement anywhere, the maximum of the two, is the **Hausdorff distance**
(Hausdorff 1914).

The comparison is also available from the command line
(`--compare-routes`), which makes a whole set of routes testable in one script.

## Input requirements

Two **vector line layers** (.shp, .geojson/.json, .kml, .gml/.xml or .csv):
the computed routes — normally the LCPA paths shapefile — and the known route to
test them against. Both must be **projected and in the same CRS**.

## Outputs

No raster and no layer — the output is the **report itself**, printed in the
panel and in the log, which can be copied into a table or a publication as it
stands.

# Post-processing: site–corridor coherence

The third tool asks the question the FETE was computed for: **do the sites sit on the
movement the surface predicts?** It takes the FETE surface and a **point layer of
sites**, and gives every site a score, the sample a verdict, and — this is the part that
makes two periods comparable — a statement of how much of that could have happened by
chance.

## The four questions the tool answers

In simple terms, the site-corridor coherence tool aims at answering four main questions:

1. **Are any of the sites near a corridor at all?** If almost none is, everything below is
noise and you can stop your analysis here.
2. **How far are the sites from the corridors?** The first quantity: near is not a yes or no,
it is a distance. Two sites (e.g. site A and site B) can be equally considered 'near' to a
corridor if this is within a distance of — for example — 500 m. Nonetheless, this same corridor
might be 400 m from site A and only 40 m from site B. Clearly, this is a significant difference
that would be impossible to detect with a binary 'near/far' classification.
3. **How much corridor is around the sites?** Two sites (e.g. site A and site B) at a same
distance from a route are not in the same place if one has a single thread nearby and the other a
whole braid. Site A might be near a single, thin corridor while Site B might be near several,
larger corridors. This makes a big difference when assessing site-corridor coherence. It is
important to know not only how many corridors are near a single site, but also how big these
corridors are.
4. **How busy is the ground around the sites?** Not how much corridor, but how heavily
travelled it is. You can have sites near several or even a lot of corridors, but these corridors
might be only limitedly travelled. On the contrary, you can have a site with just a corridor in
its vicinity, but that corridor might be extremely busy.

Every number below is built so that **two runs can be compared** — two periods, two
regions — even when the two FETE surfaces were computed from different numbers of points
or at different resolutions. That is the whole purpose of the tool, and it constrains how each
measure is defined.

## 1. Are any of the sites near a corridor at all?

The **distance bands** table: the share of sites within 0, 100, 250, 500, 1000 and 2500
metres of the nearest corridor cell — you can set your own list. "Within 0 m" means
standing on a corridor cell.

The distances are fixed metres and **not** fractions of the radius, and that is deliberate:
it means two runs can be laid side by side row for row whatever radius each was given. Bands
finer than one raster cell are dropped, because a raster cannot resolve them — on a 90 m
grid a site is either on a corridor or at least 90 m from one, so a 50 m band would only repeat
the 0 m one.

## 2. How far are the sites from the corridors?

Reported as the **median**, the **deciles** (p10, p25, p75, p90) and a small
**histogram**. The median is the middle site: half are closer, half are further.

Why not just the median? Because a single middle value can hide the shape of a sample. If half
the sites sit almost on the corridors and the other half are kilometres away, with almost nothing
in between, the median lands in the empty middle and describes *nobody*. That pattern is
called a **bimodal** distribution, and it is common and interesting: it usually means you have
two kinds of site rather than one. The deciles reveal it — a big jump between two
neighbouring figures is the gap between the two groups — and the histogram shows it
directly. **If the deciles and the histogram disagree with the median, believe them.**

These figures do not depend on the radius at all. That is what makes them the ones to quote.

## 3. How much corridor is around the sites?

The **proximity index**: of the cells within the radius that have data, the percentage that
are corridor cells.

Why a percentage and not a count? Because a count is not comparable. The same piece of ground
holds nine times as many cells at 30 m resolution as at 90 m, so a site would score nine times
higher merely for being measured on a finer raster. A **share** cancels that out: if 8%
of the neighbourhood is corridor, it is 8% at either resolution.

Beside it is **enrichment**, which is the proximity index divided by the corridor's share of
the *whole* surface. This has a property worth understanding, because it is what lets you say
whether a number is large:

> *A point dropped at random on the map has, on average, exactly the
> surface's own share of corridor around it. So **enrichment 1.00 is chance, exactly** —
> not approximately, and not "as estimated by a simulation". Enrichment 5.00 means five times as
> much corridor as average ground. The higher the enrichment, the more the corridor around a site
> stands out from the rest of the surface.*

## 4. How busy is the ground around the sites?

The **intensity index**. Being surrounded by corridor is one thing; being surrounded by a
*heavily travelled* corridor is another. This measures the second.

It is built in three steps, and each removes a specific way of being wrong:

1. **Take the logarithm of every cell's value.** A FETE cell holds a count of paths, and those
counts are wildly uneven — most cells near zero, a few in the millions. If you simply
averaged them, one enormous cell would dominate its whole neighbourhood and the measure would stop
describing the area and start reporting its single busiest cell. On a logarithmic scale a cell ten
times busier counts more, but not ten times more, which is the behaviour we want.
2. **Weight by distance.** Cells close to the site count fully, cells at the edge of the radius
count nothing, in a straight line between the two.
3. **Convert the result to a percentile.** The same weighted average is measured at tens of
thousands of places across the surface, and the site is scored against that yardstick.

That third step is what makes the number comparable. A surface built from a million source
points has path counts hundreds of times larger than one built from ten thousand — but if
every place on both maps is scaled by the same factor, the *ordering* is unchanged, so the
percentile is unchanged. **50 is the average location on the surface**, and 64 really does mean
"busier than 64% of this map".

## Means and medians: which figure to quote

For questions 3 and 4 the report prints **both** a mean and a median, and here —
unusually — **the mean is the one to quote.** The reason is that both reference points are
statements about a mean: the expected share of corridor around a random point is exactly the
surface's share, and mid-ranks make the average percentile exactly 50. Neither holds for a
median.

You will usually find the medians much lower, often zero, and that is **normal, not a fault**.
Corridors are thin, linear and clustered, so on a typical FETE the majority of locations —
sites and random points alike — have no corridor within reach at all, which puts the median
at zero for both. A median enrichment of 0.00 beside a mean of 2.59 is exactly what a real
landscape looks like.

## What counts as a corridor

Distances are measured to the nearest **corridor cell**, so this setting decides what
everything else is measured towards. The default is **the top 1%** of the surface by rank,
because on a FETE surface the cells carrying real traffic are almost always inside the top
per cent.

**Use the percentage filter for anything comparative**, and it is the default for that
reason. Selecting the top q% by rank returns exactly q% of the valid cells in *every* dataset,
by construction — not approximately, and not by luck. That is what makes two surfaces
comparable at all. **Otsu's method** (Otsu 1979), computed on the logarithm of the values, finds
a threshold automatically on a single dataset; it reports which percentile it landed on and warns
when the surface has no clean split. A raw value can also be given, but a raw threshold is
**not** comparable between surfaces built from different numbers of points, because the values
themselves are on different scales.

The threshold that was actually used is always reported three ways — as a value, as a
percentile, and as a share of the surface. Those can differ from what was asked: on a sparse
surface where 99% of cells are exactly zero, "the top 1%" cannot be cut anywhere except at
the first non-zero value, and the report says so rather than pretending.

Once the corridor cells are known, **every** cell of the raster is given its distance to
the nearest one, in a single pass over the grid, with the separable distance transform of
Felzenszwalb & Huttenlocher (2012). That distance is **exact** — not the few
per cent of error the usual two-pass chamfer approximations leave — and it is computed
once for the whole surface rather than once per site, which is what makes the distance raster
free and the sensitivity curve nearly free.

Cells with no data stay **missing** throughout — they are not zero. Zero means
"measured, and nothing passes here", which is a fact about the landscape; counting the two
together would move every rank in the raster. What missing data costs a site is reported as
that site's **coverage**: the fraction of its disc that had data. A site whose coverage is
low is measured on less ground than the others, and the summary counts how many are below a
half.

## Could this have happened by chance?

A median distance of 118 m means nothing on its own. So the same median is computed again on
**999 point sets that have no relationship with the corridors** but share everything else
— the same area, the same number of points, and by default the same internal arrangement,
translated as a block, because settlements cluster and independent random points do not.

This is a **Monte Carlo significance test** (Hope 1968; Besag & Diggle 1977). The
p-value is the rank of the observed statistic among the simulated ones, taken as
(1 + the number of random sets at least as extreme) ÷ (999 + 1):
that added 1 is what keeps the test honest, and it is also why the smallest p the tool can
print is 0.001 — which means "the lowest this many replicates can resolve", not "one in
a thousand exactly". Moving the whole pattern as a block is the **random-shift** null
(Lotwick & Silverman 1982). It holds the sites' own spacing and clustering fixed and asks
only whether their *position* relative to the corridors is special, which is a stricter
question than scattering independent points — that alternative, offered as
*scattered points*, is the complete spatial randomness of classical point pattern
analysis (Baddeley et al. 2015). Shifts that would push any site off the raster or onto
missing data are discarded; if the sites cover the surface so completely that too few shifts
survive, the tool says so in the log and falls back to scattered points, rather than testing
against a handful of nearly identical sets.

**Only the distance is tested this way, and that is on purpose.** A distance has no natural
reference point — there is nothing to compare 118 m against without simulating it. The other
two measures already carry their own: enrichment is 1.00 under chance and the intensity index is
50, both exactly and by construction. Running a simulation to rediscover a number we already know
would only add a column of statistics to be misread.

What comes out is reported in metres rather than as a test statistic: *observed 118 m,
expected 240 m, with 95% of the random sets falling between 190 and 310 m*. Beside it is the
**ratio**, which is the figure to carry between periods: 0.5 means the sites are half as far
from a corridor as chance would put them, and unlike a raw score it does not depend on the units,
the area or the size of the sample. A sentence of the kind the tool is for: *"in the earlier
period the sites are 0.31× as far from the corridors as chance predicts; in the later one
0.87× — the relationship between settlement and natural routes weakens."*

## Comparing two datasets

Because this is what the tool is for, it is worth being explicit about what is and is not
required.

**Not required:** the same resolution, the same number of FETE source points, or the same
extent. Every measure above is defined so as not to depend on any of them — that is why the
proximity index is a share rather than a count, why enrichment divides by the surface's own
corridor share, and why the intensity index ends in a percentile.

**Worth knowing anyway:** a surface computed from few source points is a *noisier*
estimate of the same thing. The units are the same and the comparison is valid, but the
measurement carries more error, so small differences between two runs should not be pressed. And
one thing the intensity index deliberately cannot tell you: because every surface's average
location scores 50 by construction, it says "this site is busier than 64% of *its own*
region" — never "this region is busier than that one". That is a different question, and a
fragile one to ask of path counts.

## Sensitivity to the radius

Turning this on repeats the analysis at several radii and prints one row each. It costs
little, because ranking the surface and measuring every cell's distance to a corridor
do not depend on the radius and are done once. It is worth having whenever the result will be
shown to someone else, because it answers their first question in advance: **a relationship
that holds across the whole range is really a relationship; one that appears at a single radius
is usually the radius and should not be taken for reference.** Questions 1 and 2 do not appear in that table at all — they do not
change with the radius, which is why they are the headline result.

## Input requirements

A **FETE surface** (GeoTIFF), raw or interpolated with NNI, and a **point
layer of sites** (.shp, .geojson/.json, .kml, .gml/.xml or .csv) in the same
projected CRS as the raster.

## Outputs

A **table (.csv)** with one row per site: `dist_m` (metres to the nearest
corridor), `prox_idx` (the proximity index), `enrich` (enrichment, 1.00 =
chance), `inten_idx` (the intensity index, 50 = the average location),
`rank_site`, `coverage`, an edge flag and a `class` —
ON_CORRIDOR, NEAR_THIN, DIFFUSE or OFF. A **point layer** (GeoPackage or shapefile) carries the
same columns plus the input layer's own attributes; the **distance raster** holds, in every
cell, its distance in metres to the nearest corridor — the quickest way to see the catchment
of the network and to notice a threshold set too generously; and a **summary (.txt)** identical
to the report on screen, glossary included, so that the supplementary data of a paper and the
screen cannot disagree. Sites that fell outside the raster appear in the table marked as such, so
nothing disappears silently.

Optionally, a **histogram script (.R)** redraws question 2's distance histogram as a
**ggplot2** figure — the same bins and counts shown on screen, not a fresh binning of the
raw distances, so the script and the report can never disagree about what the sample looks like.

When the run finishes the distance raster and the scored sites open in the **Viewer**
together, and the sites are drawn **coloured by their score** — a plain ramp for the
proximity index, and for the intensity index a ramp that breaks at 50, the score the average
location gets, so that above and below are two different colours rather than two shades of one.
**Clicking a site** opens a panel at the bottom right with that site's whole row: the scores,
the class, and the columns your own layer brought with it. Clicking another site replaces it;
the cross closes it. Lines answer too, which makes the same panel useful over a set of LCPA
routes.

This tool is also available from the command line (`--coherence`), with the same
options, which makes a study of a dozen periods a single script.

# Credits

Trajecta uses several third-party software packages to work.

## GDAL

The Trajecta engine relies on the [**GDAL**](https://gdal.org/en/stable/) geospatial libraries. They are
installed together with Trajecta and sit next to the engine, so there is nothing
to install separately and no PATH to configure. The status at the bottom of the
sidebar should read **GDAL ready** from the first launch.

Trajecta looks beside its own engine before it looks anywhere else, which is
what makes this dependable: an installed copy always uses the libraries it
shipped with, and cannot be disturbed by any other GDAL on the machine — a QGIS
or OSGeo4W install included, whether it is updated, moved or removed.

If the status is not green, the installation is incomplete or the program has
been moved by hand rather than installed. Reinstalling is the proper remedy; as
a stopgap, **Locate GDAL folder** in the sidebar accepts any folder holding
`gdal*.dll`.

## Qt6

[**Qt6**](https://www.qt.io/product/qt6/qml-book/ch17-qtcpp-qtcpp) is the cross-platform application framework Trajecta Studio's
entire graphical interface is built with — every window, button and widget in
the program, including this Guide, is drawn by it. It is a separate, independent
project from GDAL: Qt draws and runs the interface, GDAL reads and writes the
geospatial data behind it. Trajecta bundles the Qt libraries it needs, the same
way it bundles GDAL, so nothing has to be installed separately for either.

# License

Trajecta is free software, distributed under the GNU General Public License,
version 3. The full text below is the license itself, reproduced verbatim.

**GNU GENERAL PUBLIC LICENSE**

Version 3, 29 June 2007

Copyright (C) 2007 Free Software Foundation, Inc. [http://fsf.org/](http://fsf.org/) Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.

## Preamble

The GNU General Public License is a free, copyleft license for software and other kinds of works.

The licenses for most software and other practical works are designed to take away your freedom to share and change the works.  By contrast, the GNU General Public License is intended to guarantee your freedom to share and change all versions of a program--to make sure it remains free software for all its users.  We, the Free Software Foundation, use the GNU General Public License for most of our software; it applies also to any other work released this way by its authors.  You can apply it to your programs, too.

When we speak of free software, we are referring to freedom, not price.  Our General Public Licenses are designed to make sure that you have the freedom to distribute copies of free software (and charge for them if you wish), that you receive source code or can get it if you want it, that you can change the software or use pieces of it in new free programs, and that you know you can do these things.

To protect your rights, we need to prevent others from denying you these rights or asking you to surrender the rights.  Therefore, you have certain responsibilities if you distribute copies of the software, or if you modify it: responsibilities to respect the freedom of others.

For example, if you distribute copies of such a program, whether gratis or for a fee, you must pass on to the recipients the same freedoms that you received.  You must make sure that they, too, receive or can get the source code.  And you must show them these terms so they know their rights.

Developers that use the GNU GPL protect your rights with two steps: (1) assert copyright on the software, and (2) offer you this License giving you legal permission to copy, distribute and/or modify it.

For the developers' and authors' protection, the GPL clearly explains that there is no warranty for this free software.  For both users' and authors' sake, the GPL requires that modified versions be marked as changed, so that their problems will not be attributed erroneously to authors of previous versions.

Some devices are designed to deny users access to install or run modified versions of the software inside them, although the manufacturer can do so.  This is fundamentally incompatible with the aim of protecting users' freedom to change the software.  The systematic pattern of such abuse occurs in the area of products for individuals to use, which is precisely where it is most unacceptable.  Therefore, we have designed this version of the GPL to prohibit the practice for those products.  If such problems arise substantially in other domains, we stand ready to extend this provision to those domains in future versions of the GPL, as needed to protect the freedom of users.

Finally, every program is threatened constantly by software patents. States should not allow patents to restrict development and use of software on general-purpose computers, but in those that do, we wish to avoid the special danger that patents applied to a free program could make it effectively proprietary.  To prevent this, the GPL assures that patents cannot be used to render the program non-free.

The precise terms and conditions for copying, distribution and modification follow.

**TERMS AND CONDITIONS**

## 0. Definitions.

"This License" refers to version 3 of the GNU General Public License.

"Copyright" also means copyright-like laws that apply to other kinds of works, such as semiconductor masks.

"The Program" refers to any copyrightable work licensed under this License.  Each licensee is addressed as "you".  "Licensees" and "recipients" may be individuals or organizations.

To "modify" a work means to copy from or adapt all or part of the work in a fashion requiring copyright permission, other than the making of an exact copy.  The resulting work is called a "modified version" of the earlier work or a work "based on" the earlier work.

A "covered work" means either the unmodified Program or a work based on the Program.

To "propagate" a work means to do anything with it that, without permission, would make you directly or secondarily liable for infringement under applicable copyright law, except executing it on a computer or modifying a private copy.  Propagation includes copying, distribution (with or without modification), making available to the public, and in some countries other activities as well.

To "convey" a work means any kind of propagation that enables other parties to make or receive copies.  Mere interaction with a user through a computer network, with no transfer of a copy, is not conveying.

An interactive user interface displays "Appropriate Legal Notices" to the extent that it includes a convenient and prominently visible feature that (1) displays an appropriate copyright notice, and (2) tells the user that there is no warranty for the work (except to the extent that warranties are provided), that licensees may convey the work under this License, and how to view a copy of this License.  If the interface presents a list of user commands or options, such as a menu, a prominent item in the list meets this criterion.

## 1. Source Code.

The "source code" for a work means the preferred form of the work for making modifications to it.  "Object code" means any non-source form of a work.

A "Standard Interface" means an interface that either is an official standard defined by a recognized standards body, or, in the case of interfaces specified for a particular programming language, one that is widely used among developers working in that language.

The "System Libraries" of an executable work include anything, other than the work as a whole, that (a) is included in the normal form of packaging a Major Component, but which is not part of that Major Component, and (b) serves only to enable use of the work with that Major Component, or to implement a Standard Interface for which an implementation is available to the public in source code form.  A "Major Component", in this context, means a major essential component (kernel, window system, and so on) of the specific operating system (if any) on which the executable work runs, or a compiler used to produce the work, or an object code interpreter used to run it.

The "Corresponding Source" for a work in object code form means all the source code needed to generate, install, and (for an executable work) run the object code and to modify the work, including scripts to control those activities.  However, it does not include the work's System Libraries, or general-purpose tools or generally available free programs which are used unmodified in performing those activities but which are not part of the work.  For example, Corresponding Source includes interface definition files associated with source files for the work, and the source code for shared libraries and dynamically linked subprograms that the work is specifically designed to require, such as by intimate data communication or control flow between those subprograms and other parts of the work.

The Corresponding Source need not include anything that users can regenerate automatically from other parts of the Corresponding Source.

The Corresponding Source for a work in source code form is that same work.

## 2. Basic Permissions.

All rights granted under this License are granted for the term of copyright on the Program, and are irrevocable provided the stated conditions are met.  This License explicitly affirms your unlimited permission to run the unmodified Program.  The output from running a covered work is covered by this License only if the output, given its content, constitutes a covered work.  This License acknowledges your rights of fair use or other equivalent, as provided by copyright law.

You may make, run and propagate covered works that you do not convey, without conditions so long as your license otherwise remains in force.  You may convey covered works to others for the sole purpose of having them make modifications exclusively for you, or provide you with facilities for running those works, provided that you comply with the terms of this License in conveying all material for which you do not control copyright.  Those thus making or running the covered works for you must do so exclusively on your behalf, under your direction and control, on terms that prohibit them from making any copies of your copyrighted material outside their relationship with you.

Conveying under any other circumstances is permitted solely under the conditions stated below.  Sublicensing is not allowed; section 10 makes it unnecessary.

## 3. Protecting Users' Legal Rights From Anti-Circumvention Law.

No covered work shall be deemed part of an effective technological measure under any applicable law fulfilling obligations under article 11 of the WIPO copyright treaty adopted on 20 December 1996, or similar laws prohibiting or restricting circumvention of such measures.

When you convey a covered work, you waive any legal power to forbid circumvention of technological measures to the extent such circumvention is effected by exercising rights under this License with respect to the covered work, and you disclaim any intention to limit operation or modification of the work as a means of enforcing, against the work's users, your or third parties' legal rights to forbid circumvention of technological measures.

## 4. Conveying Verbatim Copies.

You may convey verbatim copies of the Program's source code as you receive it, in any medium, provided that you conspicuously and appropriately publish on each copy an appropriate copyright notice; keep intact all notices stating that this License and any non-permissive terms added in accord with section 7 apply to the code; keep intact all notices of the absence of any warranty; and give all recipients a copy of this License along with the Program.

You may charge any price or no price for each copy that you convey, and you may offer support or warranty protection for a fee.

## 5. Conveying Modified Source Versions.

You may convey a work based on the Program, or the modifications to produce it from the Program, in the form of source code under the terms of section 4, provided that you also meet all of these conditions:

> a) The work must carry prominent notices stating that you modified it, and giving a relevant date.

> b) The work must carry prominent notices stating that it is released under this License and any conditions added under section 7.  This requirement modifies the requirement in section 4 to "keep intact all notices".

> c) You must license the entire work, as a whole, under this License to anyone who comes into possession of a copy.  This License will therefore apply, along with any applicable section 7 additional terms, to the whole of the work, and all its parts, regardless of how they are packaged.  This License gives no permission to license the work in any other way, but it does not invalidate such permission if you have separately received it.

> d) If the work has interactive user interfaces, each must display Appropriate Legal Notices; however, if the Program has interactive interfaces that do not display Appropriate Legal Notices, your work need not make them do so.

A compilation of a covered work with other separate and independent works, which are not by their nature extensions of the covered work, and which are not combined with it such as to form a larger program, in or on a volume of a storage or distribution medium, is called an "aggregate" if the compilation and its resulting copyright are not used to limit the access or legal rights of the compilation's users beyond what the individual works permit.  Inclusion of a covered work in an aggregate does not cause this License to apply to the other parts of the aggregate.

## 6. Conveying Non-Source Forms.

You may convey a covered work in object code form under the terms of sections 4 and 5, provided that you also convey the machine-readable Corresponding Source under the terms of this License, in one of these ways:

> a) Convey the object code in, or embodied in, a physical product (including a physical distribution medium), accompanied by the Corresponding Source fixed on a durable physical medium customarily used for software interchange.

> b) Convey the object code in, or embodied in, a physical product (including a physical distribution medium), accompanied by a written offer, valid for at least three years and valid for as long as you offer spare parts or customer support for that product model, to give anyone who possesses the object code either (1) a copy of the Corresponding Source for all the software in the product that is covered by this License, on a durable physical medium customarily used for software interchange, for a price no more than your reasonable cost of physically performing this conveying of source, or (2) access to copy the Corresponding Source from a network server at no charge.

> c) Convey individual copies of the object code with a copy of the written offer to provide the Corresponding Source.  This alternative is allowed only occasionally and noncommercially, and only if you received the object code with such an offer, in accord with subsection 6b.

> d) Convey the object code by offering access from a designated place (gratis or for a charge), and offer equivalent access to the Corresponding Source in the same way through the same place at no further charge.  You need not require recipients to copy the Corresponding Source along with the object code.  If the place to copy the object code is a network server, the Corresponding Source may be on a different server (operated by you or a third party) that supports equivalent copying facilities, provided you maintain clear directions next to the object code saying where to find the Corresponding Source.  Regardless of what server hosts the Corresponding Source, you remain obligated to ensure that it is available for as long as needed to satisfy these requirements.

> e) Convey the object code using peer-to-peer transmission, provided you inform other peers where the object code and Corresponding Source of the work are being offered to the general public at no charge under subsection 6d.

A separable portion of the object code, whose source code is excluded from the Corresponding Source as a System Library, need not be included in conveying the object code work.

A "User Product" is either (1) a "consumer product", which means any tangible personal property which is normally used for personal, family, or household purposes, or (2) anything designed or sold for incorporation into a dwelling.  In determining whether a product is a consumer product, doubtful cases shall be resolved in favor of coverage.  For a particular product received by a particular user, "normally used" refers to a typical or common use of that class of product, regardless of the status of the particular user or of the way in which the particular user actually uses, or expects or is expected to use, the product.  A product is a consumer product regardless of whether the product has substantial commercial, industrial or non-consumer uses, unless such uses represent the only significant mode of use of the product.

"Installation Information" for a User Product means any methods, procedures, authorization keys, or other information required to install and execute modified versions of a covered work in that User Product from a modified version of its Corresponding Source.  The information must suffice to ensure that the continued functioning of the modified object code is in no case prevented or interfered with solely because modification has been made.

If you convey an object code work under this section in, or with, or specifically for use in, a User Product, and the conveying occurs as part of a transaction in which the right of possession and use of the User Product is transferred to the recipient in perpetuity or for a fixed term (regardless of how the transaction is characterized), the Corresponding Source conveyed under this section must be accompanied by the Installation Information.  But this requirement does not apply if neither you nor any third party retains the ability to install modified object code on the User Product (for example, the work has been installed in ROM).

The requirement to provide Installation Information does not include a requirement to continue to provide support service, warranty, or updates for a work that has been modified or installed by the recipient, or for the User Product in which it has been modified or installed.  Access to a network may be denied when the modification itself materially and adversely affects the operation of the network or violates the rules and protocols for communication across the network.

Corresponding Source conveyed, and Installation Information provided, in accord with this section must be in a format that is publicly documented (and with an implementation available to the public in source code form), and must require no special password or key for unpacking, reading or copying.

## 7. Additional Terms.

"Additional permissions" are terms that supplement the terms of this License by making exceptions from one or more of its conditions. Additional permissions that are applicable to the entire Program shall be treated as though they were included in this License, to the extent that they are valid under applicable law.  If additional permissions apply only to part of the Program, that part may be used separately under those permissions, but the entire Program remains governed by this License without regard to the additional permissions.

When you convey a copy of a covered work, you may at your option remove any additional permissions from that copy, or from any part of it.  (Additional permissions may be written to require their own removal in certain cases when you modify the work.)  You may place additional permissions on material, added by you to a covered work, for which you have or can give appropriate copyright permission.

Notwithstanding any other provision of this License, for material you add to a covered work, you may (if authorized by the copyright holders of that material) supplement the terms of this License with terms:

> a) Disclaiming warranty or limiting liability differently from the terms of sections 15 and 16 of this License; or

> b) Requiring preservation of specified reasonable legal notices or author attributions in that material or in the Appropriate Legal Notices displayed by works containing it; or

> c) Prohibiting misrepresentation of the origin of that material, or requiring that modified versions of such material be marked in reasonable ways as different from the original version; or

> d) Limiting the use for publicity purposes of names of licensors or authors of the material; or

> e) Declining to grant rights under trademark law for use of some trade names, trademarks, or service marks; or

> f) Requiring indemnification of licensors and authors of that material by anyone who conveys the material (or modified versions of it) with contractual assumptions of liability to the recipient, for any liability that these contractual assumptions directly impose on those licensors and authors.

All other non-permissive additional terms are considered "further restrictions" within the meaning of section 10.  If the Program as you received it, or any part of it, contains a notice stating that it is governed by this License along with a term that is a further restriction, you may remove that term.  If a license document contains a further restriction but permits relicensing or conveying under this License, you may add to a covered work material governed by the terms of that license document, provided that the further restriction does not survive such relicensing or conveying.

If you add terms to a covered work in accord with this section, you must place, in the relevant source files, a statement of the additional terms that apply to those files, or a notice indicating where to find the applicable terms.

Additional terms, permissive or non-permissive, may be stated in the form of a separately written license, or stated as exceptions; the above requirements apply either way.

## 8. Termination.

You may not propagate or modify a covered work except as expressly provided under this License.  Any attempt otherwise to propagate or modify it is void, and will automatically terminate your rights under this License (including any patent licenses granted under the third paragraph of section 11).

However, if you cease all violation of this License, then your license from a particular copyright holder is reinstated (a) provisionally, unless and until the copyright holder explicitly and finally terminates your license, and (b) permanently, if the copyright holder fails to notify you of the violation by some reasonable means prior to 60 days after the cessation.

Moreover, your license from a particular copyright holder is reinstated permanently if the copyright holder notifies you of the violation by some reasonable means, this is the first time you have received notice of violation of this License (for any work) from that copyright holder, and you cure the violation prior to 30 days after your receipt of the notice.

Termination of your rights under this section does not terminate the licenses of parties who have received copies or rights from you under this License.  If your rights have been terminated and not permanently reinstated, you do not qualify to receive new licenses for the same material under section 10.

## 9. Acceptance Not Required for Having Copies.

You are not required to accept this License in order to receive or run a copy of the Program.  Ancillary propagation of a covered work occurring solely as a consequence of using peer-to-peer transmission to receive a copy likewise does not require acceptance.  However, nothing other than this License grants you permission to propagate or modify any covered work.  These actions infringe copyright if you do not accept this License.  Therefore, by modifying or propagating a covered work, you indicate your acceptance of this License to do so.

## 10. Automatic Licensing of Downstream Recipients.

Each time you convey a covered work, the recipient automatically receives a license from the original licensors, to run, modify and propagate that work, subject to this License.  You are not responsible for enforcing compliance by third parties with this License.

An "entity transaction" is a transaction transferring control of an organization, or substantially all assets of one, or subdividing an organization, or merging organizations.  If propagation of a covered work results from an entity transaction, each party to that transaction who receives a copy of the work also receives whatever licenses to the work the party's predecessor in interest had or could give under the previous paragraph, plus a right to possession of the Corresponding Source of the work from the predecessor in interest, if the predecessor has it or can get it with reasonable efforts.

You may not impose any further restrictions on the exercise of the rights granted or affirmed under this License.  For example, you may not impose a license fee, royalty, or other charge for exercise of rights granted under this License, and you may not initiate litigation (including a cross-claim or counterclaim in a lawsuit) alleging that any patent claim is infringed by making, using, selling, offering for sale, or importing the Program or any portion of it.

## 11. Patents.

A "contributor" is a copyright holder who authorizes use under this License of the Program or a work on which the Program is based.  The work thus licensed is called the contributor's "contributor version".

A contributor's "essential patent claims" are all patent claims owned or controlled by the contributor, whether already acquired or hereafter acquired, that would be infringed by some manner, permitted by this License, of making, using, or selling its contributor version, but do not include claims that would be infringed only as a consequence of further modification of the contributor version.  For purposes of this definition, "control" includes the right to grant patent sublicenses in a manner consistent with the requirements of this License.

Each contributor grants you a non-exclusive, worldwide, royalty-free patent license under the contributor's essential patent claims, to make, use, sell, offer for sale, import and otherwise run, modify and propagate the contents of its contributor version.

In the following three paragraphs, a "patent license" is any express agreement or commitment, however denominated, not to enforce a patent (such as an express permission to practice a patent or covenant not to sue for patent infringement).  To "grant" such a patent license to a party means to make such an agreement or commitment not to enforce a patent against the party.

If you convey a covered work, knowingly relying on a patent license, and the Corresponding Source of the work is not available for anyone to copy, free of charge and under the terms of this License, through a publicly available network server or other readily accessible means, then you must either (1) cause the Corresponding Source to be so available, or (2) arrange to deprive yourself of the benefit of the patent license for this particular work, or (3) arrange, in a manner consistent with the requirements of this License, to extend the patent license to downstream recipients.  "Knowingly relying" means you have actual knowledge that, but for the patent license, your conveying the covered work in a country, or your recipient's use of the covered work in a country, would infringe one or more identifiable patents in that country that you have reason to believe are valid.

If, pursuant to or in connection with a single transaction or arrangement, you convey, or propagate by procuring conveyance of, a covered work, and grant a patent license to some of the parties receiving the covered work authorizing them to use, propagate, modify or convey a specific copy of the covered work, then the patent license you grant is automatically extended to all recipients of the covered work and works based on it.

A patent license is "discriminatory" if it does not include within the scope of its coverage, prohibits the exercise of, or is conditioned on the non-exercise of one or more of the rights that are specifically granted under this License.  You may not convey a covered work if you are a party to an arrangement with a third party that is in the business of distributing software, under which you make payment to the third party based on the extent of your activity of conveying the work, and under which the third party grants, to any of the parties who would receive the covered work from you, a discriminatory patent license (a) in connection with copies of the covered work conveyed by you (or copies made from those copies), or (b) primarily for and in connection with specific products or compilations that contain the covered work, unless you entered into that arrangement, or that patent license was granted, prior to 28 March 2007.

Nothing in this License shall be construed as excluding or limiting any implied license or other defenses to infringement that may otherwise be available to you under applicable patent law.

## 12. No Surrender of Others' Freedom.

If conditions are imposed on you (whether by court order, agreement or otherwise) that contradict the conditions of this License, they do not excuse you from the conditions of this License.  If you cannot convey a covered work so as to satisfy simultaneously your obligations under this License and any other pertinent obligations, then as a consequence you may not convey it at all.  For example, if you agree to terms that obligate you to collect a royalty for further conveying from those to whom you convey the Program, the only way you could satisfy both those terms and this License would be to refrain entirely from conveying the Program.

## 13. Use with the GNU Affero General Public License.

Notwithstanding any other provision of this License, you have permission to link or combine any covered work with a work licensed under version 3 of the GNU Affero General Public License into a single combined work, and to convey the resulting work.  The terms of this License will continue to apply to the part which is the covered work, but the special requirements of the GNU Affero General Public License, section 13, concerning interaction through a network will apply to the combination as such.

## 14. Revised Versions of this License.

The Free Software Foundation may publish revised and/or new versions of the GNU General Public License from time to time.  Such new versions will be similar in spirit to the present version, but may differ in detail to address new problems or concerns.

Each version is given a distinguishing version number.  If the Program specifies that a certain numbered version of the GNU General Public License "or any later version" applies to it, you have the option of following the terms and conditions either of that numbered version or of any later version published by the Free Software Foundation.  If the Program does not specify a version number of the GNU General Public License, you may choose any version ever published by the Free Software Foundation.

If the Program specifies that a proxy can decide which future versions of the GNU General Public License can be used, that proxy's public statement of acceptance of a version permanently authorizes you to choose that version for the Program.

Later license versions may give you additional or different permissions.  However, no additional obligations are imposed on any author or copyright holder as a result of your choosing to follow a later version.

## 15. Disclaimer of Warranty.

THERE IS NO WARRANTY FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW.  EXCEPT WHEN OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.  THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU.  SHOULD THE PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR OR CORRECTION.

## 16. Limitation of Liability.

IN NO EVENT UNLESS REQUIRED BY APPLICABLE LAW OR AGREED TO IN WRITING WILL ANY COPYRIGHT HOLDER, OR ANY OTHER PARTY WHO MODIFIES AND/OR CONVEYS THE PROGRAM AS PERMITTED ABOVE, BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY GENERAL, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OR INABILITY TO USE THE PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH ANY OTHER PROGRAMS), EVEN IF SUCH HOLDER OR OTHER PARTY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

## 17. Interpretation of Sections 15 and 16.

If the disclaimer of warranty and limitation of liability provided above cannot be given local legal effect according to their terms, reviewing courts shall apply local law that most closely approximates an absolute waiver of all civil liability in connection with the Program, unless a warranty or assumption of liability accompanies a copy of the Program in return for a fee.

**END OF TERMS AND CONDITIONS**

## How to Apply These Terms to Your New Programs

If you develop a new program, and you want it to be of the greatest possible use to the public, the best way to achieve this is to make it free software which everyone can redistribute and change under these terms.

To do so, attach the following notices to the program.  It is safest to attach them to the start of each source file to most effectively state the exclusion of warranty; and each file should have at least the "copyright" line and a pointer to where the full notice is found.

> <one line to give the program's name and a brief idea of what it does.>
> 
> 
>     Copyright (C) <year>  <name of author>
> 
> 
> 
> 
> 
>     This program is free software: you can redistribute it and/or modify
> 
> 
>     it under the terms of the GNU General Public License as published by
> 
> 
>     the Free Software Foundation, either version 3 of the License, or
> 
> 
>     (at your option) any later version.
> 
> 
> 
> 
> 
>     This program is distributed in the hope that it will be useful,
> 
> 
>     but WITHOUT ANY WARRANTY; without even the implied warranty of
> 
> 
>     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
> 
> 
>     GNU General Public License for more details.
> 
> 
> 
> 
> 
>     You should have received a copy of the GNU General Public License
> 
> 
>     along with this program.  If not, see <http://www.gnu.org/licenses/>.

Also add information on how to contact you by electronic and paper mail.

If the program does terminal interaction, make it output a short notice like this when it starts in an interactive mode:

> <program>  Copyright (C) <year>  <name of author>
> 
> 
>     This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
> 
> 
>     This is free software, and you are welcome to redistribute it
> 
> 
>     under certain conditions; type `show c' for details.

The hypothetical commands `show w' and `show c' should show the appropriate parts of the General Public License.  Of course, your program's commands might be different; for a GUI interface, you would use an "about box".

You should also get your employer (if you work as a programmer) or school, if any, to sign a "copyright disclaimer" for the program, if necessary. For more information on this, and how to apply and follow the GNU GPL, see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/).

The GNU General Public License does not permit incorporating your program into proprietary programs.  If your program is a subroutine library, you may consider it more useful to permit linking proprietary applications with the library.  If this is what you want to do, use the GNU Lesser General Public License instead of this License.  But first, please read [http://www.gnu.org/philosophy/why-not-lgpl.html](http://www.gnu.org/philosophy/why-not-lgpl.html).

# References

Baddeley, A., Rubak, E., & Turner, R. (2015). *Spatial Point Patterns:
Methodology and Applications with R*. Chapman & Hall/CRC.

Besag, J., & Diggle, P. J. (1977). Simple Monte Carlo tests for spatial
pattern. *Journal of the Royal Statistical Society: Series C (Applied
Statistics)*, 26(3), 327–333.
[doi:10.2307/2346974](https://doi.org/10.2307/2346974)

Bresenham, J. E. (1965). Algorithm for computer control of a digital plotter.
*IBM Systems Journal*, 4(1), 25–30.
[doi:10.1147/sj.41.0025](https://doi.org/10.1147/sj.41.0025)

Campbell, M. J., Dennison, P. E., Butler, B. W., & Page, W. G. (2019).
Using crowdsourced fitness tracker data to model the relationship between slope
and travel rates. *Applied Geography*, 106, 93–107.
[doi:10.1016/j.apgeog.2019.03.008](https://doi.org/10.1016/j.apgeog.2019.03.008)

Dijkstra, E. W. (1959). A note on two problems in connexion with graphs.
*Numerische Mathematik*, 1, 269–271.
[doi:10.1007/BF01386390](https://doi.org/10.1007/BF01386390)

Felzenszwalb, P. F., & Huttenlocher, D. P. (2012). Distance transforms of
sampled functions. *Theory of Computing*, 8(19), 415–428.
[doi:10.4086/toc.2012.v008a019](https://doi.org/10.4086/toc.2012.v008a019)

Hausdorff, F. (1914). *Grundzüge der Mengenlehre*. Veit & Comp.

Herzog, I. (2013). The potential and limits of Optimal Path Analysis. In
A. Bevan & M. Lake (eds.), *Computational Approaches to Archaeological
Spaces* (pp. 179–211). Left Coast Press.

Herzog, I. (2014). A review of case studies in archaeological least-cost
analysis. *Archeologia e Calcolatori*, 25, 223–239.

Hope, A. C. A. (1968). A simplified Monte Carlo significance test procedure.
*Journal of the Royal Statistical Society: Series B (Methodological)*,
30(3), 582–598.
[doi:10.1111/j.2517-6161.1968.tb00759.x](https://doi.org/10.1111/j.2517-6161.1968.tb00759.x)

Irmischer, I. J., & Clarke, K. C. (2017). Measuring and modeling the
speed of human navigation. *Cartography and Geographic Information Science*,
45(2), 177–186.
[doi:10.1080/15230406.2017.1292150](https://doi.org/10.1080/15230406.2017.1292150)

Liang, Y.-D., & Barsky, B. A. (1984). A new concept and method for line
clipping. *ACM Transactions on Graphics*, 3(1), 1–22.
[doi:10.1145/357332.357333](https://doi.org/10.1145/357332.357333)

Lotwick, H. W., & Silverman, B. W. (1982). Methods for analysing spatial
processes of several types of points. *Journal of the Royal Statistical
Society: Series B (Methodological)*, 44(3), 406–413.
[doi:10.1111/j.2517-6161.1982.tb01221.x](https://doi.org/10.1111/j.2517-6161.1982.tb01221.x)

Márquez-Pérez, J., Vallejo-Villalta, I., &
Álvarez-Francoso, J. I. (2017). Estimated travel time for walking trails
in natural areas. *Geografisk Tidsskrift–Danish Journal of Geography*,
117(1), 53–62.
[doi:10.1080/00167223.2017.1316212](https://doi.org/10.1080/00167223.2017.1316212)

Minetti, A. E., Moia, C., Roi, G. S., Susta, D., & Ferretti, G. (2002).
Energy cost of walking and running at extreme uphill and downhill slopes.
*Journal of Applied Physiology*, 93(3), 1039–1046.
[doi:10.1152/japplphysiol.01177.2001](https://doi.org/10.1152/japplphysiol.01177.2001)

Otsu, N. (1979). A threshold selection method from gray-level histograms.
*IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
[doi:10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076)

Park, S. W., Linsen, L., Kreylos, O., Owens, J. D., & Hamann, B. (2006).
Discrete Sibson interpolation. *IEEE Transactions on Visualization and
Computer Graphics*, 12(2), 243–253.
[doi:10.1109/TVCG.2006.27](https://doi.org/10.1109/TVCG.2006.27)

Sibson, R. (1981). A brief description of natural neighbour interpolation. In
V. Barnett (ed.), *Interpreting Multivariate Data* (pp. 21–36). Wiley.

Tobler, W. (1993). *Three presentations on geographical analysis and
modeling*. National Center for Geographic Information and Analysis,
Technical Report 93-1.

White, D. A. (2015). The Basics of Least Cost Analysis for Archaeological
Applications. *Advances in Archaeological Practice*, 3(4), 407–414.
[doi:10.7183/2326-3768.3.4.407](https://doi.org/10.7183/2326-3768.3.4.407)

White, D. A., & Barber, S. B. (2012). Geospatial modeling of pedestrian
transportation networks: A case study from precolumbian Oaxaca, Mexico.
*Journal of Archaeological Science*, 39(8), 2684–2696.
[doi:10.1016/j.jas.2012.04.017](https://doi.org/10.1016/j.jas.2012.04.017)

# Trajecta

Trajecta is a user-friendly least-cost analysis software specifically developed to be used in Digital Humanities and by users with only a basic computer science background. **Be patient, this software is currently under development and can contain bugs or errors**. Please, contact me for bug reporting, problems during the installation or additional features you would like to see developed and included in future releases. 

## Available Versions

### Current Release (Installer)
> [!IMPORTANT]
> - [v1.0.0](https://github.com/ArcheoHacker1501/trajecta/releases/tag/v0.1.1) --> **With full GUI!**

### Source Code
- v1.0.0 (improved algorithm, full GUI)
- v0.1.1 (Improved algorithm)
- v0.1.0 (Initial release)

## Overview

Currently, Trajecta provides two complementary workflows for movement modeling:
- **FETE** (From-Everywhere-To-Everywhere): accessibility and path density from many sources ([D. A. White and S. B. Barber 2012](https://www.sciencedirect.com/science/article/pii/S0305440312001379)).
- **LCPA** (Least-Cost Path Analysis): optimal routes from one origin to multiple destinations ([D. A. White 2015](https://www.cambridge.org/core/journals/advances-in-archaeological-practice/article/basics-of-least-cost-analysis-for-archaeological-applications/DE502C37794C0E200AE7FA6A7529E25E?utm_campaign=shareaholic&utm_medium=copy_link&utm_source=bookmark)).

Both modes use anisotropic cost functions and support cost surface modifiers from polyline shapefiles. Read the instructions for more information on currently implemented functionalities. 

## Expected I/O
Below you can find some samples of the results obtainable using Trajecta. You can read the `USER_GUIDE.md` for additional information and quick start unsing Trajecta.

### FETE (From-Everywhere-To-Everywhere)
The FETE algorithm implemented by Trajecta allows to calculate a high number of least-cost paths connecting every point to every other point of a regular or randomly scattered point grid. To compute LCPs, DEM or other elevation based data can be used to calculate slope which can then be transformed using cost functions (e.g. Tobler's Hiking Function). Additional costs can be added as for waterbodies or terrain indexes. 

Based on all the LCPs calculated, overlap analysis is then computed and a density raster is created. Color gradients can be used to sispaly most probable paths among all calculated LCPs. 

|![](https://github.com/user-attachments/assets/32d90698-3ecb-41f0-ba7e-44d931dec1d6)|![](https://github.com/user-attachments/assets/5931b184-6342-48fe-a284-82ae6dc1ff2a)|
|:-:|:-:|
|Example of regular point grid and SRTM 30m DEM used as input for FETE computation. Point density shown in the image was decreased for improved visibility (~20% of the real density; total number of points of full-size sample ~15000 points).|Unfiltered FETE raster resulting from computation with Trajecta.|

|![](https://github.com/user-attachments/assets/5ea66d9a-10aa-430b-81ea-90468a1de6f0)|
|:-:|
|Filtered FETE raster using only top 20% results.|

### LCPA (Least-Cost Path Analysis)
Trajecta allows high-speed computation of Least-Cost Path Analysis between a single origin and one or more destinations. To compute LCPs, DEM or other elevation based data can be used to calculate slope which can then be transformed using cost functions (e.g. Tobler's Hiking Function). Additional costs can be added as for waterbodies or terrain indexes. 

|![LCPA](https://github.com/user-attachments/assets/d177db40-1ef9-47ab-947d-e2b75d67fb68)|
|:-:|
|Least-Cost Paths from single origin to multiple destinations calculated using Trajecta and SRTM 30m DEM.|

## Currently supported Platform

- **Windows 10/11**: Supported (CPU and GPU with CUDA)
- **Linux**: Experimental (CPU and GPU with CUDA). Some Windows-specific code paths still need portability updates.
- **macOS**: CURRENTLY NOT SUPPORTED

## Requirements

- CMake 3.24+
- C++17 compiler
- GDAL 3.x (external dependency)

Windows uses **OSGeo4W** for GDAL. Linux uses system packages. macOS is not supported yet.

## Release (Maintainers)

- Windows: `scripts/release.ps1`
- Linux: `scripts/release.sh`

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
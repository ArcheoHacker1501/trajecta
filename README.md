# Trajecta

Trajecta is a user-friendly least-cost analysis tool specifically developed to be used in Digital Humanities and by users with only a basic computer science background. Please, contact me for bug reporting, problems during the installation or additional features you would like to see developed and included in future releases. 

## Available Versions

### Release (Installer)
- v1.0.0 - now with **Trajecta Studio UI** - NEW!
- v0.1.1
- v0.1.0

### Source Code
- v1.0.0 - NEW!
- v0.1.1
- v0.1.0

## Overview

Currently, Trajecta provides two complementary workflows for movement modeling:
- **FETE** (From-Everywhere-To-Everywhere): accessibility and path density from many sources ([D. A. White and S. B. Barber 2012](https://www.sciencedirect.com/science/article/pii/S0305440312001379)).
- **LCPA** (Least-Cost Path Analysis): optimal routes from one origin to multiple destinations ([D. A. White 2015](https://www.cambridge.org/core/journals/advances-in-archaeological-practice/article/basics-of-least-cost-analysis-for-archaeological-applications/DE502C37794C0E200AE7FA6A7529E25E?utm_campaign=shareaholic&utm_medium=copy_link&utm_source=bookmark)).

Both modes use anisotropic cost functions and support cost surface modifiers in both raster and vector form.

### Graphical Interface — Trajecta Studio

Trajecta ships with **Trajecta Studio**, a graphical interface installed automatically by the installer. It exposes every parameter of the console program through a validated form, shows live progress and results, and finds the GDAL/OSGeo4W libraries automatically — no PATH configuration required.

## Currently supported Platform

- **Windows 10/11**: Supported
- **Linux**: Experimental. Some Windows-specific code paths still need portability updates.

## Quick Start

Follow instructions in `USER_FRIENDLY_INSTALL.md` for a user-firendly installation of Trajecta through the Trajecta installer (RECOMMENDED FOR MOST USERS).

## Documentation

- `INSTALL_GUIDE.md` - User-firendly installation instructions
- `CONTRIBUTING.md` - Contributing guidelines
- `THIRD_PARTY_NOTICES.md` - Third-party licenses

## Release (Maintainers)

- Windows: `scripts/release.ps1`
- Linux: `scripts/release.sh`

## Citation

If you use Trajecta in your research, please cite:

```
Stefano Aprà - Institute for the Study of the Ancient World (NYU)
```

## License

GPL-3.0. See `LICENSE` for details.

## Acknowledgments

- GDAL for geospatial data I/O


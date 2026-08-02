# Third-Party Notices

Trajecta is distributed under the **GNU General Public License v3.0** (see
`LICENSE`).

Starting with version 1.0.0 the Windows package is **self-contained**: it ships
the GDAL runtime, its transitive dependencies, the Qt libraries and the
Microsoft C/C++ runtime, so no separate GDAL installation is required. The
components below are therefore redistributed inside the installer, each under
its own licence.

This list is generated from the actual contents of the package. Every component
keeps its own copyright and licence text; where a project ships one, that file
is the authoritative statement and takes precedence over the summary here.

---

## Geospatial core

| Component | Files | Licence |
|---|---|---|
| **GDAL** | `gdal312.dll`, `share/gdal/` | MIT |
| **PROJ** | `proj_9.dll`, `share/proj/` (incl. `proj.db`) | MIT |
| **GEOS** | `geos.dll`, `geos_c.dll` | LGPL-2.1-or-later |
| **SpatiaLite** | `spatialite.dll` | MPL-1.1 / GPL-2.0+ / LGPL-2.1+ (tri-licence) |
| **FreeXL** | `freexl.dll` | MPL-1.1 / GPL-2.0+ / LGPL-2.1+ (tri-licence) |
| **netCDF** | `netcdf.dll` | MIT-style (UCAR/Unidata) |
| **HDF4** | `hdf.dll`, `mfhdf.dll` | BSD-style (The HDF Group) |
| **HDF5** | `hdf5.dll`, `hdf5_hl.dll` | BSD-3-Clause (The HDF Group) |
| **Szip / libaec** | `szip.dll` | BSD-2-Clause (libaec) |

## Raster and compression codecs

| Component | Files | Licence |
|---|---|---|
| **libtiff** | `tiff.dll` | libtiff (BSD-style) |
| **libjpeg-turbo** | `jpeg62.dll` | IJG / BSD-3-Clause |
| **libpng** | `libpng16.dll` | PNG Reference Library License v2 |
| **OpenJPEG** | `openjp2.dll` | BSD-2-Clause |
| **libwebp** | `libwebp.dll`, `libsharpyuv.dll` | BSD-3-Clause |
| **JPEG XL** | `jxl.dll`, `jxl_cms.dll`, `jxl_threads.dll` | BSD-3-Clause |
| **LERC** | `Lerc.dll` | Apache-2.0 |
| **zlib** | `zlib.dll` | zlib |
| **Zstandard** | `zstd.dll` | BSD-3-Clause / GPL-2.0 (dual) |
| **LZ4** | `lz4.dll` | BSD-2-Clause |
| **XZ Utils / liblzma** | `liblzma.dll` | 0BSD (public-domain equivalent) |
| **Blosc** | `blosc.dll` | BSD-3-Clause |
| **Brotli** | `brotlicommon.dll`, `brotlidec.dll`, `brotlienc.dll` | MIT |
| **libarchive** | `archive.dll` | BSD-2-Clause |

## Data, text and network

| Component | Files | Licence |
|---|---|---|
| **SQLite** | `sqlite3.dll` | Public domain |
| **Expat** | `libexpat.dll` | MIT |
| **libxml2** | `libxml2.dll` | MIT |
| **Xerces-C++** | `xerces-c_3_2.dll` | Apache-2.0 |
| **libiconv** | `iconv-2.dll` | LGPL-2.1-or-later |
| **libcurl** | `libcurl.dll` | curl licence (MIT/X derivative) |
| **OpenSSL 3.x** | `libssl-3-x64.dll`, `libcrypto-3-x64.dll` | Apache-2.0 |
| **FreeType** | `freetype.dll` | FTL or GPL-2.0+ (dual) |
| **Apache Arrow / Parquet** | `arrow*.dll`, `parquet.dll`, `thriftmd.dll` | Apache-2.0 |
| **PostgreSQL client (libpq)** | `libpq.dll` | PostgreSQL Licence (BSD-style) |
| **MySQL client library** | `libmysql.dll` | **GPL-2.0 with FOSS License Exception** — see note |
| **Poppler** | `poppler.dll` | GPL-2.0-or-later |

## Application runtime

| Component | Files | Licence |
|---|---|---|
| **Qt 6** | `Qt6Core/Gui/Widgets/Network/Svg.dll` + plugin folders | LGPL-3.0 |
| **MinGW GCC runtime** | `libgcc_s_seh-1.dll`, `libstdc++-6.dll` | GPL-3.0 **with the GCC Runtime Library Exception** |
| **winpthreads** | `libwinpthread-1.dll` | MIT / BSD (mingw-w64) |
| **Microsoft C/C++ runtime** | `MSVCP140.dll`, `VCRUNTIME140.dll`, `VCRUNTIME140_1.dll`, `VCOMP140.DLL` | Microsoft *Distributable Code* terms |

---

## Notes on specific components

### MySQL client library

`libmysql.dll` is present because GDAL 3.12, as built for OSGeo4W/QGIS, imports
it **unconditionally** rather than through delayed loading. Trajecta never opens
a MySQL data source, but the library cannot be removed from the package without
`gdal312.dll` failing to load — this was verified by removing it and observing
the engine fail to start.

It is Oracle's MySQL client, licensed GPL-2.0 with the **FOSS License
Exception**, which permits linking from software released under a list of free
licences that includes GPL-3.0, the licence of Trajecta. Anyone redistributing a
modified package should read that exception rather than rely on this summary.

Building GDAL from source with a reduced driver set would drop this component
(along with Poppler, Arrow, HDF and netCDF) and shrink the package
considerably. That is the cleaner long-term option.

### Copyleft components

Poppler is GPL-2.0-**or-later** and therefore combinable with Trajecta's
GPL-3.0. GEOS, libiconv, SpatiaLite and FreeXL are LGPL: they are linked
dynamically and shipped as separate DLLs, which is what the LGPL requires so a
user can replace them.

### Qt

Qt 6 is used under the **LGPL-3.0** open-source terms. It is linked dynamically
and shipped as separate libraries, so it can be replaced. No Qt source has been
modified.

### Microsoft runtime

The Visual C++ runtime libraries are redistributed under Microsoft's
Distributable Code terms, which permit shipping them with an application. They
are copied unmodified from the system.

---

## Source code

Trajecta's own source is at <https://github.com/ArcheoHacker1501/trajecta>.

For the GPL- and LGPL-licensed components above, the corresponding source is
available from each project's own website. The binaries shipped here are the
unmodified upstream builds distributed by the OSGeo4W/QGIS project; no patches
have been applied by Trajecta.

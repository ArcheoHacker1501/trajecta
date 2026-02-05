# Release Checklist

Use this list before publishing a new Trajecta release.

## Source and Versioning
- [ ] Update version in `CMakeLists.txt` (`CPACK_PACKAGE_VERSION`)
- [ ] Update README/INSTALL/USAGE if behavior or requirements changed
- [ ] Verify repository license and `THIRD_PARTY_NOTICES.md` are accurate
- [ ] Ensure Gunrock submodule is at the intended commit

## Build and Smoke Test (Windows)
- [ ] Configure:
  - `cmake .. -DGDAL_ROOT="C:/OSGeo4W64" -DCMAKE_CUDA_ARCHITECTURES=<arch>`
- [ ] Build: `cmake --build . --config Release`
- [ ] If using automatic DLL copy, confirm DLLs exist in `build\Release`
- [ ] Run: `build\Release\trajecta.exe` and complete a small test

## Build and Smoke Test (Linux)
- [ ] Configure: `cmake .. -DCMAKE_CUDA_ARCHITECTURES=<arch>`
- [ ] Build: `cmake --build .`
- [ ] Run: `./trajecta` and complete a small test

## Packaging
- [ ] Windows installer: `cpack -G NSIS`
- [ ] Verify installer runs and launches Trajecta
- [ ] Confirm installer includes docs (`README.md`, `INSTALL.md`, `USAGE.md`)
- [ ] Generate SHA256 checksums for release artifacts

## GitHub Release
- [ ] Create tag (e.g., `v0.1.0`)
- [ ] Push tag
- [ ] Publish GitHub release with:
  - Windows NSIS installer
  - Source code snapshot
  - Release notes (changes, known issues)

## Post-Release
- [ ] Update any documentation links if necessary
- [ ] Confirm downloads work from GitHub

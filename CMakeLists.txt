cmake_minimum_required(VERSION 3.24)
project(trajecta LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TRAJECTA_COPY_GDAL_RUNTIME "Copy GDAL runtime DLLs next to the executable (Windows)" ON)

# OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
else()
  message(WARNING "OpenMP not found - CPU performance may be reduced")
endif()

# GDAL
if (WIN32)
  set(GDAL_ROOT "C:/OSGeo4W64" CACHE PATH "Path to OSGeo4W installation")
  set(GDAL_INCLUDE_DIR "${GDAL_ROOT}/include")
  set(GDAL_LIBRARY "${GDAL_ROOT}/lib/gdal_i.lib")

  if (NOT EXISTS "${GDAL_INCLUDE_DIR}")
    message(FATAL_ERROR "GDAL include not found: ${GDAL_INCLUDE_DIR}")
  endif()
  if (NOT EXISTS "${GDAL_LIBRARY}")
    message(FATAL_ERROR "GDAL library not found: ${GDAL_LIBRARY}")
  endif()

  set(GDAL_INCLUDE_DIRS "${GDAL_INCLUDE_DIR}")
  set(GDAL_LIBRARIES "${GDAL_LIBRARY}")
else()
  find_package(GDAL REQUIRED)
endif()

message(STATUS "GDAL Include: ${GDAL_INCLUDE_DIRS}")
message(STATUS "GDAL Library: ${GDAL_LIBRARIES}")

# Executable
add_executable(trajecta
  src/main_fete.cpp
  src/main_lcpa.cpp
  src/main_interp.cpp
)

# Includes
target_include_directories(trajecta PRIVATE
  ${GDAL_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(trajecta PRIVATE ${GDAL_LIBRARIES})
if (OpenMP_CXX_FOUND)
  target_link_libraries(trajecta PRIVATE OpenMP::OpenMP_CXX)
endif()

# Compile options
if (MSVC)
  target_compile_options(trajecta PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/openmp>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
  )
endif()

# NOMINMAX prevents windows.h from defining min/max macros that break STL templates
# TRAJECTA_VERSION is stamped into every run manifest, so it is defined here —
# next to CPACK_PACKAGE_VERSION below — rather than hard-coded in the sources.
target_compile_definitions(trajecta PRIVATE
  NOMINMAX
  TRAJECTA_VERSION="1.0.1"
)

# Optional: copy GDAL runtime next to the executable (Windows)
if (WIN32 AND TRAJECTA_COPY_GDAL_RUNTIME)
  add_custom_command(TARGET trajecta POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Copying GDAL runtime from ${GDAL_ROOT}/bin"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${GDAL_ROOT}/bin" "$<TARGET_FILE_DIR:trajecta>"
  )
endif()

# Install rules
install(TARGETS trajecta RUNTIME DESTINATION .)
install(FILES README.md LICENSE THIRD_PARTY_NOTICES.md DESTINATION .)

# ---------------------------------------------------------------------------
# Self-contained GDAL runtime
#
# trajecta.exe imports gdal312.dll by name, so "install GDAL" was not a
# sufficient instruction: it had to be *that* version. Anyone with, say, QGIS
# 3.34 (which ships gdal308.dll) got a missing-DLL error despite following the
# guide. The runtime is therefore shipped with the application, which also
# removes the C++ runtime dependency and pins the PROJ database to a known-good
# copy instead of whichever one happens to be first on PATH.
#
# The dependency closure is resolved by CMake rather than listed by hand: a
# hardcoded list silently rots the first time GDAL changes what it links
# against, and this one is ~50 entries deep.
# ---------------------------------------------------------------------------
option(TRAJECTA_BUNDLE_GDAL "Ship the GDAL runtime inside the package" ON)

# Where to take the runtime from. QGIS carries a current proj.db; OSGeo4W64's
# own copy is years older than its DLLs and makes PROJ fail to identify a CRS.
set(TRAJECTA_RUNTIME_DIR "C:/Program Files/QGIS 3.44.10" CACHE PATH
    "Installation providing gdal312.dll, its dependencies and share/{proj,gdal}")

if (WIN32 AND TRAJECTA_BUNDLE_GDAL)
  if (NOT EXISTS "${TRAJECTA_RUNTIME_DIR}/bin/gdal312.dll")
    message(FATAL_ERROR
      "TRAJECTA_RUNTIME_DIR does not contain bin/gdal312.dll: ${TRAJECTA_RUNTIME_DIR}\n"
      "Point it at a GDAL 3.12 installation, or configure with "
      "-DTRAJECTA_BUNDLE_GDAL=OFF to keep depending on an external GDAL.")
  endif()
  if (NOT EXISTS "${TRAJECTA_RUNTIME_DIR}/share/proj/proj.db")
    message(FATAL_ERROR
      "TRAJECTA_RUNTIME_DIR has no share/proj/proj.db: ${TRAJECTA_RUNTIME_DIR}")
  endif()

  install(CODE "
    set(_bin \"${TRAJECTA_RUNTIME_DIR}/bin\")
    file(GET_RUNTIME_DEPENDENCIES
      EXECUTABLES \"\$<TARGET_FILE:trajecta>\"
      RESOLVED_DEPENDENCIES_VAR _resolved
      UNRESOLVED_DEPENDENCIES_VAR _unresolved
      DIRECTORIES \"\${_bin}\"
      # Everything under System32 belongs to Windows and must not be copied;
      # api-ms-win-* are the Universal CRT, present since Windows 10.
      PRE_EXCLUDE_REGEXES \"^api-ms-\" \"^ext-ms-\"
      POST_EXCLUDE_REGEXES \"[Ss]ystem32\"
    )
    list(LENGTH _resolved _n)
    message(STATUS \"Bundling \${_n} runtime libraries from \${_bin}\")
    foreach(_lib IN LISTS _resolved)
      file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}\"
           TYPE SHARED_LIBRARY FOLLOW_SYMLINK_CHAIN FILES \"\${_lib}\")
    endforeach()
    # The MSVC redistributable is excluded above because it lives in System32;
    # Microsoft's Distributable Code terms allow shipping it with the app, and
    # doing so is what makes the package work on a machine without it.
    foreach(_rt MSVCP140.dll VCRUNTIME140.dll VCRUNTIME140_1.dll VCOMP140.DLL)
      if (EXISTS \"C:/Windows/System32/\${_rt}\")
        file(INSTALL DESTINATION \"\${CMAKE_INSTALL_PREFIX}\"
             TYPE SHARED_LIBRARY FILES \"C:/Windows/System32/\${_rt}\")
      endif()
    endforeach()
    if (_unresolved)
      message(WARNING \"Unresolved runtime dependencies: \${_unresolved}\")
    endif()
  ")

  # Data directories. PROJ without proj.db cannot identify a CRS; the engine
  # points at these itself (use_bundled_gdal_data in main_fete.cpp).
  #
  # The layout differs between distributions: OSGeo4W keeps GDAL's data under
  # share/gdal, QGIS under apps/gdal/share/gdal. Probe rather than assume.
  set(_gdal_data "")
  foreach(_cand "share/gdal" "apps/gdal/share/gdal")
    if (EXISTS "${TRAJECTA_RUNTIME_DIR}/${_cand}/gdalicon.png"
        OR EXISTS "${TRAJECTA_RUNTIME_DIR}/${_cand}/gdalvrt.xsd")
      set(_gdal_data "${TRAJECTA_RUNTIME_DIR}/${_cand}")
      break()
    endif()
  endforeach()
  if (NOT _gdal_data)
    message(FATAL_ERROR
      "Could not find GDAL's data directory under ${TRAJECTA_RUNTIME_DIR} "
      "(looked in share/gdal and apps/gdal/share/gdal).")
  endif()
  message(STATUS "GDAL data: ${_gdal_data}")

  install(DIRECTORY "${TRAJECTA_RUNTIME_DIR}/share/proj/" DESTINATION share/proj)
  install(DIRECTORY "${_gdal_data}/" DESTINATION share/gdal)
endif()

# Optional: bundle a pre-built, windeployqt-deployed Trajecta Studio folder
# (the graphical interface). scripts/release.ps1 builds the GUI and passes
# this path automatically; see gui/TrajectaStudio/README.md.
set(TRAJECTA_GUI_DIR "" CACHE PATH
    "Deployed Trajecta Studio folder to bundle into the package")
if (WIN32 AND TRAJECTA_GUI_DIR)
  if (NOT EXISTS "${TRAJECTA_GUI_DIR}/TrajectaStudio.exe")
    message(FATAL_ERROR "TRAJECTA_GUI_DIR does not contain TrajectaStudio.exe: ${TRAJECTA_GUI_DIR}")
  endif()
  install(DIRECTORY "${TRAJECTA_GUI_DIR}/" DESTINATION .)
endif()

# CPack (NSIS on Windows, TGZ/ZIP on other platforms)
set(CPACK_PACKAGE_NAME "Trajecta")
set(CPACK_PACKAGE_VENDOR "Trajecta")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Least-cost path analysis for spatial movement modeling")
set(CPACK_PACKAGE_VERSION "1.0.1")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")

# Platform goes in the installer's file name (not just its contents) so that
# Windows and macOS builds sitting side by side on the same GitHub release —
# or the same Downloads folder — never collide and are told apart at a
# glance, without opening either one.
if (WIN32)
  set(CPACK_PACKAGE_FILE_NAME "Trajecta-${CPACK_PACKAGE_VERSION}_Windows")
elseif (APPLE)
  set(CPACK_PACKAGE_FILE_NAME "Trajecta-${CPACK_PACKAGE_VERSION}_macOS")
else()
  set(CPACK_PACKAGE_FILE_NAME "Trajecta-${CPACK_PACKAGE_VERSION}_Linux")
endif()

if (WIN32)
  set(CPACK_GENERATOR "NSIS")
  # No version in the Start Menu group or the Add/Remove entry, and none in the
  # install directory: anything that carries a version number in its *name* goes
  # stale the moment the next release lands next to it. That is what left a
  # "Trajecta 1.0.0" group in the Start Menu holding a 1.0.1 shortcut — NSIS
  # takes this string as the default group, and installing 1.0.1 over 1.0.0
  # without uninstalling first simply added to the old folder rather than
  # replacing it.
  #
  # The version is still visible where it cannot age: on the shortcut label
  # (CPACK_PACKAGE_EXECUTABLES below) and in Add/Remove Programs' own Version
  # column, which CPack fills from CPACK_PACKAGE_VERSION.
  set(CPACK_NSIS_DISPLAY_NAME "Trajecta")
  set(CPACK_NSIS_PACKAGE_NAME "Trajecta")
  set(CPACK_PACKAGE_INSTALL_DIRECTORY "Trajecta")
  set(CPACK_NSIS_MODIFY_PATH OFF)
  set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/gui/TrajectaStudio/assets/trajecta.ico")
  set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/gui/TrajectaStudio/assets/trajecta.ico")
  if (TRAJECTA_GUI_DIR)
    # Start Menu and Desktop shortcuts for the graphical interface, and offer
    # to launch it when the installer finishes. Executables live in the
    # install root (not the default "bin"), hence EXECUTABLES_DIRECTORY.
    # Label under the desktop and Start Menu icons.
    set(CPACK_PACKAGE_EXECUTABLES
        "TrajectaStudio" "Trajecta Studio ${CPACK_PACKAGE_VERSION}")
    set(CPACK_CREATE_DESKTOP_LINKS "TrajectaStudio")
    set(CPACK_NSIS_EXECUTABLES_DIRECTORY ".")
    set(CPACK_NSIS_INSTALLED_ICON_NAME "TrajectaStudio.exe")

    # "Run Trajecta Studio" on the finish page, launched through Explorer so it
    # does NOT inherit the installer's elevated token — which is what killed
    # drag & drop on the first run after every install. CPACK_NSIS_MUI_FINISHPAGE_RUN
    # is deliberately left unset: the replacement lives in the file below,
    # because it is the only place the escaping survives. See it for why.
    set(CPACK_PROJECT_CONFIG_FILE "${CMAKE_SOURCE_DIR}/cmake/CPackNSISFinish.cmake")
  endif()
else()
  set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)

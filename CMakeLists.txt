cmake_minimum_required(VERSION 3.24)
project(trajecta LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TRAJECTA_FETCH_GUNROCK "Download Gunrock if not found" ON)
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

# Gunrock
set(GUNROCK_ROOT "${CMAKE_SOURCE_DIR}/external/gunrock" CACHE PATH "Path to Gunrock installation")
if (EXISTS "${GUNROCK_ROOT}/include")
  message(STATUS "Gunrock found: ${GUNROCK_ROOT}")
  set(GUNROCK_INCLUDE_DIR "${GUNROCK_ROOT}/include")
else()
  if (TRAJECTA_FETCH_GUNROCK)
    include(FetchContent)
    FetchContent_Declare(
      gunrock
      GIT_REPOSITORY https://github.com/gunrock/gunrock.git
      GIT_TAG v1.0.0
    )
    FetchContent_MakeAvailable(gunrock)
    set(GUNROCK_INCLUDE_DIR "${gunrock_SOURCE_DIR}/include")
    message(STATUS "Gunrock downloaded: ${gunrock_SOURCE_DIR}")
  else()
    message(WARNING "Gunrock not found at ${GUNROCK_ROOT} - GPU functionality will be limited")
  endif()
endif()

# Executable
add_executable(trajecta
  src/main_fete.cu
  src/main_lcpa.cu
  src/main_fete_gpu.cu
)

# Includes
target_include_directories(trajecta PRIVATE
  include
  ${GDAL_INCLUDE_DIRS}
)

if (GUNROCK_INCLUDE_DIR)
  target_include_directories(trajecta PRIVATE ${GUNROCK_INCLUDE_DIR})
endif()

# Link libraries
target_link_libraries(trajecta PRIVATE ${GDAL_LIBRARIES})
if (OpenMP_CXX_FOUND)
  target_link_libraries(trajecta PRIVATE OpenMP::OpenMP_CXX)
endif()

# CUDA compile options
if (MSVC)
  target_compile_options(trajecta PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/openmp,/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/openmp>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
  )
else()
  target_compile_options(trajecta PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endif()

# Default CUDA architecture
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Definitions
target_compile_definitions(trajecta PRIVATE
  __HIP_PLATFORM_NVIDIA__
  ESSENTIALS_NVIDIA_BACKEND=1
  SM_TARGET=${CMAKE_CUDA_ARCHITECTURES}
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
install(FILES README.md INSTALL.md USAGE.md THIRD_PARTY_NOTICES.md CONTRIBUTING.md DESTINATION .)

# CPack (NSIS on Windows, TGZ/ZIP on other platforms)
set(CPACK_PACKAGE_NAME "Trajecta")
set(CPACK_PACKAGE_VENDOR "Trajecta")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU-accelerated least-cost path analysis")
set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_PACKAGE_FILE_NAME "Trajecta-${CPACK_PACKAGE_VERSION}")

if (WIN32)
  set(CPACK_GENERATOR "NSIS")
  set(CPACK_NSIS_DISPLAY_NAME "Trajecta")
  set(CPACK_NSIS_PACKAGE_NAME "Trajecta")
  set(CPACK_NSIS_MODIFY_PATH OFF)
else()
  set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)

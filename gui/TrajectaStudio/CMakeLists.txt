cmake_minimum_required(VERSION 3.19)

project(TrajectaStudio VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Widgets Network)
qt_standard_project_setup()

qt_add_executable(TrajectaStudio
    WIN32
    main.cpp
    mainwindow.cpp
    mainwindow.h
    confirmdialog.cpp
    confirmdialog.h
    trajectarunner.cpp
    trajectarunner.h
    consoleview.cpp
    consoleview.h
    pathpicker.cpp
    pathpicker.h
    gdalapi.cpp
    gdalapi.h
    rangeslider.cpp
    rangeslider.h
    smoothcombobox.cpp
    smoothcombobox.h
    thememanager.cpp
    thememanager.h
    viewerpage.cpp
    viewerpage.h
    systeminfo.h
    resources.qrc
    app.rc
)

target_compile_definitions(TrajectaStudio PRIVATE
    TRAJECTA_STUDIO_VERSION="${PROJECT_VERSION}"
)

target_link_libraries(TrajectaStudio PRIVATE Qt6::Widgets Qt6::Network)
if (WIN32)
    target_link_libraries(TrajectaStudio PRIVATE dwmapi)
endif()

# Headless harness that exercises the TrajectaRunner state machine against a
# real trajecta.exe. For development/testing only.
option(TRAJECTA_STUDIO_BUILD_TESTTOOL "Build the runner_probe test tool" OFF)
if (TRAJECTA_STUDIO_BUILD_TESTTOOL)
    qt_add_executable(runner_probe
        runner_probe.cpp
        trajectarunner.cpp
        trajectarunner.h
    )
    target_link_libraries(runner_probe PRIVATE Qt6::Core)
endif()

install(TARGETS TrajectaStudio RUNTIME DESTINATION .)

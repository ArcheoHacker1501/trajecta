cmake_minimum_required(VERSION 3.19)

project(TrajectaStudio VERSION 1.0.1 LANGUAGES CXX)

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
    framelessdialog.cpp
    framelessdialog.h
    advancedsettingsdialog.cpp
    advancedsettingsdialog.h
    trajectarunner.cpp
    trajectarunner.h
    batchmodel.cpp
    batchmodel.h
    batchcontroller.cpp
    batchcontroller.h
    batchtable.cpp
    batchtable.h
    batchpage.cpp
    batchpage.h
    postbatchmodel.cpp
    postbatchmodel.h
    postbatchcontroller.cpp
    postbatchcontroller.h
    postbatchpage.cpp
    postbatchpage.h
    checkpointstore.cpp
    checkpointstore.h
    coherence.cpp
    coherence.h
    consoleview.cpp
    consoleview.h
    pathpicker.cpp
    pathpicker.h
    gdalapi.cpp
    gdalapi.h
    largepages.cpp
    largepages.h
    rangeslider.cpp
    rangeslider.h
    reportform.cpp
    reportform.h
    routecompare.cpp
    routecompare.h
    smoothcombobox.cpp
    smoothcombobox.h
    walkthrough.cpp
    walkthrough.h
    thememanager.cpp
    thememanager.h
    uiwidgets.cpp
    uiwidgets.h
    viewerpage.cpp
    viewerpage.h
    systeminfo.h
    resources.qrc
    app.rc
)

target_compile_definitions(TrajectaStudio PRIVATE
    TRAJECTA_STUDIO_VERSION="${PROJECT_VERSION}"
)

# src/checkpoint.h defines the on-disk layout of a checkpoint. The interface
# reads what the engine wrote, so it includes that header rather than keeping a
# second copy of the struct in step with the first.
target_include_directories(TrajectaStudio PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

target_link_libraries(TrajectaStudio PRIVATE Qt6::Widgets Qt6::Network)
if (WIN32)
    # dwmapi: the rounded corners and the dark title bar of the frameless
    # window. advapi32: LsaAddAccountRights, which is how the large-pages
    # privilege is granted (see largepages.cpp).
    target_link_libraries(TrajectaStudio PRIVATE dwmapi advapi32)
endif()

# Headless harness that exercises the TrajectaRunner state machine against a
# real trajecta.exe. For development/testing only.
option(TRAJECTA_STUDIO_BUILD_TESTTOOL "Build the runner_probe test tool" OFF)
if (TRAJECTA_STUDIO_BUILD_TESTTOOL)
    qt_add_executable(runner_probe
        runner_probe.cpp
        trajectarunner.cpp
        trajectarunner.h
        batchmodel.cpp
        batchmodel.h
        batchcontroller.cpp
        batchcontroller.h
        checkpointstore.cpp
        checkpointstore.h
    )
    target_link_libraries(runner_probe PRIVATE Qt6::Core)
    target_include_directories(runner_probe PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src
    )
endif()

install(TARGETS TrajectaStudio RUNTIME DESTINATION .)

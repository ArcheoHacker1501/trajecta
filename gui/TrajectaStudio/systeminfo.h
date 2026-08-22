#pragma once

#include <QtGlobal>

#ifdef Q_OS_WIN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace SystemInfo {

inline qint64 totalRamMb()
{
#ifdef Q_OS_WIN
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        return qint64(status.ullTotalPhys / (1024 * 1024));
#endif
    return 8192;  // conservative fallback
}

// What the engine is actually offered, on any machine — never a share of what
// is installed, which is what earlier versions did and which reserved tens of
// gigabytes that were never touched.
//
// It is a floor, not a forecast. A FETE keeps one set of working buffers per
// thread (23 bytes per cell each, plus 17 shared), so what a run really needs
// grows with the DEM *and* with the thread count: at 16 threads this ceiling
// covers a DEM of roughly 4700 x 4700 cells. Above that the engine does not
// fail — it fits the thread count to the budget and says so — but raising this
// value is what gets all the threads working again.
constexpr int kRecommendedRamMb = 8192;

} // namespace SystemInfo

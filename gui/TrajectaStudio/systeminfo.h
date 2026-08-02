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

} // namespace SystemInfo

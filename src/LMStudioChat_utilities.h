









#ifndef LMSTUDIO_CHAT_UTILITIES_H
#define LMSTUDIO_CHAT_UTILITIES_H

#include <string>
#include <fmt/core.h>
#include "Log.h"

// Safe formatting utility for the LMStudio Chat module.
// This will catch all fmt::format errors and log them.
template<typename... Args>
inline std::string SafeFormat(const std::string& templ, Args&&... args) {
    try {
        return fmt::format(templ, std::forward<Args>(args)...);
    } catch (const fmt::format_error& e) {
        LOG_ERROR("server.loading", "[LMStudio Chat] Format error: {} | Template: {}", e.what(), templ);
        return "[Format Error]";
    }
}

#endif // LMSTUDIO_CHAT_UTILITIES_H










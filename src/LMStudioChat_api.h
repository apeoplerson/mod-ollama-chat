





#ifndef LMSTUDIO_CHAT_API_H
#define LMSTUDIO_CHAT_API_H

#include <string>
#include <future>
#include "LMStudioChat_querymanager.h"

std::string QueryLMStudioAPI(const std::string& prompt);

// Submits a query to the API.
std::future<std::string> SubmitQuery(const std::string& prompt);

// Declare the global QueryManager variable.
extern QueryManager g_queryManager;

#endif // LMSTUDIO_CHAT_API_H






#include "LMStudioChat_api.h"
#include "LMStudioChat_config.h"
#include "LMStudioChat_httpclient.h"
#include "Log.h"
#include <sstream>
#include <nlohmann/json.hpp>
#include <fmt/core.h>
#include <thread>
#include <mutex>
#include <queue>
#include <future>

std::string ExtractTextBetweenDoubleQuotes(const std::string& response)
{
    size_t first = response.find('"');
    size_t second = response.find('"', first + 1);
    if (first != std::string::npos && second != std::string::npos) {
        return response.substr(first + 1, second - first - 1);
    }
    return response;
}

// Function to perform the API call.
std::string QueryLMStudioAPI(const std::string& prompt)
{
    // Initialize our custom HTTP client
    static LMStudioHttpClient httpClient;
    
    if (!httpClient.IsAvailable())
    {
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[LMStudio Chat] HTTP client not available.");
        }
        return "Hmm... I'm lost in thought.";
    }

    std::string url   = g_LMStudioUrl;
    std::string model = g_LMStudioModel;

    nlohmann::json requestData = {
        {"model",  model},
        {"prompt", prompt},
        {"stream", false}
    };

    // Create options object for model parameters
    nlohmann::json options;
    bool hasOptions = false;

    // Only include if set (do not send defaults if user did not set them)
    if (g_LMStudioMaxTokens > 0) {
        options["num_predict"] = g_LMStudioMaxTokens;
        hasOptions = true;
    }
    if (g_LMStudioTemperature != 0.8f) {
        options["temperature"] = g_LMStudioTemperature;
        hasOptions = true;
    }
    if (g_LMStudioTopP != 0.95f) {
        options["top_p"] = g_LMStudioTopP;
        hasOptions = true;
    }
    if (g_LMStudioRepeatPenalty != 1.1f) {
        options["repeat_penalty"] = g_LMStudioRepeatPenalty;
        hasOptions = true;
    }
    if (g_LMStudioContextSize > 0) {
        options["num_ctx"] = g_LMStudioContextSize;
        hasOptions = true;
    }
    if (g_LMStudioNumThreads > 0) {
        options["num_thread"] = g_LMStudioNumThreads;
        hasOptions = true;
        if(g_DebugEnabled) {
            //LOG_INFO("server.loading", "[LMStudio Chat] Setting num_thread to: {}", g_LMStudioNumThreads);
        }
    } else if(g_DebugEnabled) {
        //LOG_INFO("server.loading", "[LMStudio Chat] g_LMStudioNumThreads is: {} (not sending num_thread)", g_LMStudioNumThreads);
    }
    if (!g_LMStudioSeed.empty()) {
        try {
            int seedValue = std::stoi(g_LMStudioSeed);
            options["seed"] = seedValue; 
            hasOptions = true;
        } catch (const std::exception& e) {
            if(g_DebugEnabled) {
                LOG_INFO("server.loading", "[LMStudio Chat] Invalid seed value: {}", g_LMStudioSeed);
            }
        }
    }

    // Add options object if any options were set
    if (hasOptions) {
        requestData["options"] = options;
    }

    // Root-level parameters (these stay at root level)
    if (!g_LMStudioStop.empty()) {
        // If comma-separated, convert to array
        std::vector<std::string> stopSeqs;
        std::stringstream ss(g_LMStudioStop);
        std::string item;
        while (std::getline(ss, item, ',')) {
            // trim whitespace
            size_t start = item.find_first_not_of(" \t");
            size_t end = item.find_last_not_of(" \t");
            if (start != std::string::npos && end != std::string::npos)
                stopSeqs.push_back(item.substr(start, end - start + 1));
        }
        if (!stopSeqs.empty())
            requestData["stop"] = stopSeqs;
    }
    if (!g_LMStudioSystemPrompt.empty())   requestData["system"]          = g_LMStudioSystemPrompt;

    if (g_ThinkModeEnableForModule)
    {
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[LMStudio Chat] LLM set to Think mode.");
        }
        requestData["think"] = true;
        requestData["hidethinking"] = true;
    }

    std::string requestDataStr = requestData.dump();

    // Make HTTP POST request using our custom client
    std::string responseBuffer = httpClient.Post(url, requestDataStr);

    if (responseBuffer.empty())
    {
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[LMStudio Chat] Failed to reach LMStudio API.");
        }
        return "Failed to reach LMStudio API.";
    }

    std::stringstream ss(responseBuffer);
    std::string line;
    std::ostringstream extractedResponse;

    try
    {
        while (std::getline(ss, line))
        {
            if (line.empty() || std::all_of(line.begin(), line.end(), isspace))
                continue;

            nlohmann::json jsonResponse = nlohmann::json::parse(line);

            if (jsonResponse.contains("response") && !jsonResponse["response"].get<std::string>().empty())
            {
                extractedResponse << jsonResponse["response"].get<std::string>();
            }
        }
    }
    catch (const std::exception& e)
    {
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading",
                    "[LMStudio Chat] JSON Parsing Error: {}",
                    e.what());
        }
        return "Error processing response.";
    }

    std::string botReply = extractedResponse.str();

    botReply = ExtractTextBetweenDoubleQuotes(botReply);

    if (botReply.empty())
    {
        if(g_DebugEnabled)
        {
            LOG_INFO("server.loading", "[LMStudio Chat] No valid response extracted.");
        }
        return "I'm having trouble understanding.";
    }

    if(g_DebugEnabled)
    {
        LOG_INFO("server.loading", "[LMStudio Chat] Parsed bot response: {}", botReply);

        if (g_ThinkModeEnableForModule)
        {
            if(g_DebugEnabled)
            {
                LOG_INFO("server.loading", "[LMStudio Chat] Bot used think.");
            }
        }
    }

    return botReply;
}

QueryManager g_queryManager;

// Interface function to submit a query.
std::future<std::string> SubmitQuery(const std::string& prompt)
{
    return g_queryManager.submitQuery(prompt);
}
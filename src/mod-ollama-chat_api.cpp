#include "mod-ollama-chat_api.h"
#include "mod-ollama-chat_config.h"
#include "Log.h"
#include <curl/curl.h>
#include <sstream>
#include <nlohmann/json.hpp>
#include <fmt/core.h>
#include <thread>
#include <mutex>
#include <queue>
#include <future>

// Callback for cURL write function.
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
    std::string* responseBuffer = static_cast<std::string*>(userp);
    size_t totalSize = size * nmemb;
    responseBuffer->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// Function to handle OpenRouter.ai API errors based on HTTP status codes
void HandleOpenRouterErrors(long response_code, const std::string& response_body)
{
    switch (response_code) {
        case 400:
            throw std::runtime_error("Bad Request: Invalid parameters - " + response_body);
        case 401:
            throw std::runtime_error("Unauthorized: Invalid API key - " + response_body);
        case 402:
            throw std::runtime_error("Payment Required: Account issue - " + response_body);
        case 429:
            throw std::runtime_error("Rate Limited: Too many requests - " + response_body);
        default:
            if (response_code >= 400) {
                throw std::runtime_error("HTTP Error " + std::to_string(response_code) + " - " + response_body);
            }
    }
}

// Function to construct OpenRouter.ai request format
nlohmann::json ConstructOpenRouterRequest(const std::string& prompt)
{
    nlohmann::json request;
    request["model"] = g_OpenRouterModel;
    request["messages"] = nlohmann::json::array();
    
    // Add system message if system prompt is configured
    if (!g_OpenRouterSystemPrompt.empty()) {
        request["messages"].push_back({
            {"role", "system"},
            {"content", g_OpenRouterSystemPrompt}
        });
    }
    
    // Add user message
    request["messages"].push_back({
        {"role", "user"},
        {"content", prompt}
    });
    
    // Add optional parameters only if they differ from defaults
    if (g_OpenRouterTemperature != 0.7f) {
        request["temperature"] = g_OpenRouterTemperature;
    }
    if (g_OpenRouterTopP != 0.9f) {
        request["top_p"] = g_OpenRouterTopP;
    }
    if (g_OpenRouterTopK > 0) {
        request["top_k"] = g_OpenRouterTopK;
    }
    if (g_OpenRouterMaxTokens > 0) {
        request["max_tokens"] = g_OpenRouterMaxTokens;
    }
    if (!g_OpenRouterSeed.empty()) {
        try {
            int seed_value = std::stoi(g_OpenRouterSeed);
            request["seed"] = seed_value;
        } catch (const std::exception& e) {
            if (g_DebugEnabled) {
                LOG_INFO("server.loading", "Invalid seed value, ignoring: {}", g_OpenRouterSeed);
            }
        }
    }
    
    // Always set stream to false for this implementation
    request["stream"] = false;
    
    return request;
}

// Function to parse OpenRouter.ai response format
std::string ParseOpenRouterResponse(const std::string& response_json)
{
    try {
        nlohmann::json response = nlohmann::json::parse(response_json);
        
        // Check for error in response
        if (response.contains("error")) {
            std::string error_msg = "API Error";
            if (response["error"].contains("message")) {
                error_msg = response["error"]["message"].get<std::string>();
            }
            throw std::runtime_error("OpenRouter API Error: " + error_msg);
        }
        
        // Extract content from choices array
        if (response.contains("choices") && !response["choices"].empty()) {
            auto& choice = response["choices"][0];
            if (choice.contains("message") && choice["message"].contains("content")) {
                return choice["message"]["content"].get<std::string>();
            }
        }
        
        throw std::runtime_error("Invalid response format: missing choices or content");
        
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("Failed to parse JSON response: " + std::string(e.what()));
    }
}

// Updated function to perform the OpenRouter.ai API call
std::string QueryOllamaAPI(const std::string& prompt)
{
    // Check if API key is configured
    if (g_OpenRouterApiKey.empty()) {
        if (g_DebugEnabled) {
            LOG_INFO("server.loading", "OpenRouter API key not configured.");
        }
        return "AI service not properly configured.";
    }
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        if (g_DebugEnabled) {
            LOG_INFO("server.loading", "Failed to initialize cURL.");
        }
        return "Hmm... I'm lost in thought.";
    }

    std::string url = g_OpenRouterUrl;
    
    // Construct request in OpenRouter.ai format
    nlohmann::json requestData;
    try {
        requestData = ConstructOpenRouterRequest(prompt);
    } catch (const std::exception& e) {
        if (g_DebugEnabled) {
            LOG_INFO("server.loading", "Failed to construct request: {}", e.what());
        }
        curl_easy_cleanup(curl);
        return "Error preparing request.";
    }
    
    std::string requestDataStr = requestData.dump();

    // Set up headers with authentication
    struct curl_slist* headers = nullptr;
    std::string auth_header = "Authorization: Bearer " + g_OpenRouterApiKey;
    headers = curl_slist_append(headers, auth_header.c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    // Optional headers for better tracking
    if (!g_OpenRouterSiteUrl.empty()) {
        std::string referer_header = "HTTP-Referer: " + g_OpenRouterSiteUrl;
        headers = curl_slist_append(headers, referer_header.c_str());
    }
    if (!g_OpenRouterSiteName.empty()) {
        std::string title_header = "X-Title: " + g_OpenRouterSiteName;
        headers = curl_slist_append(headers, title_header.c_str());
    }

    std::string responseBuffer;
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, requestDataStr.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, long(requestDataStr.length()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseBuffer);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    
    // Set timeout to prevent hanging
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);

    CURLcode res = curl_easy_perform(curl);
    
    // Get HTTP response code
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        if (g_DebugEnabled) {
            LOG_INFO("server.loading",
                    "Failed to reach OpenRouter AI. cURL error: {}",
                    curl_easy_strerror(res));
        }
        return "Failed to reach OpenRouter AI.";
    }

    // Handle HTTP errors
    try {
        HandleOpenRouterErrors(response_code, responseBuffer);
    } catch (const std::exception& e) {
        if (g_DebugEnabled) {
            LOG_INFO("server.loading", "OpenRouter API Error: {}", e.what());
        }
        return "AI service error occurred.";
    }

    // Parse the response
    std::string botReply;
    try {
        botReply = ParseOpenRouterResponse(responseBuffer);
    } catch (const std::exception& e) {
        if (g_DebugEnabled) {
            LOG_INFO("server.loading", "Response parsing error: {}", e.what());
        }
        return "Error processing response.";
    }

    if (botReply.empty()) {
        if (g_DebugEnabled) {
            LOG_INFO("server.loading", "No valid response extracted.");
        }
        return "I'm having trouble understanding.";
    }

    if (g_DebugEnabled) {
        LOG_INFO("server.loading", "Parsed bot response: {}", botReply);
    }
    
    return botReply;
}

QueryManager g_queryManager;

// Interface function to submit a query.
std::future<std::string> SubmitQuery(const std::string& prompt)
{
    return g_queryManager.submitQuery(prompt);
}

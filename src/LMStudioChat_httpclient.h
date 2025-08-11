#ifndef LMSTUDIO_HTTP_CLIENT_H
#define LMSTUDIO_HTTP_CLIENT_H

#include <string>

class LMStudioHttpClient
{
public:
    LMStudioHttpClient();
    ~LMStudioHttpClient();

    // Make HTTP POST request to LMStudio API
    std::string Post(const std::string& url, const std::string& jsonData);
    
    // Set timeout for requests (in seconds)
    void SetTimeout(int seconds);
    
    // Check if HTTP client is available
    bool IsAvailable() const;

private:
    int m_timeout;
    bool m_available;
};

#endif // LMSTUDIO_HTTP_CLIENT_H
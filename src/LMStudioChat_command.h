



#ifndef LMSTUDIO_CHAT_COMMAND_H
#define LMSTUDIO_CHAT_COMMAND_H

#include "ScriptMgr.h"
#include "Chat.h"

class LMStudioChatConfigCommand : public CommandScript
{
public:
    LMStudioChatConfigCommand();
    Acore::ChatCommands::ChatCommandTable GetCommands() const override;

    static bool HandleLMStudioReloadCommand(ChatHandler* handler);
    static bool HandleLMStudioSentimentViewCommand(ChatHandler* handler, Optional<std::string> botName, Optional<std::string> playerName);
    static bool HandleLMStudioSentimentSetCommand(ChatHandler* handler, std::string botName, std::string playerName, float sentimentValue);
    static bool HandleLMStudioSentimentResetCommand(ChatHandler* handler, Optional<std::string> botName, Optional<std::string> playerName);
};

#endif // LMSTUDIO_CHAT_COMMAND_H




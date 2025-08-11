





#ifndef LMSTUDIO_CHAT_HANDLER_H
#define LMSTUDIO_CHAT_HANDLER_H

#include "ScriptMgr.h"
#include <string>

enum ChatChannelSourceLocal
{
    SRC_UNDEFINED_LOCAL  = 0,
    SRC_SAY_LOCAL        = 1,
    SRC_PARTY_LOCAL      = 2,
    SRC_RAID_LOCAL       = 3,
    SRC_GUILD_LOCAL      = 4,
    SRC_OFFICER_LOCAL    = 5,
    SRC_YELL_LOCAL       = 6,
    SRC_WHISPER_LOCAL    = 7,
    SRC_GENERAL_LOCAL    = 17
};

extern const char* ChatChannelSourceLocalStr[];

std::string rtrim(const std::string& s);
ChatChannelSourceLocal GetChannelSourceLocal(uint32_t type);

void SaveBotConversationHistoryToDB();

class PlayerBotChatHandler : public PlayerScript
{
public:
    PlayerBotChatHandler() : PlayerScript("PlayerBotChatHandler", {
        { "PLAYER_CHAT", OnPlayerChat, true }
    }) {}

private:
    void OnPlayerChat(Player* player, uint32_t type, bool isLocal, std::string& msg);
};

#endif // LMSTUDIO_CHAT_HANDLER_H






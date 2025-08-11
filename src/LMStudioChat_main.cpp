#include "LMStudioChat_config.h"
#include "LMStudioChat_handler.h"
#include "LMStudioChat_random.h"
#include "LMStudioChat_events.h"
#include "LMStudioChat_command.h"
#include "Log.h"

void AddLMStudioChatScripts()
{
    LOG_INFO("server.loading", "[LMStudio Chat] Registering LMStudio Chat scripts.");
    new LMStudioChatConfigWorldScript();
    new PlayerBotChatHandler();
    new LMStudioBotRandomChatter();

    LOG_INFO("server.loading", "[LMStudio Chat] Registering LMStudio Chat events.");
    new ChatOnKill();
    new ChatOnLoot();
    new ChatOnDeath();
    new ChatOnQuest();
    new ChatOnLearn();
    new ChatOnDuel();
    new ChatOnLevelUp();
    new ChatOnAchievement();
    new ChatOnGameObjectUse();
    new LMStudioChatConfigCommand();
}










#ifndef LMSTUDIO_CHAT_EVENTS_H
#define LMSTUDIO_CHAT_EVENTS_H

#include "ScriptMgr.h"
#include "Player.h"
#include <string>

class LMStudioBotEventChatter
{
public:
    void DispatchGameEvent(Player* source, std::string type, std::string detail);
    void QueueEvent(Player* bot, std::string type, std::string detail, std::string actorName, bool isGuildEvent = false);
    std::string BuildPrompt(Player* bot, std::string promptTemplate, std::string eventType, std::string eventDetail, std::string actorName);
};

class ChatOnKill : public PlayerScript
{
public:
    ChatOnKill();
    void OnPlayerCreatureKill(Player* killer, Creature* victim);
    void OnPlayerPVPKill(Player* killer, Player* killed);
    void OnPlayerCreatureKilledByPet(Player* owner, Creature* victim);
};

class ChatOnLoot : public PlayerScript
{
public:
    ChatOnLoot();
    void OnPlayerStoreNewItem(Player* player, Item* item, uint32 count);
};

#endif // LMSTUDIO_CHAT_EVENTS_H









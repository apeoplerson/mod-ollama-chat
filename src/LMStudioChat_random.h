




#ifndef LMSTUDIO_CHAT_RANDOM_H
#define LMSTUDIO_CHAT_RANDOM_H

#include "ScriptMgr.h"

class LMStudioBotRandomChatter : public WorldScript
{
public:
    LMStudioBotRandomChatter();
    void OnUpdate(uint32 diff) override;

private:
    void HandleRandomChatter();
};

#endif // LMSTUDIO_CHAT_RANDOM_H





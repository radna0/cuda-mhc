
import json
import os
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    RenderConversationConfig,
)

def test():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    config = RenderConversationConfig(auto_drop_analysis=False)
    
    system_text = "You are ChatGPT."
    msg_system = Message.from_role_and_content(Role.SYSTEM, system_text)
    
    file_path = "/dev/shm/nemotron_raw/data/high.part_00.jsonl"
    with open(file_path, 'r') as f:
        for i in range(5):
            line = f.readline()
            if not line: break
            item = json.loads(line)
            msgs_raw = item.get("messages", [])
            print(f"Item {i} message count: {len(msgs_raw)}")
            
            problem = msgs_raw[0].get("content", "")
            reasoning = msgs_raw[1].get("reasoning_content", "")
            answer = msgs_raw[1].get("content", "")
            
            msg_user = Message.from_role_and_content(Role.USER, problem)
            msg_analysis = Message.from_role_and_content(Role.ASSISTANT, reasoning).with_channel("analysis")
            msg_final = Message.from_role_and_content(Role.ASSISTANT, answer).with_channel("final")
            
            msgs = [msg_system, msg_user, msg_analysis, msg_final]
            convo = Conversation.from_messages(msgs)
            tokens = encoding.render_conversation_for_training(convo, config=config)
            print(f"Item {i} tokens length: {len(tokens)}")

if __name__ == "__main__":
    test()

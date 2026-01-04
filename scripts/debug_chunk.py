
import json
import os
import sys

# Mocking the imports if necessary or just importing for real
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    RenderConversationConfig,
)

def debug_process_chunk(file_path, start_offset, end_offset, part_id, out_dir):
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    config = RenderConversationConfig(auto_drop_analysis=False)
    
    system_text = "test system"
    msg_system = Message.from_role_and_content(Role.SYSTEM, system_text)
    
    developer_text = "test developer"
    msg_developer = Message.from_role_and_content(Role.DEVELOPER, developer_text)

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"part_{part_id}.jsonl")
    
    print(f"Opening {file_path}")
    with open(file_path, 'rb') as f_in, open(out_file, 'w') as f_out:
        f_in.seek(start_offset)
        print(f"Seeked to {start_offset}")
        
        line_bytes = f_in.readline()
        print(f"Read first line, length: {len(line_bytes)}")
        if not line_bytes:
            print("No line bytes read!")
            return

        try:
            line = line_bytes.decode('utf-8')
            item = json.loads(line)
            print("JSON loaded")
            msgs_raw = item.get("messages", [])
            print(f"Messages count: {len(msgs_raw)}")
            
            problem = msgs_raw[0].get("content", "")
            reasoning = msgs_raw[1].get("reasoning_content", "")
            answer = msgs_raw[1].get("content", "")
            
            print(f"Problem len: {len(problem)}, Reasoning len: {len(reasoning)}, Answer len: {len(answer)}")

            msg_user = Message.from_role_and_content(Role.USER, problem)
            msgs = [msg_system, msg_developer, msg_user]
            
            if reasoning and reasoning.strip():
                msgs.append(Message.from_role_and_content(Role.ASSISTANT, reasoning.strip()).with_channel("analysis"))
            
            msgs.append(Message.from_role_and_content(Role.ASSISTANT, answer.strip()).with_channel("final"))
            
            print("Building conversation")
            convo = Conversation.from_messages(msgs)
            print("Rendering")
            tokens = encoding.render_conversation_for_training(convo, config=config)
            print(f"Tokens: {len(tokens)}")
            
            f_out.write(json.dumps({"text": encoding.decode(tokens)}) + "\n")
            print("Wrote to output")
        except Exception as e:
            print(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_process_chunk("/dev/shm/nemotron_raw/data/high.part_00.jsonl", 0, 1000000, 888, "/dev/shm/debug_out")


import os
import sys
import multiprocessing
import json
import time
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    ReasoningEffort,
    RenderConversationConfig,
)

LOCAL_DATA_FILE = "/dev/shm/nemotron_raw/data/high.part_00.jsonl"

def process_item_simple(line):
    if not hasattr(process_item_simple, "encoding"):
        process_item_simple.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        process_item_simple.config = RenderConversationConfig(auto_drop_analysis=False)
        process_item_simple.msg_sys = Message.from_role_and_content(
            Role.SYSTEM, 
            SystemContent.new().with_reasoning_effort(ReasoningEffort.HIGH)
        )
    
    encoding = process_item_simple.encoding
    config = process_item_simple.config
    msg_sys = process_item_simple.msg_sys

    try:
        item = json.loads(line)
        problem = item.get("problem", "")
        trajectories = item.get("reasoningTrajectories", [])
        reasoning = ""
        answer = ""
        
        if trajectories and isinstance(trajectories, list) and len(trajectories) > 0:
            traj = trajectories[0]
            reasoning = traj.get("process", traj.get("reasoning", ""))
            answer = traj.get("outcome", traj.get("answer", ""))
        else:
            reasoning = item.get("reasoning", item.get("solution", ""))
            answer = item.get("answer", item.get("response", ""))

        if not problem or not answer:
            return None

        msg_user = Message.from_role_and_content(Role.USER, problem)
        msgs = [msg_sys, msg_user]
        
        if reasoning and reasoning.strip():
            msg_analysis = Message.from_role_and_content(
                Role.ASSISTANT, 
                reasoning.strip()
            ).with_channel("analysis")
            msgs.append(msg_analysis)
            
        msg_final = Message.from_role_and_content(
            Role.ASSISTANT,
            answer.strip()
        ).with_channel("final")
        msgs.append(msg_final)
        
        convo = Conversation.from_messages(msgs)
        tokens = encoding.render_conversation_for_training(convo, config=config)
        return encoding.decode(tokens)
        
    except Exception as e:
        return None

def main():
    print("Reading 1000 lines...")
    lines = []
    with open(LOCAL_DATA_FILE, "r") as f:
        for _ in range(1000):
            line = f.readline()
            if not line: break
            lines.append(line)
    
    print(f"Testing with {len(lines)} lines and various pool sizes...")
    
    for num_workers in [1, 4, 16, 64]:
        start_time = time.time()
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(process_item_simple, lines)
        end_time = time.time()
        
        valid_count = len([r for r in results if r])
        print(f"Workers: {num_workers:2d} | Time: {end_time - start_time:6.2f}s | Speed: {len(lines)/(end_time - start_time):8.2f} items/s | Valid: {valid_count}")

if __name__ == "__main__":
    main()

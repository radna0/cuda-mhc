
import json
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

def test_single_item():
    file_path = "/dev/shm/nemotron_raw/data/high.part_00.jsonl"
    with open(file_path, "r") as f:
        line = f.readline()
    
    item = json.loads(line)
    print("--- RAW ITEM KEYS ---")
    print(item.keys())
    
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    config = RenderConversationConfig(auto_drop_analysis=False)
    msg_sys = Message.from_role_and_content(
        Role.SYSTEM, 
        SystemContent.new().with_reasoning_effort(ReasoningEffort.HIGH)
    )

    problem = item.get("problem", "")
    trajectories = item.get("reasoningTrajectories", [])
    
    print(f"Problem length: {len(problem)}")
    print(f"Trajectories type: {type(trajectories)}")
    
    reasoning = ""
    answer = ""
    
    if trajectories and isinstance(trajectories, list) and len(trajectories) > 0:
        traj = trajectories[0]
        print(f"Traj keys: {traj.keys()}")
        # Check for Nemotron-Math-v2 schema
        # Based on previous research: problem, reasoning, answer are common.
        # But 'reasoningTrajectories' has 'process' and 'outcome' in some Nemotron versions.
        reasoning = traj.get("process", traj.get("reasoning", ""))
        answer = traj.get("outcome", traj.get("answer", ""))
    else:
        reasoning = item.get("reasoning", item.get("solution", ""))
        answer = item.get("answer", item.get("response", ""))

    print(f"Reasoning length: {len(reasoning)}")
    print(f"Answer length: {len(answer)}")

    if not problem or not answer:
        print("FAILED: Missing problem or answer")
        return

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
    decoded = encoding.decode(tokens)
    print("--- FORMATTED PREVIEW ---")
    print(decoded[:500] + "...")
    print("--- END ---")

if __name__ == "__main__":
    test_single_item()

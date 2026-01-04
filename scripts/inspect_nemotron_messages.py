
import json

def inspect_messages():
    file_path = "/dev/shm/nemotron_raw/data/high.part_00.jsonl"
    with open(file_path, "r") as f:
        line = f.readline()
    
    item = json.loads(line)
    messages = item.get("messages", [])
    
    print(f"--- MESSAGES (Count: {len(messages)}) ---")
    for i, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        print(f"[{i}] Role: {role}")
        print(f"    Content Preview: {content[:100]}...")
        if reasoning:
            print(f"    Reasoning Preview: {reasoning[:100]}...")
    
    print("\n--- EXPECTED ANSWER ---")
    print(item.get("expected_answer"))

if __name__ == "__main__":
    inspect_messages()

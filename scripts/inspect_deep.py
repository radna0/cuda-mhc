from datasets import load_dataset
import json

print("Deep Inspection of nvidia/Nemotron-Math-v2 (high_part01)...")
ds = load_dataset("nvidia/Nemotron-Math-v2", split="high_part01", streaming=True)

print("\n--------------------------------------------------")
for i, item in enumerate(ds):
    if i >= 5: break
    
    print(f"\n[SAMPLE {i}]")
    msgs = item.get("messages", [])
    print(f"Total Messages: {len(msgs)}")
    
    has_tool_calls = False
    for m in msgs:
        role = m.get('role')
        content = m.get('content')
        tool_calls = m.get('tool_calls')
        
        print(f"  Role: {role}")
        if tool_calls:
            has_tool_calls = True
            print(f"  Tool Calls: {json.dumps(tool_calls, default=str)}")
        if role == 'tool':
            print(f"  Tool Output (truncated): {str(content)[:100]}...")
        
        # Check specific content of last assistant message
    
    print(f"  Has Tool Calls? {has_tool_calls}")
    print("--------------------------------------------------")


import openai_harmony
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    ReasoningEffort,
    Author,
    TextContent,
)
import inspect

def print_structure(obj, name):
    print(f"\n=== {name} ({type(obj)}) ===")
    for attr in sorted(dir(obj)):
        if not attr.startswith("_"):
            try:
                val = getattr(obj, attr)
                if inspect.ismethod(val) or inspect.isfunction(val):
                    print(f"  [M] {attr}")
                else:
                    print(f"  [A] {attr}: {val}")
            except Exception as e:
                print(f"  [E] {attr}: Error {e}")

def main():
    # 1. Inspect Role
    print_structure(Role, "Role")
    
    # 2. Inspect HarmonyEncodingName
    print_structure(HarmonyEncodingName, "HarmonyEncodingName")
    
    # 3. Create a basic conversation and render it
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    print(f"\nEncoding loaded: {encoding}")
    
    sys_content = SystemContent.new().with_reasoning_effort(ReasoningEffort.HIGH)
    msg_sys = Message.from_role_and_content(Role.SYSTEM, sys_content)
    
    msg_user = Message.from_role_and_content(Role.USER, "Hello, what is 2+2?")
    
    convo = Conversation.from_messages([msg_sys, msg_user])
    
    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    print(f"\nRendered tokens (first 50): {tokens[:50]}")
    
    # Attempt to decode tokens to see the raw string format
    # Note: load_harmony_encoding might provide a way to see raw strings
    # or we can check the encoding object methods
    print_structure(encoding, "Encoding Object")
    
    # Let's try to find a way to see the raw text representation
    # Often there's a 'decode' or 'render_to_string'
    try:
        # Looking at HarmonyEncoding methods in many implementations
        # If it's a wrapper around tiktoken, we might use .decode
        raw_text = encoding._encoding.decode(tokens)
        print(f"\nRaw Rendered Text:\n{raw_text}")
    except:
        pass

if __name__ == "__main__":
    main()

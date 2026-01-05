from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    RenderConversationConfig,
)
import json

try:
    # Try to inspect Message fields keys
    print(Message.model_fields.keys())
except:
    pass

try:
    # Mock tool call structure
    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {"name": "python", "arguments": "print('hello')"}
    }
    
    # Try creating message with tool_calls
    # Note: openai_harmony might require specific object types for tool_calls, not dicts.
    # But usually it mirrors pydantic.
    msg = Message(
        role=Role.ASSISTANT, 
        content="Running code...",
        tool_calls=[tool_call] 
    )
    print("Created Message with tool_calls successfully.")
    
    # Render
    c = Conversation(messages=[msg])
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    tokens = enc.render_conversation_for_training(c)
    print("Rendered:", enc.decode(tokens))

except Exception as e:
    print(f"Error: {e}")

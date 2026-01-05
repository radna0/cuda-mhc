from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    RenderConversationConfig,
)

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
config = RenderConversationConfig(auto_drop_analysis=False)

# Replicate a tool use flow
msgs = []
msgs.append(Message.from_role_and_content(Role.SYSTEM, "System prompt..."))
msgs.append(Message.from_role_and_content(Role.USER, "Calculate 1+1"))

# Assistant call
# Assuming Message has a way to add tool calls?
# Or we construct it raw?
# Let's inspect Message attributes or constructor helper
import inspect
print(inspect.signature(Message.from_role_and_content))

# Assistant Text + Tool Call
# In Harmony, is this one message or split?
# Usually text then call.
# Let's try adding a message with tool call data if possible.
# Inspecting Message class
print(dir(Message))

# Fallback: Create dummy conversation to see if it renders
try: 
    # Try generic construction if specific methods aren't obvious
    m = Message(role=Role.ASSISTANT, content="Thinking...")
    # m.tool_calls = ...?
    pass
except:
    pass

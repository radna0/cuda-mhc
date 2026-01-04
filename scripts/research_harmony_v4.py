
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    ReasoningEffort,
)
import inspect

def main():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    
    # Check signature of render_conversation_for_training
    print(f"Signature: {inspect.signature(encoding.render_conversation_for_training)}")
    
    msg_user = Message.from_role_and_content(Role.USER, "What is 2+2?")
    msg_analysis = Message.from_role_and_content(
        Role.ASSISTANT, 
        "Thinking..."
    ).with_channel("analysis")
    msg_final = Message.from_role_and_content(
        Role.ASSISTANT,
        "It is 4"
    ).with_channel("final")
    
    convo = Conversation.from_messages([
        msg_user, 
        msg_analysis, 
        msg_final
    ])
    
    # Try different rendering methods
    try:
        print("\n--- render_conversation_for_training ---")
        tokens = encoding.render_conversation_for_training(convo)
        print(encoding.decode(tokens))
    except Exception as e:
        print(f"Error: {e}")

    try:
        print("\n--- render_conversation (raw) ---")
        tokens = encoding.render_conversation(convo)
        print(encoding.decode(tokens))
    except Exception as e:
        print(f"Error: {e}")

    # Inspect the Conversation object itself to see if it holds both
    print("\n--- Conversation Messages ---")
    for i, msg in enumerate(convo.messages):
        print(f"Msg {i}: Role={msg.role}, Channel={msg.channel}, Content={msg.content}")

if __name__ == "__main__":
    main()

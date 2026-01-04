
from openai_harmony import (
    HarmonyEncodingName,
    load_harmony_encoding,
    Conversation,
    Message,
    Role,
    SystemContent,
    ReasoningEffort,
)

def main():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    
    # 1. Basic User Message
    msg_user = Message.from_role_and_content(Role.USER, "What is 2+2?")
    convo_basic = Conversation.from_messages([msg_user])
    
    # render_conversation_for_training usually returns tokens
    tokens_train = encoding.render_conversation_for_training(convo_basic)
    
    # Let's try to decode using the internal tiktoken encoding if available, 
    # or find the right method on the wrapper.
    # HarmonyEncoding usually has a .decode(tokens) method.
    try:
        text_train = encoding.decode(tokens_train)
        print("=== Basic Training Format ===")
        print(text_train)
        print("----------------------------")
    except Exception as e:
        print(f"Error decoding training: {e}")

    # 2. System + User + Assistant (Analysis + Final)
    sys_content = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_conversation_start_date("2024-06-01")
    )
    msg_sys = Message.from_role_and_content(Role.SYSTEM, sys_content)
    
    msg_analysis = Message.from_role_and_content(
        Role.ASSISTANT, 
        "The user is asking for the sum of two and two. This is a basic arithmetic problem."
    ).with_channel("analysis")
    
    msg_final = Message.from_role_and_content(
        Role.ASSISTANT,
        "2 + 2 = 4"
    ).with_channel("final")
    
    convo_full = Conversation.from_messages([
        msg_sys, 
        msg_user, 
        msg_analysis, 
        msg_final
    ])
    
    tokens_full = encoding.render_conversation_for_training(convo_full)
    try:
        text_full = encoding.decode(tokens_full)
        print("=== Full Training Format (with Channels) ===")
        print(text_full)
        print("------------------------------------------")
    except Exception as e:
        print(f"Error decoding full: {e}")

if __name__ == "__main__":
    main()

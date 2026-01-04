
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
    ReasoningEffort,
    RenderConversationConfig
)

def research_formats():
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    config = RenderConversationConfig(auto_drop_analysis=False)

    # 1. Test System Content
    system_content = (
        SystemContent.new()
        .with_reasoning_effort(ReasoningEffort.HIGH)
    )
    # The blog mentions "Knowledge cutoff" and "Current date".
    # Let's see if the defaults match.
    
    # 2. Test Developer Content
    developer_content = (
        DeveloperContent.new()
        .with_instructions("You are a helpful assistant.")
    )

    convo = Conversation.from_messages([
        Message.from_role_and_content(Role.SYSTEM, system_content),
        Message.from_role_and_content(Role.DEVELOPER, developer_content),
        Message.from_role_and_content(Role.USER, "Hello"),
        Message.from_role_and_content(Role.ASSISTANT, "Thinking...").with_channel("analysis"),
        Message.from_role_and_content(Role.ASSISTANT, "Hi there!").with_channel("final"),
    ])

    tokens = encoding.render_conversation_for_training(convo, config=config)
    decoded = encoding.decode(tokens)
    
    print("--- FULL RENDERED FORMAT ---")
    print(decoded)
    print("----------------------------")

if __name__ == "__main__":
    research_formats()

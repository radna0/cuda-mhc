
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
    
    msg_user = Message.from_role_and_content(Role.USER, "What is 2+2?")
    
    msg_analysis = Message.from_role_and_content(
        Role.ASSISTANT, 
        "Thinking about 2+2..."
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
    
    # 1. Check for training rendering
    tokens = encoding.render_conversation_for_training(convo)
    text = encoding.decode(tokens)
    print("=== Render for Training ===")
    print(text)
    print("----------------------------")
    
    # 2. Check for completion rendering (what the model sees as input)
    # This should end with <|start|>assistant
    tokens_comp = encoding.render_conversation_for_completion(convo, Role.ASSISTANT)
    text_comp = encoding.decode(tokens_comp)
    print("=== Render for Completion (prefix) ===")
    print(text_comp)
    print("----------------------------")

if __name__ == "__main__":
    main()


import os
import sys
import multiprocessing
import json
import glob
import time
import subprocess
from datasets import load_dataset
from huggingface_hub import login

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
SOURCE_REPO = "nvidia/Nemotron-Math-v2"
REPO_ID = "radna0/nemotron-harmony-formatted"
CACHE_DIR = "/dev/shm/hf_cache"
LOCAL_DATA_DIR = "/dev/shm/nemotron_final"

# Target splits to convert
TARGET_SPLITS = ["high_part01", "high_part02", "medium", "low"]
# TARGET_SPLITS = ["high_part01"] # Debug

# FORCE RAM-ONLY CACHING
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def process_and_upload_split(split_name):
    """
    Downloads a split in streaming mode, processes it to Harmony format, 
    writes to JSONL, uploads to HF, and deletes the local file.
    """
    print(f"\n[Worker] Starting processing for split: {split_name}", flush=True)
    
    # Imports inside worker to avoid fork issues
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        RenderConversationConfig,
        Author
    )
    
    try:
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        config = RenderConversationConfig(auto_drop_analysis=False)
    except Exception as e:
        print(f"Failed to load Harmony encoding: {e}")
        return

    # Updated System/Developer Prompt matching openai-harmony docs
    system_text = (
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n"
        "Current date: 2025-06-28\n\n"
        "Reasoning: high\n\n"
        "# Tools\n\n"
        "## python\n\n"
        "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\n"
        "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )
    msg_system = Message.from_role_and_content(Role.SYSTEM, system_text)
    
    developer_text = (
        "# Instructions\n\n"
        "You will solve the problem and return the final answer in \\boxed{}. The answer is expected to be an integer between 0 and 99999, inclusive. Do not guess the answer, unless specifically given permission to."
    )
    msg_developer = Message.from_role_and_content(Role.DEVELOPER, developer_text)

    # 1. Download Streaming
    print(f"[{split_name}] loading dataset (streaming)...")
    try:
        # Use streaming=True to bypass schema validation issues with 'tools' column
        dataset = load_dataset(SOURCE_REPO, split=split_name, streaming=True)
    except Exception as e:
        print(f"[{split_name}] Failed to load dataset: {e}")
        return

    out_file = os.path.join(LOCAL_DATA_DIR, f"{split_name}.jsonl")
    count = 0
    errors = 0
    
    print(f"[{split_name}] Writing to {out_file}...")
    
    with open(out_file, 'w') as f_out:
        for item in dataset:
            try:
                msgs_raw = item.get("messages", [])
                if len(msgs_raw) < 2:
                    continue

                # --- HARMONY MESSAGE CONSTRUCTION ---
                msgs = [msg_system, msg_developer]
                
                # Iterate through all messages in the conversation
                for m in msgs_raw:
                    role = m.get("role")
                    content = m.get("content")
                    tool_calls = m.get("tool_calls")
                    
                    if role == "user":
                        msgs.append(Message.from_role_and_content(Role.USER, content))
                    
                    elif role == "assistant":
                        # Assistant can have text, reasoning, and tool calls
                        reasoning = m.get("reasoning_content")
                        
                        # 1. Reasoning Output (Analysis)
                        if reasoning and reasoning.strip():
                            msgs.append(Message.from_role_and_content(
                                Role.ASSISTANT, 
                                reasoning.strip()
                            ).with_channel("analysis"))
                        
                        # 2. Tool Calls (Analysis channel, recipient=python)
                        if tool_calls:
                            for tc in tool_calls:
                                func = tc.get("function", {})
                                name = func.get("name")
                                args_str = func.get("arguments", "{}")
                                try:
                                    args = json.loads(args_str)
                                    code = args.get("code", "")
                                except:
                                    code = args_str # Fallback
                                
                                if name == "stateful_python_code_exec" or name == "python":
                                    msgs.append(Message.from_role_and_content(
                                        Role.ASSISTANT, 
                                        code
                                    ).with_channel("analysis").with_recipient("python"))

                        # 3. Content (Final or Analysis)
                        # If content exists and NOT accompanied by tool calls (or even if it is)
                        # Nemotron usually has content as final answer on last turn.
                        if content:
                            # Heuristic: If last message, Final. Else Analysis.
                            # But here we are iterating. We don't know if it's last strictly without lookahead.
                            # However, in Nemotron, intermediate text is usually reasoning (handled above) 
                            # or just part of the chain. 
                            # If we put it in 'final', it stops generation? No.
                            # Safer to put in 'analysis' unless we are sure.
                            # BUT, for the VERY LAST message, we want 'final'.
                            
                            is_last = (m == msgs_raw[-1])
                            channel = "final" if is_last else "analysis"
                            
                            # However, if tool calls are present, content might be empty or preamble.
                            if not tool_calls:
                                msgs.append(Message.from_role_and_content(
                                    Role.ASSISTANT, 
                                    content.strip()
                                ).with_channel(channel))
                            
                    elif role == "tool":
                        # Tool Output -> Role.TOOL
                        # recipient? 'python' usually means it comes FROM python or TO python?
                        # Docs: Author.new(Role.TOOL, "functions.get_current_weather")
                        # We use 'python'
                        msgs.append(Message.from_author_and_content(
                            Author.new(Role.TOOL, "python"),
                            content
                        ).with_channel("analysis")) 
                        # Docs say "commentary" for functions, but python tool is usually analysis flow?
                        # Sample uses 'analysis' for python request, then output?
                        # Actually docs say: "python will respond...". 
                        # Tools generally respond to the same channel? 
                        # Let's use 'analysis' for robust CoT.

                convo = Conversation.from_messages(msgs)
                tokens = encoding.render_conversation_for_training(convo, config=config)
                
                # Create 'conversations' list for pure transparency/debugging (optional, but good for structured dataset)
                # Omitted here to save space, we just need 'text'.
                
                f_out.write(json.dumps({
                    "text": encoding.decode(tokens)
                }) + "\n")
                
                count += 1
                if count % 1000 == 0:
                    print(f"[{split_name}] Processed {count} items...", flush=True)

            except Exception as e:
                errors += 1
                if errors < 10:
                    print(f"[{split_name}] Error processing item: {e}")
                continue

    print(f"[{split_name}] Finished writing {count} items. Uploading to Hub...")
    
    # 2. Upload
    dataset_processed = load_dataset("json", data_files=out_file, split="train")
    dataset_processed.push_to_hub(REPO_ID, split=split_name, private=False)
    print(f"[{split_name}] Uploaded successfully.")
    
    # 3. Cleanup
    os.remove(out_file)
    print(f"[{split_name}] Local file deleted.")


def main():
    print("Turbo Conversion: FULL HARMONY + STREAMING (V2)", flush=True)
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
    
    # Sequential processing to manage memory/disk, or parallel?
    # Streaming uses less disk.
    # Parallel might saturate network.
    # Sequential is safer for reliability.
    
    for split in TARGET_SPLITS:
        process_and_upload_split(split)

if __name__ == "__main__":
    main()

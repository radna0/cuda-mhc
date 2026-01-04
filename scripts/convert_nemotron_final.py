
import os
import sys
import multiprocessing
import json
import glob
import time
import subprocess

# Configuration
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "radna0/nemotron-harmony-formatted"
CACHE_DIR = "/dev/shm/hf_cache"
LOCAL_DATA_DIR = "/dev/shm/nemotron_raw/data"

# FORCE RAM-ONLY CACHING
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def process_chunk(file_path, start_offset, end_offset, part_id, out_dir):
    """Worker function with robust logging and structural Harmony output."""
    from openai_harmony import (
        HarmonyEncodingName,
        load_harmony_encoding,
        Conversation,
        Message,
        Role,
        RenderConversationConfig,
    )
    
    try:
        encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        config = RenderConversationConfig(auto_drop_analysis=False)
    except Exception as e:
        print(f"Worker {part_id}: Failed to load encoding: {e}")
        return

    system_text = (
        "You are ChatGPT, a large language model trained by OpenAI.\n"
        "Knowledge cutoff: 2024-06\n\n"
        "Reasoning: high\n\n"
        "# Tools\n\n"
        "## python\n\n"
        "Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).\n"
        "When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. Internet access for this session is disabled.\n\n"
        "# Valid channels: analysis, commentary, final. Channel must be included for every message."
    )
    msg_system = Message.from_role_and_content(Role.SYSTEM, system_text)
    
    developer_text = (
        "# Instructions\n\n"
        "You will solve the problem and return the final answer in \\boxed{}. The answer is expected to be an integer between 0 and 99999, inclusive. Do not guess the answer, unless specifically given permission to."
    )
    msg_developer = Message.from_role_and_content(Role.DEVELOPER, developer_text)

    out_file = os.path.join(out_dir, f"part_{part_id}.jsonl")
    count = 0
    errors = 0
    
    with open(file_path, 'rb') as f_in, open(out_file, 'w') as f_out:
        if start_offset > 0:
            f_in.seek(start_offset - 1)
            if f_in.read(1) != b'\n':
                f_in.readline()
        
        while f_in.tell() < end_offset:
            line_bytes = f_in.readline()
            if not line_bytes:
                break
            
            try:
                item = json.loads(line_bytes.decode('utf-8'))
                msgs_raw = item.get("messages", [])
                if len(msgs_raw) < 2:
                    continue
                
                problem = msgs_raw[0].get("content", "")
                reasoning = msgs_raw[1].get("reasoning_content", "")
                answer = msgs_raw[1].get("content", "")

                if not problem or not answer:
                    continue

                msg_user = Message.from_role_and_content(Role.USER, problem)
                msgs = [msg_system, msg_developer, msg_user]
                
                if reasoning and reasoning.strip():
                    msgs.append(Message.from_role_and_content(Role.ASSISTANT, reasoning.strip()).with_channel("analysis"))
                    
                msgs.append(Message.from_role_and_content(Role.ASSISTANT, answer.strip()).with_channel("final"))
                
                convo = Conversation.from_messages(msgs)
                tokens = encoding.render_conversation_for_training(convo, config=config)
                
                convo = Conversation.from_messages(msgs)
                tokens = encoding.render_conversation_for_training(convo, config=config)
                
                # Structural output for masking (conversations column)
                # Use original string content to avoid TextContent serialization issues
                conv_list = [
                    {"role": "system", "content": system_text},
                    {"role": "developer", "content": developer_text},
                    {"role": "user", "content": problem}
                ]
                if reasoning and reasoning.strip():
                    conv_list.append({"role": "assistant", "channel": "analysis", "content": reasoning.strip()})
                conv_list.append({"role": "assistant", "channel": "final", "content": answer.strip()})

                f_out.write(json.dumps({
                    "text": encoding.decode(tokens),
                    "conversations": conv_list
                }) + "\n")
                count += 1
                if count % 100 == 0:
                    f_out.flush()
            except Exception as e:
                errors += 1
                if errors < 10:
                    print(f"Worker {part_id} error processing item: {e}")
                    import traceback
                    traceback.print_exc()
                continue

    if count > 0:
        print(f"Worker {part_id} finished: {count} items, {errors} errors.")

def main():
    from huggingface_hub import login
    from datasets import load_dataset
    
    print("Turbo Conversion: STRUCTURAL HARMONY VERSION", flush=True)
    login(token=HF_TOKEN, add_to_git_credential=False)
    
    jsonl_files = sorted(glob.glob(f"{LOCAL_DATA_DIR}/*.jsonl"))
    num_cpus = max(1, multiprocessing.cpu_count() // 2) # Use half cores to avoid pressure
    
    for file_path in jsonl_files:
        split_name = os.path.basename(file_path).replace(".jsonl", "").replace(".", "_")
        print(f"\nProcessing {split_name}...", flush=True)
        
        temp_dir = f"/dev/shm/tmp_{split_name}"
        os.makedirs(temp_dir, exist_ok=True)
        
        file_size = os.path.getsize(file_path)
        chunk_size = file_size // num_cpus
        processes = []
        for p_id in range(num_cpus):
            start = p_id * chunk_size
            end = file_size if p_id == (num_cpus - 1) else (p_id+1) * chunk_size
            p = multiprocessing.Process(target=process_chunk, args=(file_path, start, end, p_id, temp_dir))
            p.start()
            processes.append(p)
        
        for p in processes: p.join()
        
        part_files = sorted(glob.glob(os.path.join(temp_dir, "part_*.jsonl")))
        valid_files = [f for f in part_files if os.path.getsize(f) > 0]
        
        if not valid_files:
            print(f"FAILED: No items processed for {split_name}")
            continue
            
        print(f"Loading {len(valid_files)} parts into Arrow...")
        ds = load_dataset("json", data_files=valid_files, num_proc=os.cpu_count())["train"]
        print(f"Pushing {len(ds)} samples to {REPO_ID} (split: {split_name})...")
        ds.push_to_hub(REPO_ID, split=split_name, private=False)
        
        # Cleanup
        for f in part_files: os.remove(f)
        os.rmdir(temp_dir)

if __name__ == "__main__":
    main()

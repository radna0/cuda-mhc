import os
import json
import glob
import multiprocessing
from datasets import Dataset, disable_caching
import tiktoken
import time
import sys
import traceback

# Force Cache and Temporary files to SHM
os.environ["HF_HOME"] = "/dev/shm/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/dev/shm/hf_cache/datasets"
os.environ["TMPDIR"] = "/dev/shm/tmp"
os.environ["XDG_CACHE_HOME"] = "/dev/shm/cache"
os.makedirs("/dev/shm/hf_cache", exist_ok=True)
os.makedirs("/dev/shm/tmp", exist_ok=True)
os.makedirs("/dev/shm/cache", exist_ok=True)
disable_caching()

# HF Config
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "radna0/nemotron-harmony-formatted"
LOCAL_DATA_DIR = "/dev/shm/nemotron_raw/data"

def process_chunk(file_path, start, end, part_id, temp_dir):
    encoding = tiktoken.get_encoding("cl100k_base")
    out_path = os.path.join(temp_dir, f"part_{part_id}.jsonl")
    
    count = 0
    errors = 0
    with open(file_path, 'rb') as f, open(out_path, 'w') as f_out:
        f.seek(start)
        if start != 0:
            f.readline()
            
        while f.tell() < end:
            line = f.readline()
            if not line: break
            try:
                item = json.loads(line)
                source_messages = item.get("messages", item.get("conversations", []))
                
                if not source_messages:
                    tokens = item.get("tokens", [])
                    if tokens:
                        text = encoding.decode(tokens)
                        conv_list = [{"role": "user", "content": text, "channel": "final"}]
                    else:
                        conv_list = []
                else:
                    conv_list = []
                    for msg in source_messages:
                        # Strict normalization to Hub schema
                        normalized_msg = {
                            "role": str(msg.get("role", "")),
                            "content": str(msg.get("content", "")),
                            "channel": str(msg.get("channel", "final"))
                        }
                        conv_list.append(normalized_msg)
                
                if "text" not in item:
                    top_text = "\n".join([m.get("content", "") for m in conv_list])
                else:
                    top_text = str(item["text"])

                f_out.write(json.dumps({"text": top_text, "conversations": conv_list}) + "\n")
                count += 1
                if count % 2000 == 0: f_out.flush()
            except Exception as e:
                errors += 1
                continue

    if count > 0:
        print(f"Worker {part_id} finished: {count} items.")

def gen_from_jsonl(files):
    for f_path in files:
        try:
            with open(f_path, 'r') as f:
                for line in f:
                    yield json.loads(line)
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            continue

def main():
    from huggingface_hub import login
    print("Turbo Conversion: ROBUST GENERATOR V2 (NO CACHE)", flush=True)
    login(token=HF_TOKEN, add_to_git_credential=False)
    
    jsonl_files = sorted(glob.glob(f"{LOCAL_DATA_DIR}/*.jsonl"))
    num_cpus = max(1, multiprocessing.cpu_count() // 2)
    
    for file_path in jsonl_files:
        split_name = os.path.basename(file_path).replace(".jsonl", "").replace(".", "_")
        if split_name == "high_part_00": continue # already uploaded
            
        print(f"\nProcessing {split_name}...", flush=True)
        temp_dir = f"/dev/shm/tmp_{split_name}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Parallel chunk processing
        if not glob.glob(os.path.join(temp_dir, "part_*.jsonl")):
            print(f"Starting parallel chunk processing for {split_name}...")
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
        else:
            print(f"Found existing JSONL parts for {split_name}, skipping conversion phase.")
        
        part_files = sorted(glob.glob(os.path.join(temp_dir, "part_*.jsonl")))
        valid_files = [f for f in part_files if os.path.getsize(f) > 0]
        
        if not valid_files:
            print(f"FAILED: No items for {split_name}")
            continue
            
        print(f"Pushing {len(valid_files)} shards via Generator to {REPO_ID} (split: {split_name})...")
        try:
            # Reconstruct Dataset with explicit cache_dir to avoid /dev/root
            ds = Dataset.from_generator(
                gen_from_jsonl, 
                gen_kwargs={"files": valid_files},
                cache_dir="/dev/shm/hf_cache/datasets"
            )
            ds.push_to_hub(REPO_ID, split=split_name, private=False, max_shard_size="1000MB")
            print(f"SUCCESS: {split_name} uploaded.")
            # Cleanup only on success
            for f in part_files: os.remove(f)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"CRITICAL UPLOAD FAILURE for {split_name}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()

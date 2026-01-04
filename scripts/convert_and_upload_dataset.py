
import os
import argparse
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login, HfApi

# Strict Harmony System Prompt from sample.txt
SYSTEM_PROMPT = """<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"""

def format_sem_harmony(problem, reasoning, answer):
    # Construct strictly formatted Harmony string
    # System is constant
    # User message
    user_part = f"<|start|>user<|message|>{problem}<|end|>"
    
    # Assistant Analysis (Reasoning)
    if reasoning:
        analysis_part = f"<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>"
    else:
        analysis_part = ""

    # Assistant Final (Answer)
    final_part = f"<|start|>assistant<|channel|>final<|message|>{answer}<|end|>"
    
    # Combine
    return f"{SYSTEM_PROMPT}{user_part}{analysis_part}{final_part}"

def process_item(item):
    try:
        problem = item.get("problem", "")
        
        # Extract reasoning and answer
        # Nemotron structure: reasoningTrajectories list or direct fields
        trajectories = item.get("reasoningTrajectories", [])
        reasoning = ""
        answer = "" # Default
        
        if trajectories and isinstance(trajectories, list) and len(trajectories) > 0:
            # Take first valid trajectory (usually the correct one in fine-tuning datasets)
            traj = trajectories[0]
            # Try likely keys
            reasoning = traj.get("process", traj.get("reasoning", ""))
            answer = traj.get("outcome", traj.get("answer", ""))
        else:
             # Fallback to direct keys
            reasoning = item.get("reasoning", item.get("solution", "")) 
            answer = item.get("answer", item.get("response", ""))

        if not problem or not answer:
             return {"text": None}

        text = format_sem_harmony(problem, reasoning, answer)
        return {"text": text}
        
    except Exception as e:
        return {"text": None}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default="radna0-isekai-vn/nemotron-harmony-formatted")
    args = parser.parse_args()

    # Login
    login(token=args.hf_token)

    # Load Nemotron dataset
    print("Loading nvidia/Nemotron-Math-v2 (Streaming)...")
    # try-except block removed as we handle it in main flow or let it crash to debug

    processed_splits = {}
    
    # We use streaming=True to avoid downloading 37GB to disk (we have limited disk but high RAM)
    # This dataset has 'train' and 'test' usually, or splits in file names.
    # nvidia/Nemotron-Math-v2 usually presents as a single configuration or splits.
    # Let's inspect available splits via HfApi if possible, or just default.
    # streaming=True returns an IterableDatasetDict or IterableDataset.
    
    iterable_ds = load_dataset("nvidia/Nemotron-Math-v2", streaming=True)
    
    for split_name, dataset in iterable_ds.items():
        print(f"Processing split: {split_name} (Streaming into RAM)...")
        
        # Generator for processing
        def gen():
            for item in dataset:
                res = process_item(item)
                if res and res["text"]:
                    yield res

        # Create Dataset in RAM
        # We need to realize it to push to hub properly with metadata
        # RAM is ~178GB, dataset is ~37GB text (compressed?). Text might be larger uncompressed.
        # But we only keep 'text' column. 
        # Risky?
        # Let's try. If OOM, we'll need to shard.
        
        try:
            # from_generator is cleaner
            mem_ds = Dataset.from_generator(gen, features=None) # Features inferred
            processed_splits[split_name] = mem_ds
            print(f"Split {split_name} processed in RAM. Count: {len(mem_ds)}")
        except Exception as e:
            print(f"Failed to process {split_name} in RAM: {e}")
            continue

    if not processed_splits:
        print("No splits processed.")
        return

    # Push to Hub
    print(f"Pushing to {args.repo_id}...")
    final_dd = DatasetDict(processed_splits)
    final_dd.push_to_hub(args.repo_id, private=False)
    print("Upload complete!")

if __name__ == "__main__":
    main()

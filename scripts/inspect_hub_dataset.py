from datasets import load_dataset, get_dataset_config_names
import os

REPO_ID = "radna0/nemotron-harmony-formatted"
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"Checking {REPO_ID}...")
try:
    configs = get_dataset_config_names(REPO_ID, token=HF_TOKEN)
    print(f"Configs: {configs}")
    
    # Try loading the first split
    ds = load_dataset(REPO_ID, split="high_part_00", streaming=True, token=HF_TOKEN)
    it = iter(ds)
    first = next(it)
    print("First example keys:", first.keys())
    print("Conversations type:", type(first['conversations']))
    if len(first['conversations']) > 0:
        print("First msg keys:", first['conversations'][0].keys())
except Exception as e:
    print(f"Error: {e}")

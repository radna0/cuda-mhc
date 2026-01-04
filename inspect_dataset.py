import modal

app = modal.App("inspect-dataset")
image = modal.Image.debian_slim().pip_install("datasets", "huggingface_hub")

@app.function(image=image)
def inspect():
    from datasets import load_dataset
    import json
    
    print("Loading dataset sample...")
    ds = load_dataset("nvidia/Nemotron-Math-v2", split="high_part00", streaming=True)
    sample = next(iter(ds))
    
    print("\nKEYS:", sample.keys())
    print("\nMESSAGES COLUMN EXIST?", "messages" in sample)
    
    if "messages" in sample:
        print("\nMESSAGES SAMPLE:")
        print(json.dumps(sample["messages"], indent=2))
    else:
        print("\nSAMPLE DUMP:")
        # Print first few chars of long fields
        short_sample = {}
        for k,v in sample.items():
            if isinstance(v, str) and len(v) > 100:
                short_sample[k] = v[:100] + "..."
            else:
                short_sample[k] = v
        print(json.dumps(short_sample, indent=2, default=str))

@app.local_entrypoint()
def main():
    inspect.remote()

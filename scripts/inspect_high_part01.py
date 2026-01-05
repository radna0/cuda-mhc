from datasets import load_dataset
import json

print("Inspecting high_part01 schema...")
ds = load_dataset("nvidia/Nemotron-Math-v2", split="high_part01", streaming=True)

# Get first 3 samples
samples = []
for i, item in enumerate(ds):
    if i >= 3:
        break
    samples.append(item)

print("\n=== SCHEMA (columns) ===")
print(samples[0].keys())

print("\n=== SAMPLE 1 ===")
print(json.dumps(samples[0], indent=2, default=str))

print("\n=== SAMPLE 2 ===")
print(json.dumps(samples[1], indent=2, default=str))

# Check if 'tools' exists
if 'tools' in samples[0]:
    print("\n=== TOOLS COLUMN DETECTED ===")
    print(f"Sample 1 tools: {samples[0]['tools']}")
    print(f"Sample 2 tools: {samples[1]['tools']}")

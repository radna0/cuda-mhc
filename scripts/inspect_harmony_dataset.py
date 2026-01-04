
from datasets import load_dataset
import sys

def main():
    print("Loading radna0-isekai-vn/nemotron-harmony-formatted (Streaming)...")
    ds = load_dataset("radna0-isekai-vn/nemotron-harmony-formatted", split="train", streaming=True)
    
    print("\n=== INSPECTION START ===\n")
    
    count = 0
    for item in ds:
        text = item['text']
        print(f"--- SAMPLE {count} ---")
        print(text)
        print("\n------------------\n")
        
        # Validation checks
        errors = []
        if "<|start|>system<|message|>" not in text:
            errors.append("Missing System Prompt Header")
        if "# Valid channels: analysis, commentary, final" not in text:
            errors.append("System Prompt seems to lack channel definition")
        if "<|start|>user<|message|>" not in text:
            errors.append("Missing User Message")
        if "<|start|>assistant<|channel|>final<|message|>" not in text:
            errors.append("Missing Final Answer Channel")
            
        if errors:
            print(f"❌ ERRORS FOUND: {errors}")
        else:
            print("✅ BASIC STRUCTURE OK")
            
        count += 1
        if count >= 3:
            break
            
    print("\n=== INSPECTION END ===\n")

if __name__ == "__main__":
    main()

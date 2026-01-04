import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
import json

SYSTEM_PROMPT = """You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06

Reasoning: high

# Tools

## python

Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. Internet access for this session is disabled.

# Valid channels: analysis, commentary, final. Channel must be included for every message."""

DEVELOPER_PROMPT = """# Instructions

You will solve the problem and return the final answer in \\boxed{}. The answer is expected to be an integer between 0 and 99999, inclusive. Do not guess the answer, unless specifically given permission to."""

def format_harmony(problem, reasoning, answer):
    # Harmony format tokens (BIT-PERFECT for training)
    # <|start|>system<|message|>...<|end|>
    # <|start|>developer<|message|>...<|end|>
    # <|start|>user<|message|>...<|end|>
    # <|start|>assistant<|channel|>analysis<|message|>...<|end|>
    # <|start|>assistant<|channel|>final<|message|>...<|return|>
    
    # Ensure no leading/trailing whitespace in content
    problem = problem.strip()
    reasoning = reasoning.strip()
    answer = answer.strip()
    
    sequence = (
        f"<|start|>system<|message|>{SYSTEM_PROMPT}<|end|>"
        f"<|start|>developer<|message|>{DEVELOPER_PROMPT}<|end|>"
        f"<|start|>user<|message|>{problem}<|end|>"
        f"<|start|>assistant<|channel|>analysis<|message|>{reasoning}<|end|>"
        f"<|start|>assistant<|channel|>final<|message|>{answer}<|return|>"
    )
    return sequence

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/nemotron_harmony_train.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print(f"Loading nvidia/Nemotron-Math-v2...")
    ds = load_dataset("nvidia/Nemotron-Math-v2", split="high_part00", streaming=True)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    samples = []
    count = 0
    for item in tqdm(ds, total=args.max_samples):
        if count >= args.max_samples:
            break
            
        # Priority: Check for 'messages' column first
        messages = item.get("messages", [])
        if messages and isinstance(messages, list):
            # Extract content from messages
            # Typically user -> assistant (with reasoning/answer split sometimes, or just answer)
            # We need to map this to Harmony format logic
            
            # Simple heuristic: last assistant message is the answer/reasoning source
            # But wait, Harmony format expects specific structure. 
            # If the dataset already has messages, we should try to preserve them?
            # Actually, format_harmony function builds specific Harmony XML tags.
            # Let's see if we can extract problem/reasoning/answer from messages.
            
            p_text = ""
            r_text = ""
            a_text = ""
            
            for m in messages:
                role = m.get("role")
                content = m.get("content", "")
                if role == "user":
                    p_text = content
                elif role == "assistant":
                    # Check if it has reasoning content? 
                    # Some datasets split reasoning into a separate field or inner tags
                    # For Nemotron, usually 'reasoning' is separate. 
                    # If 'messages' is just standard chat, it might not have explicit reasoning field.
                    # Inspection showed:
                    # 'role': 'assistant', 'content': 'To find the derivative... \nFinal Answer: ...'
                    # So it mixes them.
                    
                    # Fallback to existing logic if messages doesn't split reasoning clearly
                    pass
            
            # If we couldn't easily parse messages, fall back to the explicit columns which we know exist
            pass

        problem = item.get("problem", "")
        # The dataset has 'reasoningTrajectories' which is a list.
        trajectories = item.get("reasoningTrajectories", [])
        if isinstance(trajectories, list) and len(trajectories) > 0:
            # Each trajectory has 'process' (reasoning) and 'outcome' (answer)
            # Or similar structure. Let's be robust.
            traj = trajectories[0]
            reasoning = traj.get("process", traj.get("reasoning", ""))
            answer = traj.get("outcome", traj.get("answer", item.get("answer", "")))
        else:
            reasoning = item.get("reasoning", "")
            answer = item.get("answer", "")
        
        if not problem or not answer:
            continue
            
        formatted = format_harmony(problem, reasoning, answer)
        samples.append({"text": formatted})
        count += 1
        
    if args.dry_run:
        print("\n--- DRY RUN SAMPLE ---")
        print(samples[0]["text"])
        print("----------------------\n")
    else:
        with open(args.output, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"Saved {len(samples)} samples to {args.output}")

if __name__ == "__main__":
    main()

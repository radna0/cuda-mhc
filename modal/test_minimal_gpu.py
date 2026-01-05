import modal
import os
import random
import string
from pathlib import Path

def gen_stealth_name():
    prefixes = ["backend", "service", "api", "worker", "job", "processor", "handler", "task"]
    suffixes = ["alpha", "beta", "v1", "prod", "staging", "dev", "main", "core"]
    salt = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{random.choice(prefixes)}-{random.choice(suffixes)}-{salt}"

app_name = gen_stealth_name()
print(f"[Test] Initializing App: {app_name}")
app = modal.App(app_name)

# Minimal image
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")

@app.function(image=image, gpu="A100", timeout=300)
def test_gpu():
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    return {"success": True, "gpu": torch.cuda.get_device_name(0)}

@app.local_entrypoint()
def main():
    print("Launching minimal GPU test...")
    try:
        result = test_gpu.remote()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Failed: {e}")

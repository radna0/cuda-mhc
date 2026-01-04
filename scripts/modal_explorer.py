import modal
import os

app = modal.App("gpt-oss-explorer")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .pip_install("torch", "transformers", "bitsandbytes", "accelerate")
    .run_commands(
        "pip install git+https://github.com/unslothai/unsloth-zoo.git",
        "pip install --no-deps git+https://github.com/unslothai/unsloth.git"
    )
)

@app.function(image=image)
def find_files():
    import subprocess
    res = subprocess.run(["find", "/usr/local/lib/python3.11/site-packages", "-name", "*gpt_oss*"], capture_output=True, text=True)
    return res.stdout

@app.local_entrypoint()
def main():
    print(find_files.remote())

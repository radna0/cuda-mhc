import modal
import os

app = modal.App("gpt-oss-explorer-v2")
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .pip_install("torch", "transformers", "bitsandbytes", "accelerate")
    .run_commands(
        "pip install git+https://github.com/unslothai/unsloth-zoo.git",
        "pip install --no-deps git+https://github.com/unslothai/unsloth.git"
    )
)

@app.function(image=image)
def examine_utils():
    import subprocess
    # Read the file directly
    res = subprocess.run(["cat", "/usr/local/lib/python3.11/site-packages/unsloth/models/_utils.py"], capture_output=True, text=True)
    return res.stdout

@app.local_entrypoint()
def main():
    content = examine_utils.remote()
    with open("logs/unsloth_utils_source.py", "w") as f:
        f.write(content)
    print("Source saved to logs/unsloth_utils_source.py")

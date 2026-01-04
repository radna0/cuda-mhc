import modal
import os

app = modal.App("gpt-oss-explorer-v4")
vol = modal.Volume.from_name("mhc-model-vol", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .pip_install("torch", "transformers", "bitsandbytes", "accelerate")
    .run_commands(
        "pip install git+https://github.com/unslothai/unsloth-zoo.git",
        "pip install --no-deps git+https://github.com/unslothai/unsloth.git"
    )
)

@app.function(image=image, volumes={"/vol": vol})
def write_utils():
    import subprocess
    content = subprocess.run(["cat", "/usr/local/lib/python3.11/site-packages/unsloth/models/_utils.py"], capture_output=True, text=True).stdout
    with open("/vol/unsloth_utils.py", "w") as f:
        f.write(content)
    return "Done"

@app.local_entrypoint()
def main():
    print(write_utils.remote())

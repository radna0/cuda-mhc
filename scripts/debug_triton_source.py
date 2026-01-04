import modal
from pathlib import Path

# Local Source Paths (Must match the main script)
UNSLOTH_SRC = Path("/home/kojoe/CUDA_mhc/unsloth_src")
UNSLOTH_ZOO_SRC = Path("/home/kojoe/CUDA_mhc/unsloth_zoo_src")

app = modal.App("debug-triton-shapes")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git", "build-essential", "clang", "python3-dev")
    .run_commands("pip install --upgrade pip")
    .run_commands(
        "pip install torch==2.9.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
    .run_commands(
        "pip install 'transformers>=4.57.1,!=4.57.2' bitsandbytes peft accelerate sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec ninja wandb"
    )
    .run_commands(
        "pip install git+https://github.com/unslothai/unsloth-zoo.git",
        "pip install --no-deps git+https://github.com/unslothai/unsloth.git",
        "pip install git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels",
        "pip install --upgrade 'trl==0.9.6' 'rich'"
    )
    .add_local_dir(UNSLOTH_SRC, remote_path="/root/unsloth_src", copy=True)
    .add_local_dir(UNSLOTH_ZOO_SRC, remote_path="/root/unsloth_zoo_src", copy=True)
    .run_commands(
        "cp -rfv /root/unsloth_src/unsloth/* /usr/local/lib/python3.11/site-packages/unsloth/",
        "cp -rfv /root/unsloth_zoo_src/unsloth_zoo/* /usr/local/lib/python3.11/site-packages/unsloth_zoo/",
        "find /usr/local/lib/python3.11/site-packages/unsloth -name '__pycache__' -type d -exec rm -rf {} +",
        "find /usr/local/lib/python3.11/site-packages/unsloth_zoo -name '__pycache__' -type d -exec rm -rf {} +",
    )
)

@app.function(image=image)
def debug_kernel():
    import os
    import triton_kernels.matmul_ogs as m
    print(f"File: {m.__file__}")
    with open(m.__file__, "r") as f:
        lines = f.readlines()
        
    start = max(0, 321 - 50)
    end = min(len(lines), 321 + 50)
    for i in range(start, end):
        print(f"{i+1:4}: {lines[i].rstrip()}")

if __name__ == "__main__":
    with modal.Retrying():
        app.run()

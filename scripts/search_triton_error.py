import modal
from pathlib import Path

# Local Source Paths
UNSLOTH_SRC = Path("/home/kojoe/CUDA_mhc/unsloth_src")
UNSLOTH_ZOO_SRC = Path("/home/kojoe/CUDA_mhc/unsloth_zoo_src")

app = modal.App("debug-triton-blackwell")

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
)

@app.function(image=image)
def search_triton():
    import os
    import subprocess
    import triton_kernels
    path = os.path.dirname(triton_kernels.__file__)
    print(f"SEARCHING IN {path}")
    
    # Use grep to find the error message
    try:
        res = subprocess.check_output(["grep", "-r", "Only Hopper swizzling is supported for values", path], text=True)
        print(res)
    except subprocess.CalledProcessError as e:
        print(f"Grep failed or found nothing: {e}")

if __name__ == "__main__":
    with modal.Retrying():
        app.run()

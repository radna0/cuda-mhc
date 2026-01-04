import modal
from pathlib import Path

app = modal.App("debug-triton-kernels")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .run_commands("pip install git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels")
)

@app.function(image=image)
def inspect_kernel():
    import os
    import triton_kernels
    path = os.path.dirname(triton_kernels.__file__)
    target_file = os.path.join(path, "matmul_ogs.py")
    
    if os.path.exists(target_file):
        print(f"READING {target_file}")
        with open(target_file, "r") as f:
            content = f.readlines()
            
        start = max(0, 321 - 50)
        end = min(len(content), 321 + 100)
        
        for i in range(start, end):
            print(f"{i+1}: {content[i].strip()}")
    else:
        print(f"File not found: {target_file}")
        print(f"Contents of {path}:")
        print(os.listdir(path))

if __name__ == "__main__":
    with modal.Retrying(modal.ConnectionError):
        app.run()

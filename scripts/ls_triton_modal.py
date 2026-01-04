import modal
from pathlib import Path

app = modal.App("debug-triton-ls")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .run_commands("pip install git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels")
)

@app.function(image=image)
def ls_triton():
    import os
    import triton_kernels
    path = os.path.dirname(triton_kernels.__file__)
    print(f"CONTENTS OF {path}:")
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

if __name__ == "__main__":
    with app.run():
        ls_triton.remote()

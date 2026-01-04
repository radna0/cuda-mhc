import modal
import os

app = modal.App("trl-debug")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.11",
    )
    .apt_install("git")
    .run_commands("pip install --upgrade pip")
    .run_commands(
        "pip install torch==2.9.0 torchvision torchaudio triton --extra-index-url https://download.pytorch.org/whl/cu128"
    )
    .run_commands(
        "pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128"
    )
    .run_commands(
        "pip install 'transformers==4.56.2' sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec"
    )
    .pip_install(
        "bitsandbytes",
        "trl==0.9.6",
        "peft",
        "accelerate",
        "wandb",
        "rich"
    )
    .run_commands(
        "pip install git+https://github.com/unslothai/unsloth-zoo.git",
        "pip install --no-deps git+https://github.com/unslothai/unsloth.git",
        "pip install git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels"
    )
)

@app.function(image=image, gpu="A10G")
def check_trl():
    import subprocess
    import unsloth
    import inspect
    import os
    
    # Try to find patch_gpt_oss
    found = False
    for name, obj in inspect.getmembers(unsloth.models.loader if hasattr(unsloth.models, 'loader') else unsloth):
        if name == "patch_gpt_oss":
            print(f"FOUND patch_gpt_oss in {obj.__module__}")
            print("SOURCE:")
            print(inspect.getsource(obj))
            found = True
            break
            
    if not found:
        print("Searching all modules for patch_gpt_oss...")
        import pkgutil
        import importlib
        for loader, module_name, is_pkg in pkgutil.walk_packages(unsloth.__path__, unsloth.__name__ + "."):
            try:
                module = importlib.import_module(module_name)
                for name, obj in inspect.getmembers(module):
                    if name == "patch_gpt_oss":
                        print(f"FOUND patch_gpt_oss in {module_name}")
                        print("SOURCE:")
                        print(inspect.getsource(obj))
                        found = True
                        break
            except:
                continue
            if found: break

    if not found:
        print("Could not find patch_gpt_oss source.")
    
    paths = [
        "trl.DataCollatorForCompletionOnlyLM",
        "trl.trainer.DataCollatorForCompletionOnlyLM",
        "trl.trainer.utils.DataCollatorForCompletionOnlyLM"
    ]
    
    for path in paths:
        try:
            parts = path.split(".")
            import importlib
            mod = importlib.import_module(".".join(parts[:-1]))
            obj = getattr(mod, parts[-1])
            print(f"Found {path}")
        except (ImportError, AttributeError):
            print(f"Not found {path}")

if __name__ == "__main__":
    with modal.Retrying(max_retries=1):
        check_trl.remote()

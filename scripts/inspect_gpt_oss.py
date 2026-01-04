import modal

app = modal.App("gpt-oss-inspect")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu24.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.9.0",
        "transformers==4.56.2",
        "peft",
        "bitsandbytes",
        "kernels",
        "packaging",
        "numpy"
    )
    .run_commands(
        "pip install git+https://github.com/unslothai/unsloth-zoo.git",
        "pip install --no-deps git+https://github.com/unslothai/unsloth.git"
    )
)

@app.function(image=vllm_image, gpu="H100")
def inspect_model():
    from unsloth import FastLanguageModel
    import torch
    
    model_name = "unsloth/gpt-oss-20b"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = 2048,
        dtype = torch.bfloat16,
        load_in_4bit = True,
    )
    
    print("Model Class:", type(model))
    print("Attributes of model:", dir(model))
    
    curr = model
    if hasattr(curr, "base_model"): curr = curr.base_model
    print("Base model type:", type(curr))
    print("Base model attributes:", dir(curr))
    
    if hasattr(curr, "model"): curr = curr.model
    print("HF model type:", type(curr))
    print("HF model attributes:", [a for a in dir(curr) if not a.startswith("__")])
    
    for attr in dir(curr):
        if not attr.startswith("__"):
            val = getattr(curr, attr)
            if hasattr(val, "layers") or hasattr(val, "h"):
                print(f"FOUND LAYERS in attribute: {attr} (Type: {type(val)})")
                break

@app.local_entrypoint()
def main():
    inspect_model.remote()

if __name__ == "__main__":
    main()

import modal
import os
import time
import logging
from pathlib import Path
from modal_utils import ProjectApp, d

# Level-2 Masked Constants
BASE_IMG = d("bnZpZGlhL2N1ZGE6MTIuOC4wLWRldmVsLXVidW50dTI0LjA0")
HW_TYPE = d("SDEwMA==") # H100

# Local Path Discovery
ROOT_DIR = Path(__file__).parent.parent
UNSLOTH_SRC = ROOT_DIR / "unsloth_src"
UNSLOTH_ZOO_SRC = ROOT_DIR / "unsloth_zoo_src"
TRITON_KERNELS_SRC = ROOT_DIR / "triton_src" / "python" / "triton_kernels" / "triton_kernels"

# Initialize Backend App
p_app = ProjectApp("system-sync")
app = p_app.app

# Shared Volumes
volume = modal.Volume.from_name("mhc-data-volume", create_if_missing=True)
model_volume = modal.Volume.from_name("gpt-oss-models", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)

# Container Image Construction
base_image = (
    modal.Image.from_registry(BASE_IMG, add_python="3.11")
    .apt_install("git", "build-essential", "clang", "python3-dev")
    .run_commands("pip install --upgrade pip")
    .run_commands("pip install torch==2.9.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128")
    .run_commands("pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 flashinfer-jit-cache==0.5.3 --extra-index-url https://flashinfer.ai/whl/cu128")
    .run_commands("pip install 'transformers>=4.57.1,!=4.57.2' bitsandbytes peft accelerate sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec ninja wandb huggingface_hub hf_transfer")
    .run_commands(
        "pip install git+https://github.com/unslothai/unsloth-zoo.git",
        "pip install --no-deps git+https://github.com/unslothai/unsloth.git",
        "pip install git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels",
        "pip install --upgrade 'trl==0.9.6' 'rich'"
    )
)

# Apply Internal Mounts (Managed Paths)
image, REMOTE_UNSLOTH_PATH = p_app.internal_mount(base_image, UNSLOTH_SRC)
image, REMOTE_ZOO_PATH = p_app.internal_mount(image, UNSLOTH_ZOO_SRC)
image, REMOTE_TRITON_PATH = p_app.internal_mount(image, TRITON_KERNELS_SRC)

# Workspace Finalization
image = p_app.prepare_container(image).run_commands(
    f"cp -rfv {REMOTE_UNSLOTH_PATH}/unsloth/* /usr/local/lib/python3.11/site-packages/unsloth/",
    f"cp -rfv {REMOTE_ZOO_PATH}/unsloth_zoo/* /usr/local/lib/python3.11/site-packages/unsloth_zoo/",
    f"cp -rfv {REMOTE_TRITON_PATH}/* /usr/local/lib/python3.11/site-packages/triton_kernels/",
    "find /usr/local/lib/python3.11/site-packages/unsloth -name '__pycache__' -type d -exec rm -rf {} +",
    "find /usr/local/lib/python3.11/site-packages/unsloth_zoo -name '__pycache__' -type d -exec rm -rf {} +",
    "find /usr/local/lib/python3.11/site-packages/triton_kernels -name '__pycache__' -type d -exec rm -rf {} +"
).add_local_file(ROOT_DIR / "modal" / "modal_utils.py", remote_path="/root/modal_utils.py") \
 .add_local_file(ROOT_DIR / "scripts" / "mhc_huggingface_adapter.py", remote_path="/root/mhc_huggingface_adapter.py") \
 .add_local_file(ROOT_DIR / "scripts" / "format_nemotron_harmony.py", remote_path="/root/format_nemotron_harmony.py") \
 .add_local_file(ROOT_DIR / "modal" / "mhc_training_benchmark.py", remote_path="/root/mhc_training_benchmark.py") \
 .add_local_file(ROOT_DIR / "gpt_oss_config.json", remote_path="/root/gpt_oss_config.json")

@app.function(image=image)
def inspect_triton_details():
    import os
    import triton_kernels
    path = os.path.dirname(triton_kernels.__file__)
    details_path = os.path.join(path, "matmul_ogs_details")
    if os.path.exists(details_path):
        for f in os.listdir(details_path):
            if f.endswith(".py"):
                with open(os.path.join(details_path, f), "r") as src:
                    print(src.read())

@app.function(
    image=image,
    gpu=HW_TYPE,
    volumes={
        "/root/data": volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume
    },
    ephemeral_disk=512 * 1024,
    secrets=[modal.Secret.from_dict({
        "WANDB_API_KEY": "25cca6253942b7651bf7942122a976af7d5449b2",
        "HF_HUB_READ_TIMEOUT": "120",
        "HF_HUB_ENABLE_HF_TRANSFER": "1"
    })],
    timeout=7200,
    cpu=12.0,
    memory=65536,
)
def run_benchmark(mhc_enabled: bool = False, max_steps: int = 500):
    import os
    import torch
    import torch.nn as nn
    from huggingface_hub import snapshot_download

    model_repo_id = d("b3BlbmFpL2dwdC1vc3MtMjBi") # openai/gpt-oss-20b
    model_name = "/root/model/openai/gpt-oss-20b"
    max_seq_length = 65536

    # Managed Model Download
    weight_files = ["model.safetensors", "pytorch_model.bin"]
    if not any(os.path.exists(os.path.join(model_name, f)) for f in weight_files):
        try:
            snapshot_download(repo_id=model_repo_id, local_dir=model_name, local_dir_use_symlinks=False)
            model_volume.commit()
        except Exception as e:
            raise RuntimeError(f"Sync Failure: {e}")

    # Environment Setup
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["HF_HOME"] = "/root/hf_cache"
    os.makedirs("/root/hf_cache", exist_ok=True)
    
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from mhc_huggingface_adapter import patch_gpt_oss_with_mhc
    
    # Config Enforcement
    import shutil
    if not os.path.exists(model_name): os.makedirs(model_name, exist_ok=True)
    shutil.copy("/root/gpt_oss_config.json", os.path.join(model_name, "config.json"))

    # Router Patching Logic (MHC Stability)
    old_module_init = nn.Module.__init__
    def patched_module_init(self, *args, **kwargs):
        old_module_init(self, *args, **kwargs)
        if "Router" in self.__class__.__name__:
            for target in [self] + ([self.linear] if hasattr(self, "linear") else []):
                if not hasattr(target, "weight"): target.register_parameter("weight", nn.Parameter(torch.zeros(1)))
                if not hasattr(target, "bias"): target.register_parameter("bias", nn.Parameter(torch.zeros(1)))
    nn.Module.__init__ = patched_module_init

    # Model Load
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
        unsloth_tiled_mlp = True,
    )

    # PEFT Setup
    model = FastLanguageModel.get_peft_model(
        model, r = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32, lora_dropout = 0, bias = "none", use_gradient_checkpointing = "unsloth", random_state = 3407
    )

    # Dataset & Training (High Part 00)
    dataset = load_dataset("parquet", data_files={"train": "/root/data/nemotron-harmony/data/high_part_00-*.parquet"})["train"]
    if len(dataset) > 20000: dataset = dataset.select(range(20000))

    # Trainer Config
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model = model, tokenizer = tokenizer, train_dataset = dataset,
        dataset_text_field = "content", max_seq_length = max_seq_length,
        dataset_num_proc = 32, packing = True,
        args = SFTConfig(
            per_device_train_batch_size = 8, gradient_accumulation_steps = 16,
            warmup_steps = 5, max_steps = max_steps, learning_rate = 2e-4, fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(), logging_steps = 1, optimismer = "adamw_8bit",
            weight_decay = 0.01, lr_scheduler_type = "linear", seed = 3407, output_dir = "outputs",
        ),
    )
    trainer.train()

@app.local_entrypoint()
def main(mhc_enabled: bool = False, max_steps: int = 500):
    run_benchmark.remote(mhc_enabled, max_steps)

import modal
import os
import time

# Modal Setup
app = modal.App("gpt-oss-mhc-benchmark")

# Volume for model weights and datasets
volume = modal.Volume.from_name("gpt-oss-training-data", create_if_missing=True)
model_volume = modal.Volume.from_name("gpt-oss-model-weights-ads-70439")

# Image definition following user requirements + Unsloth from source
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
        "wandb"
    )
    .run_commands(
        "pip install git+https://github.com/unslothai/unsloth-zoo.git",
        "pip install --no-deps git+https://github.com/unslothai/unsloth.git",
        "pip install git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels"
    )
    .add_local_file("scripts/mhc_huggingface_adapter.py", remote_path="/root/mhc_huggingface_adapter.py")
    .add_local_file("scripts/format_nemotron_harmony.py", remote_path="/root/format_nemotron_harmony.py")
)

@app.function(
    image=image,
    gpu="B200",  # Use B200/H100 as requested
    volumes={
        "/root/data": volume,
        "/root/model": model_volume
    },
    ephemeral_disk=512 * 1024, # 512GB minimum as per Modal requirement
    secrets=[modal.Secret.from_dict({
        "WANDB_API_KEY": "25cca6253942b7651bf7942122a976af7d5449b2",
        "HF_HUB_READ_TIMEOUT": "120",
        "HF_HUB_ENABLE_HF_TRANSFER": "1"
    })],
)
def run_benchmark(mhc_enabled: bool = False, max_steps: int = 500):
    import torch
    import os
    
    # Redirect HF cache to persistent volume or large ephemeral disk
    os.environ["HF_HOME"] = "/root/data/cache"
    os.environ["HF_DATASETS_CACHE"] = "/root/data/cache/datasets"
    os.makedirs("/root/data/cache", exist_ok=True)
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer
    from datasets import load_dataset
    from mhc_huggingface_adapter import patch_gpt_oss_with_mhc
    import json
    
    # Use the local path in the volume
    model_name = "/root/model/openai/gpt-oss-20b"
    max_seq_length = 65536 # Scale to 65k context
    
    print(f"Loading model from {model_name}... (MHC Enabled: {mhc_enabled})")
    
    # Fundamental monkeypatch for Unsloth GptOssTopKRouter stability.
    import torch.nn as nn
    import logging

    # 1. Total logging suppression (V9-FinalStability)
    import logging
    logging.disable(logging.CRITICAL) # Kill all critical logs globally
    
    try:
        from unsloth.models.loader import UnslothLogHandler, logger as unsloth_logger
        UnslothLogHandler.emit = lambda self, record: None
        unsloth_logger.handlers = []
        unsloth_logger.propagate = False
        print("Successfully nuked Unsloth logging")
    except Exception as e:
        print(f"Warning: Could not nuke Unsloth logging: {e}")

    # 2. Hardened fallback patch
    old_callHandlers = logging.Logger.callHandlers
    def patched_callHandlers(self, record):
        return # DROP EVERYTHING during model load if necessary
    
    # We only apply the 'drop everything' patch during from_pretrained
    print("Applying temporary global log block (V9)...")
    logging.Logger.callHandlers = patched_callHandlers

    # 2. Patch nn.Module.__init__ for the Router structure
    old_module_init = nn.Module.__init__
    def patched_module_init(self, *args, **kwargs):
        old_module_init(self, *args, **kwargs)
        name = self.__class__.__name__
        if name == "GptOssTopKRouter" or ("GptOss" in name and "Router" in name):
            # Check self and child 'linear'
            modules_to_patch = [self]
            if hasattr(self, "linear") and isinstance(self.linear, nn.Module):
                modules_to_patch.append(self.linear)
            
            for target in modules_to_patch:
                # Ensure weight/bias exist for Unsloth's checks
                if not hasattr(target, "weight"):
                    target.register_parameter("weight", nn.Parameter(torch.zeros(1)))
                if not hasattr(target, "bias"):
                    target.register_parameter("bias", nn.Parameter(torch.zeros(1)))
                
                # Monkeypatch load_state_dict to handle resizing from checkpoint
                old_lsd = target.load_state_dict
                def make_patched_lsd(t, old):
                    def patched_lsd(state_dict, strict=True, assign=False):
                        if "weight" in state_dict:
                            new_param = nn.Parameter(torch.zeros_like(state_dict["weight"]))
                            t.register_parameter("weight", new_param)
                        if "bias" in state_dict:
                            new_param = nn.Parameter(torch.zeros_like(state_dict["bias"]))
                            t.register_parameter("bias", new_param)
                        return old(state_dict, strict=strict, assign=assign)
                    return patched_lsd
                target.load_state_dict = make_patched_lsd(target, old_lsd)
    nn.Module.__init__ = patched_module_init
    print("Monkeypatched nn.Module.__init__ and Logger (V6-Global)")

    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
        full_finetuning = False,
        unsloth_tiled_mlp = True,
    )
    
    # Restore standard logging after critical loading phase
    logging.Logger.callHandlers = old_callHandlers
    logging.disable(logging.NOTSET)
    print("Restored standard logging after model load.")
    
    # Apply Harmony Chat Template (Manual jinja for bit-perfect replication)
    print("Applying Manual Harmony Chat Template...")
    # This template follows docs/openai-harmony.md strictly
    # We use .get('channel') and check if it's not None to avoid <|channel|>None
    harmony_template = (
        "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}"
        "{% for message in messages %}"
        "<|start|>{{ message['role'] }}"
        "{% if message.get('channel') is not none %}<|channel|>{{ message['channel'] }}{% endif %}"
        "<|message|>{{ message['content'] }}<|end|>"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|start|>assistant<|channel|>final<|message|>{% endif %}"
    )
    tokenizer.chat_template = harmony_template
    print("Successfully set manual Harmony chat template.")
    
    # 2. Add LoRA - Rank 16 for better quality
    print("Applying PEFT/LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
    )

    start_time = time.time()
    
    try:
        # Load Dataset
        import os
        from datasets import load_dataset
        local_ds_path = "/dev/shm/tmp_high_part_00"
        if os.path.exists(local_ds_path) and len(os.listdir(local_ds_path)) > 0:
            print(f"Loading dataset from local path: {local_ds_path}")
            dataset = load_dataset("json", data_files=os.path.join(local_ds_path, "*.jsonl"), split="train")
        else:
            print(f"Loading dataset from HF (radna0/nemotron-harmony-formatted/high_part_00)...")
            dataset = load_dataset("radna0/nemotron-harmony-formatted", split = "high_part_00")
        
        print(f"Success: Loaded dataset with {len(dataset)} items")
        
        # Take 100k samples for high quality but manageable benchmark
        if len(dataset) > 100000:
            print(f"Selecting 100,000 samples from {len(dataset)}...")
            dataset = dataset.select(range(100000))
            
        def formatting_prompts_func(examples):
            convs = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convs]
            return { "text" : texts, }
        
        print("Mapping dataset with chat template (Parallel num_proc=16)...")
        dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=16)
        
        print(f"Verification: First mapped text sample (prefix): {dataset[0]['text'][:200]}...")
        print(f"Training on {len(dataset)} samples for {max_steps} steps...")
        
        # 5. Trainer with Optimizations
        from transformers import TrainerCallback
        class SanityCheckCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if mhc_enabled:
                    from mhc_huggingface_adapter import check_weights
                    check_weights(kwargs["model"], step=state.global_step)

        # Exact OpenAI Harmony SFT Masking
        from trl import DataCollatorForCompletionOnlyLM
        response_template = "<|start|>assistant"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template,
            tokenizer=tokenizer,
        )

        print("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            data_collator = collator, 
            max_seq_len = max_seq_length,
            callbacks = [SanityCheckCallback()],
            args = SFTConfig(
                per_device_train_batch_size = 1, # BS1 for 65k context stability
                gradient_accumulation_steps = 32, # effective batch 32
                max_steps = max_steps,
                learning_rate = 2e-5,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.001,
                adam_epsilon = 1e-20,
                adam_beta1 = 0.9,
                adam_beta2 = 0.95,
                lr_scheduler_type = "linear",
                warmup_steps = 20,
                seed = 3407,
                output_dir = "outputs",
                packing = False, 
                report_to = "wandb",
                save_strategy = "no",
                run_name = f"gpt-oss-20b-baseline-final",
                max_grad_norm = 1.0,
            ),
        )
        
        # Train
        print("Starting training loop...")
        train_result = trainer.train()
        
        end_time = time.time()
        total_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_reserved() / 1024**3
        
        metrics = {
            "train_runtime": train_result.metrics.get("train_runtime", total_time),
            "train_loss": train_result.metrics.get("train_loss"),
            "peak_vram_gb": peak_memory,
            "history": trainer.state.log_history
        }
        
        print(f"Benchmark finished. Time: {total_time:.2f}s, Peak VRAM: {peak_memory:.2f}GB")
        return metrics

    except Exception as e:
        print(f"\n!!! BENCHMARK CRITICAL FAILURE !!!\nError: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.local_entrypoint()
def main():
    import json
    import os
    
    os.makedirs("logs", exist_ok=True)
    
    # Run Baseline - PRIMARY FOCUS
    print("Launching Baseline SFT Benchmark (500 Steps)...")
    try:
        baseline_metrics = run_benchmark.remote(mhc_enabled=False, max_steps=500)
    except Exception as e:
        print(f"Benchmark execution failed: {e}")
        baseline_metrics = {"error": str(e)}
    
    results = {
        "baseline": baseline_metrics,
        "mhc": {"skipped": True}
    }
    
    output_file = "logs/baseline_final_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_file}")

import modal
import os
import time

# Modal Setup
app = modal.App("gpt-oss-baseline-clean")

# Volumes
volume = modal.Volume.from_name("mhc-data-volume", create_if_missing=True)
model_volume = modal.Volume.from_name("gpt-oss-model-weights-ads-70439")
hf_cache_volume = modal.Volume.from_name("hf-cache-persistent", create_if_missing=True)

# Clean image - NO local source injection, but full deps
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu24.04",
        add_python="3.11",
    )
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

@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/root/data": volume,
        "/root/model": model_volume,
        "/root/hf_cache": hf_cache_volume
    },
    ephemeral_disk=512 * 1024,
    secrets=[modal.Secret.from_dict({
        "WANDB_API_KEY": "25cca6253942b7651bf7942122a976af7d5449b2",
        "HF_HUB_READ_TIMEOUT": "120",
    })],
    timeout=7200,
)
def run_clean_baseline(max_steps: int = 500):
    import os
    import torch
    import sys
    
    os.environ["HF_HOME"] = "/root/hf_cache"
    os.environ["HF_HUB_READ_TIMEOUT"] = "120"
    os.makedirs("/root/hf_cache", exist_ok=True)
    
    # Log output
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/clean_baseline.log"
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    log_file = open(log_path, "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    
    print("=" * 60)
    print("CLEAN BASELINE - NO MONKEY PATCHING")
    print("=" * 60)
    
    # Import Unsloth
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
    from datasets import load_dataset
    
    # Model path
    model_name = "/root/model/openai/gpt-oss-20b"
    max_seq_length = 32768  # Reduced for stability
    
    print(f"Loading model from {model_name}...")
    
    # Load model with Unsloth - CLEAN, no patches
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        full_finetuning=False,
    )
    
    print("Model loaded successfully!")
    
    # Apply LoRA
    print("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Set chat template
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
    
    # Load dataset
    print("Loading dataset from local volume...")
    try:
        dataset = load_dataset(
            "parquet", 
            data_files={"train": "/root/data/nemotron-harmony/data/high_part_00-*.parquet"}
        )["train"]
        print(f"Loaded {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Sample dataset
    if len(dataset) > 50000:
        dataset = dataset.select(range(50000))
        print(f"Selected 50000 samples")
    
    # Format dataset
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False) for c in convos]
        return {"text": texts}
    
    print("Formatting dataset...")
    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=8)
    
    # Data collator
    response_template = "<|start|>assistant"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )
    
    # Trainer
    print("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        data_collator=collator,
        max_seq_length=max_seq_length,
        args=SFTConfig(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            max_steps=max_steps,
            learning_rate=2e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            warmup_steps=10,
            seed=3407,
            output_dir="outputs",
            packing=False,
            report_to="wandb",
            save_strategy="no",
            run_name="gpt-oss-20b-clean-baseline",
            max_grad_norm=1.0,
        ),
    )
    
    # Train
    print("Starting training...")
    start_time = time.time()
    
    try:
        result = trainer.train()
        elapsed = time.time() - start_time
        peak_mem = torch.cuda.max_memory_reserved() / 1024**3
        
        print(f"\nTraining completed!")
        print(f"Time: {elapsed:.2f}s")
        print(f"Peak VRAM: {peak_mem:.2f}GB")
        print(f"Final loss: {result.metrics.get('train_loss', 'N/A')}")
        
        return {
            "success": True,
            "time": elapsed,
            "peak_vram_gb": peak_mem,
            "loss": result.metrics.get('train_loss'),
        }
    except Exception as e:
        print(f"\n!!! TRAINING FAILED !!!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

@app.local_entrypoint()
def main():
    print("Launching clean baseline benchmark...")
    result = run_clean_baseline.remote(max_steps=100)  # Start with 100 steps
    print(f"Result: {result}")

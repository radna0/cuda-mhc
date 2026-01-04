
import modal
import sys
import os
import time

# Define the image with necessary dependencies and the current vLLM code
vllm_image = (
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
        "pip install 'transformers>=4.57.1,!=4.57.2' sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec"
    )
    .run_commands("pip install vllm==0.13.0")
    .add_local_dir(
        "/home/kojoe/CUDA_mhc/vllm_drift/vllm",
        remote_path="/root/vllm_pkg_source",
        copy=True,
    )
    .run_commands(
        "cp -rf /root/vllm_pkg_source/* /usr/local/lib/python3.11/site-packages/vllm/"
    )
)

app = modal.App("mhc-benchmark", image=vllm_image)
model_vol = modal.Volume.from_name("gpt-oss-models", create_if_missing=True)

@app.function(
    gpu="H100", 
    timeout=1200, 
    volumes={"/root/models": model_vol}
)
def benchmark_mhc():
    import torch
    import os
    from transformers import AutoConfig
    from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig, DeviceConfig, LoadConfig, LoRAConfig
    from huggingface_hub import snapshot_download
    
    # Imports from the injected code
    # sys.path.insert(0, "/root/vllm_drift") # REMOVED
    
    # Create logs directory
    os.makedirs("/root/logs", exist_ok=True)
    log_file = "/root/logs/mhc_benchmark.log"
    
    def log(msg):
        print(msg, flush=True)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
            
    log("Starting mHC Benchmark on H100...")
    
    # Distributed Environment Init
    log("Initializing Distributed Environment...")
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    
    if not torch.distributed.is_initialized():
         init_distributed_environment(world_size=1, rank=0, distributed_init_method="env://")
    initialize_model_parallel(1, 1)

    try:
        from vllm.model_executor.models.gpt_oss import GptOssForCausalLM
        log("Successfully imported GptOssForCausalLM")
    except ImportError as e:
        log(f"CRITICAL ERROR: Failed to import GptOssForCausalLM: {e}")
        raise e

    # Step 1: Model Config
    repo_id = "openai/gpt-oss-20b"
    local_model_path = f"/root/models/{repo_id.replace('/', '--')}"
    
    if not os.path.exists(local_model_path):
        log(f"Model not found. Downloading {repo_id}...")
        snapshot_download(repo_id=repo_id, local_dir=local_model_path)
        model_vol.commit()
    else:
        log("Model found in Volume.")

    model_name = local_model_path
    hf_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
    
    # Configure mHC
    hf_config.use_mhc = True
    hf_config.num_mhc_heads = 4
    hf_config.mhc_init_identity = True
    
    # Mock VllmConfigs
    model_config = ModelConfig(
        model=model_name, tokenizer=model_name, tokenizer_mode="auto",
        trust_remote_code=True, dtype="float16", seed=42, hf_config=hf_config
    )
    
    class MockVllmConfig:
        def __init__(self):
            self.model_config = model_config
            self.quant_config = None
            self.parallel_config = ParallelConfig(1, 1, False)
            self.cache_config = CacheConfig(16, 0.9, 1)
            self.scheduler_config = SchedulerConfig(
                max_model_len=2048, is_encoder_decoder=False,
                max_num_batched_tokens=2048, max_num_seqs=256, policy="fcfs"
            )
            self.device_config = DeviceConfig("cuda")
            self.load_config = LoadConfig()
            self.lora_config = None
            self.is_attention_free = False
            
    vllm_config = MockVllmConfig()
    
    log("Initializing Model...")
    model = GptOssForCausalLM(vllm_config=vllm_config)
    model.cuda()
    log("Model Initialized.")
    
    # Benchmarking
    batch_size = 1
    seq_len = 128
    vocab_size = hf_config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).cuda()
    positions = torch.arange(0, seq_len).cuda()
    
    log(f"Starting Warmup (Batch={batch_size}, SeqLen={seq_len})...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids, positions)
    torch.cuda.synchronize()
    
    log("Starting Benchmark Loop (50 iters)...")
    start_time = time.time()
    num_iters = 50
    for i in range(num_iters):
        with torch.no_grad():
            output = model(input_ids, positions)
        if i % 10 == 0:
            print(f"Iter {i}/{num_iters}", flush=True)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = total_time / num_iters
    throughput = (batch_size * seq_len * num_iters) / total_time
    
    log(f"Benchmark Complete.")
    log(f"Total Time: {total_time:.4f}s")
    log(f"Avg Latency per Batch: {avg_latency*1000:.2f}ms")
    log(f"Throughput: {throughput:.2f} tokens/sec")
    
    return f"Benchmark Passed. Latency: {avg_latency*1000:.2f}ms"

@app.local_entrypoint()
def main():
    print("Running mHC Benchmark on Modal H100...")
    try:
        res = benchmark_mhc.remote()
        print(res)
    except Exception as e:
        print(f"FAILED: {e}")

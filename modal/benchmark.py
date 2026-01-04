import modal

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
        "pip install 'transformers>=4.57.1,!=4.57.2' sentence-transformers numpy==2.2.0 pandas polars datasets==3.2.0 scipy 'openai-harmony>=0.0.8' sentencepiece protobuf msgspec"
    )
    .run_commands("pip install vllm==0.13.0")
    .run_commands("pip install pybind11 ninja wheel setuptools")
    .add_local_dir(
        "/home/kojoe/CUDA_nvfp4/vllm_nvfp4",
        remote_path="/root/vllm_nvfp4",
        copy=True,
    )
    .run_commands(
        "PYTHONPATH=/root/vllm_nvfp4 cd /root/vllm_nvfp4 && CC=g++ CXX=g++ pip install . --no-build-isolation"
    )
)

app = modal.App("nvfp4-quant-benchmark", image=vllm_image)


@app.function(gpu="H100")
def benchmark_quantization():
    import torch
    import time
    import triton
    import triton.language as tl
    import vllm_nvfp4

    # --- TRITON KERNELS (Copied locally to avoid import issues) ---
    @triton.jit
    def quantize_nvfp4(val, scale):
        # NVFP4 magnitude set: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        # Rounded midpoints: [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
        norm_val = tl.abs(val / scale)
        sign = tl.where(val < 0, 8, 0)  # 4th bit is sign

        idx = tl.where(
            norm_val < 0.25,
            0,
            tl.where(
                norm_val < 0.75,
                1,
                tl.where(
                    norm_val < 1.25,
                    2,
                    tl.where(
                        norm_val < 1.75,
                        3,
                        tl.where(
                            norm_val < 2.5,
                            4,
                            tl.where(
                                norm_val < 3.5,
                                5,
                                tl.where(norm_val < 5.0, 6, 7),
                            ),
                        ),
                    ),
                ),
            ),
        )

        return (sign | idx).to(tl.uint8)

    @triton.jit
    def reshape_and_cache_kernel_flash(
        key_ptr,  # [num_tokens, num_heads, head_size]
        value_ptr,  # [num_tokens, num_heads, head_size]
        key_cache_ptr,  # [num_blocks, block_size, num_heads, packed_head_size]
        value_cache_ptr,  # [num_blocks, block_size, num_heads, packed_head_size]
        slot_mapping_ptr,  # [num_tokens]
        k_scale,  # float32
        v_scale,  # float32
        # strides
        key_stride_tok: tl.int64,
        key_stride_head: tl.int64,
        value_stride_tok: tl.int64,
        value_stride_head: tl.int64,
        block_stride: tl.int64,
        page_stride: tl.int64,
        head_stride: tl.int64,
        num_heads: tl.constexpr,
        head_size: tl.constexpr,
        block_size: tl.constexpr,
        # FP8 and NVFP4 flags
        FP8_KV_CACHE: tl.constexpr,
        NVFP4_KV_CACHE: tl.constexpr,
    ):
        token_idx = tl.program_id(axis=0)
        head_idx = tl.program_id(axis=1)

        slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
        if slot_idx < 0:
            return

        block_idx = slot_idx // block_size
        block_offset = slot_idx % block_size

        src_key_ptr = key_ptr + token_idx * key_stride_tok + head_idx * key_stride_head
        src_val_ptr = (
            value_ptr + token_idx * value_stride_tok + head_idx * value_stride_head
        )

        tgt_key_base = (
            key_cache_ptr
            + block_idx * block_stride
            + block_offset * page_stride
            + head_idx * head_stride
        )
        tgt_val_base = (
            value_cache_ptr
            + block_idx * block_stride
            + block_offset * page_stride
            + head_idx * head_stride
        )

        if NVFP4_KV_CACHE:
            # Block-based microscaling (16 elements per block)
            DATA_SIZE: tl.constexpr = head_size // 2
            SCALE_SIZE: tl.constexpr = head_size // 16

            # Process in blocks of 16
            for b in range(SCALE_SIZE):
                offs = b * 16 + tl.arange(0, 16)
                k_vals = tl.load(src_key_ptr + offs)
                v_vals = tl.load(src_val_ptr + offs)

                # Compute exponential scales (SGLang style: MXFP4 compatible)
                # scale = 2^ceil(log2(max_val / 6.0))
                k_exp = tl.math.ceil(tl.math.log2(tl.max(tl.abs(k_vals)) / 6.0 + 1e-10))
                v_exp = tl.math.ceil(tl.math.log2(tl.max(tl.abs(v_vals)) / 6.0 + 1e-10))

                k_s_val = tl.math.exp2(k_exp)
                v_s_val = tl.math.exp2(v_exp)

                # Store scales as 1-byte uint8 (offset by 127) at the end of the head
                # Layout: head_size//2 (data) + head_size//16 (scales)
                tl.store(tgt_key_base + DATA_SIZE + b, (k_exp + 127).to(tl.uint8))
                tl.store(tgt_val_base + DATA_SIZE + b, (v_exp + 127).to(tl.uint8))

                # Quantize and pack using separated even/odd loading to avoid Triton indexing errors
                # Even indices: 0, 2, 4, 6, 8, 10, 12, 14
                even_offs = b * 16 + tl.arange(0, 8) * 2
                k_even_vals = tl.load(src_key_ptr + even_offs)
                v_even_vals = tl.load(src_val_ptr + even_offs)

                # Odd indices: 1, 3, 5, 7, 9, 11, 13, 15
                odd_offs = even_offs + 1
                k_odd_vals = tl.load(src_key_ptr + odd_offs)
                v_odd_vals = tl.load(src_val_ptr + odd_offs)

                # Quantize even and odd values with the shared scale
                k_q_even = quantize_nvfp4(k_even_vals, k_s_val)
                k_q_odd = quantize_nvfp4(k_odd_vals, k_s_val)
                v_q_even = quantize_nvfp4(v_even_vals, v_s_val)
                v_q_odd = quantize_nvfp4(v_odd_vals, v_s_val)

                # Pack: low nibble from even, high nibble from odd (Matches SGLang)
                k_packed = (k_q_even & 0x0F) | ((k_q_odd & 0x0F) << 4)
                v_packed = (v_q_even & 0x0F) | ((v_q_odd & 0x0F) << 4)

                # Store packed bytes (8 bytes per block of 16 values)
                pack_offs = tl.arange(0, 8)
                tl.store(tgt_key_base + b * 8 + pack_offs, k_packed.to(tl.uint8))
                tl.store(tgt_val_base + b * 8 + pack_offs, v_packed.to(tl.uint8))

        elif FP8_KV_CACHE:
            tile_pos = tl.arange(0, head_size)
            tl.store(
                tgt_key_base + tile_pos,
                tl.load(src_key_ptr + tile_pos).to(tl.float8e4nv),
            )
            tl.store(
                tgt_val_base + tile_pos,
                tl.load(src_val_ptr + tile_pos).to(tl.float8e4nv),
            )
        else:
            tile_pos = tl.arange(0, head_size)
            tl.store(tgt_key_base + tile_pos, tl.load(src_key_ptr + tile_pos))
            tl.store(tgt_val_base + tile_pos, tl.load(src_val_ptr + tile_pos))

    # --- BENCHMARK ---
    device = torch.device("cuda")
    num_tokens = 4096
    num_heads = 32
    head_size = 128
    block_size = 16
    num_blocks = 1000  # Enough blocks

    # Input
    key = torch.randn(
        num_tokens, num_heads, head_size, dtype=torch.float16, device=device
    )
    value = torch.randn(
        num_tokens, num_heads, head_size, dtype=torch.float16, device=device
    )

    # Slot mapping
    slot_mapping = torch.randint(
        0, num_blocks * block_size, (num_tokens,), dtype=torch.int64, device=device
    )

    # Output caches
    packed_head_size = head_size // 2 + head_size // 16
    key_cache = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        packed_head_size,
        dtype=torch.uint8,
        device=device,
    )
    value_cache = torch.zeros(
        num_blocks,
        block_size,
        num_heads,
        packed_head_size,
        dtype=torch.uint8,
        device=device,
    )

    k_scale = torch.ones(1, device=device)
    v_scale = torch.ones(1, device=device)

    print(
        f"Benchmarking Quantization (Write Path): {num_tokens} tokens, {num_heads} heads, {head_size} head_size"
    )

    grid = (num_tokens, num_heads)

    # 1. Benchmark Triton
    for _ in range(10):
        reshape_and_cache_kernel_flash[grid](
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            k_scale,
            v_scale,
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            FP8_KV_CACHE=False,
            NVFP4_KV_CACHE=True,
        )

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        reshape_and_cache_kernel_flash[grid](
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            k_scale,
            v_scale,
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            num_heads=num_heads,
            head_size=head_size,
            block_size=block_size,
            FP8_KV_CACHE=False,
            NVFP4_KV_CACHE=True,
        )
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / 100
    print(f"Triton Soft-Quant Time: {triton_time*1000:.4f} ms")

    # 2. Benchmark CUDA
    for _ in range(10):
        vllm_nvfp4.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        vllm_nvfp4.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    torch.cuda.synchronize()
    cuda_time = (time.perf_counter() - start) / 100
    print(f"CUDA Hard-Quant Time:  {cuda_time*1000:.4f} ms")

    print(f"Speedup: {triton_time / cuda_time:.2f}x")


if __name__ == "__main__":
    with modal.App("nvfp4-quant-benchmark", image=vllm_image).run():
        benchmark_quantization.remote()

import os
import modal
from huggingface_hub import snapshot_download

app = modal.App("mhc-dataset-prepper")
volume = modal.Volume.from_name("mhc-data-volume", create_if_missing=True)

image = modal.Image.debian_slim().pip_install("huggingface_hub[hf_transfer]")

@app.function(
    image=image,
    volumes={"/root/data": volume},
    timeout=3600,
    secrets=[modal.Secret.from_dict({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_TOKEN": os.environ.get("HF_TOKEN")})]
)
def sync_dataset():
    target_dir = "/root/data/nemotron-harmony"
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Syncing radna0/nemotron-harmony-formatted to {target_dir}...")
    
    snapshot_download(
        repo_id="radna0/nemotron-harmony-formatted",
        local_dir=target_dir,
        repo_type="dataset",
        ignore_patterns=["*.jsonl", "*.md", ".git*"] # We only want the parquet files
    )
    
    volume.commit()
    print("Dataset sync complete.")

if __name__ == "__main__":
    with modal.Retrying(max_retries=3):
        sync_dataset.remote()

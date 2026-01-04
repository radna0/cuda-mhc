import modal
app = modal.App("diag")
image = modal.Image.debian_slim().pip_install("unsloth", "transformers", "trl")

@app.function(image=image)
def get_keys():
    from unsloth import CHAT_TEMPLATES
    return list(CHAT_TEMPLATES.keys())

@app.local_entrypoint()
def main():
    keys = get_keys.remote()
    print("KEYS:", keys)

import modal
app = modal.App("check-config")
vol = modal.Volume.from_name("model-openai")
@app.function(volumes={"/root/model": vol})
def check():
    import json
    import os
    path = "/root/model/openai/gpt-oss-20b/tokenizer_config.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("chat_template")
    return "NOT FOUND"
@app.local_entrypoint()
def main():
    print("TEMPLATE:", check.remote())

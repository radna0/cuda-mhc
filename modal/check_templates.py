import modal
app = modal.App("check-templates")
@app.function()
def check():
    from unsloth import CHAT_TEMPLATES
    return list(CHAT_TEMPLATES.keys())
@app.local_entrypoint()
def main():
    print(check.remote())

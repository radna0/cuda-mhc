import modal

app = modal.App("hello-world-test")

@app.function()
def hello():
    print("Hello from the cloud!")

@app.local_entrypoint()
def main():
    print("Launching hello world...")
    hello.remote()
    print("Launch complete.")

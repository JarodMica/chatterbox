import os
from huggingface_hub import snapshot_download
try:
    from beam import function, Image, Volume
except:
    print("Beam not installed, skipping import")

CHATTERBOX_PROJECT = "./chatterbox-project"

IMAGE = Image(
    python_version="python3.11",
    python_packages=[
        "huggingface_hub",
    ]
)

@function(
    image=IMAGE,
    memory="16gi",
    cpu=4,
    # gpu="T4",
    volumes=[Volume(name="chatterbox-project", mount_path=CHATTERBOX_PROJECT)],
    timeout=-1
)
def download_hf_repo():
    repo_name = "ResembleAI/chatterbox"
    repo_home_weights = os.path.join(CHATTERBOX_PROJECT, "chatterbox_weights")
    snapshot_download(repo_name, local_dir_use_symlinks=False, local_dir=repo_home_weights)

if __name__ == "__main__":
    while True:
        choice = input("1: Local\n2: Beam\n")
        if choice not in ["1", "2"]:
            print("Invalid choice")
            continue
        break

    if choice == "1":
        download_hf_repo.local()
    else:
        download_hf_repo.remote()
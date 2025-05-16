import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from huggingface_hub import snapshot_download

def download_whisper_model(model_name="openai/whisper-large-v3", local_dir="./whisper_model"):
    """
    Download the Whisper model and processor from Hugging Face and save them locally.
    """
    print(f"Downloading model '{model_name}' to '{local_dir}'...")

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    # Download the model and processor
    snapshot_download(repo_id=model_name, local_dir=local_dir, local_dir_use_symlinks=False)

    print(f"Model and processor saved to '{local_dir}'")

if __name__ == "__main__":
    # Specify the model name and local directory
    model_name = "openai/whisper-large-v3"
    local_dir = "./whisper_model"
    download_whisper_model(model_name, local_dir)
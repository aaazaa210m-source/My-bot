"""
modal_qwen_deploy.py

Ready-to-deploy Modal app for running a 4-bit quantized Qwen-14B (or other large HF model)
with a persisted SharedVolume. This file is tuned to load a quantized 4-bit model using
bitsandbytes + Transformers' quantization API (BitsAndBytesConfig / quantization_config)
and device_map="auto" so the model loads onto available GPUs. It also contains a downloader
that uses huggingface_hub.snapshot_download to save model files into the persisted volume.

Before using:
- Make sure you have the legal right to download and run the model. Accept any gated HF license.
- Add your HF token as a Modal secret named `HF_TOKEN` (see instructions below) or set env HF_TOKEN locally.
- Replace the CUDA PyTorch image with one that matches the Modal GPU CUDA version if needed.
- Ensure your Modal account can request the desired GPU (A100/other). This script requests gpu="any"; change if you need a specific SKU.

Quick run steps (summary):
1) Set HF token locally and run the downloader once (or configure Modal secret and call the local entrypoint):
   export HF_TOKEN="hf_..."
   python modal_qwen_deploy.py

2) Deploy to Modal:
   modal deploy modal_qwen_deploy.py

3) Call the endpoint (POST JSON) after deploy:
   curl -X POST -H "Content-Type: application/json" -d '{"prompt":"Hello","max_new_tokens":128}' https://<YOUR-MODAL-APP-ID>.modal.run/

"""

import os
import json
import modal

stub = modal.Stub("qwen-14b-4bit-server")
# Persisted shared volume to store model files so they are not re-downloaded on cold starts
volume = modal.SharedVolume().persisted("qwen-14b-volume")

# IMPORTANT: Use a CUDA-enabled PyTorch image compatible with the Modal GPU. Replace if needed.
# Example: pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
# You can also build a custom image with the exact CUDA + torch + bitsandbytes you require.
image = modal.Image.from_dockerhub("pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime").pip_install(
    "transformers>=4.31.0",
    "accelerate",
    "huggingface_hub",
    "sentencepiece",
    "bitsandbytes",
    "xformers",
)

# Default model repo identifier on Hugging Face. Change as required.
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "Qwen/Qwen-14B")
MODEL_MOUNT_PATH = "/vol/models"


@stub.function(
    image=image,
    shared_volumes={MODEL_MOUNT_PATH: volume},
    timeout=60 * 60,
)
def download_model(model_name: str = DEFAULT_MODEL):
    """Download model from Hugging Face into the persisted volume using snapshot_download.

    This function requires HF_TOKEN in the environment (or modal secret) with access to the model.
    """
    from huggingface_hub import snapshot_download
    import pathlib

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not set in environment. Set it locally or as a Modal secret before calling the downloader.")

    model_dirname = model_name.replace("/", "__")
    model_path = os.path.join(MODEL_MOUNT_PATH, model_dirname)
    path = pathlib.Path(model_path)
    if path.exists() and any(path.iterdir()):
        print(f"Model already exists at {model_path}")
        return model_path

    path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_name} into {model_path} ... this may be very large")

    snapshot_download(repo_id=model_name, local_dir=model_path, token=hf_token, local_dir_use_symlinks=False)

    print("Download complete")
    return model_path


@stub.function(
    image=image,
    shared_volumes={MODEL_MOUNT_PATH: volume},
    is_web_endpoint=True,
    # Request a GPU; adjust if you want a specific SKU. Modal will pick an available GPU.
    gpu="any",
    keep_warm=1,
    timeout=60 * 10,
)
def serve(request):
    """Serve HTTP POST requests. Expects JSON: {"model": "<repo-or-dir>", "prompt": "...", "max_new_tokens": 128}

    Loads model from the persisted volume and attempts a 4-bit quantized load using bitsandbytes to reduce GPU memory.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import BitsAndBytesConfig
    import os

    data = request.json()
    model_name = data.get("model", DEFAULT_MODEL)
    prompt = data.get("prompt", "Hello")
    max_new_tokens = int(data.get("max_new_tokens", 128))

    model_dirname = model_name.replace("/", "__")
    model_path = os.path.join(MODEL_MOUNT_PATH, model_dirname)

    if not os.path.exists(model_path):
        return (json.dumps({"error": "model-not-found", "message": f"Model not found at {model_path}. Run the downloader first."}), 404)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # BitsAndBytesConfig for 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Attempt quantized load (preferred). If it fails, fall back to default loading.
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print("Quantized 4-bit load failed:", e)
        print("Falling back to normal load (may OOM on GPU)")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)

    device = 0 if torch.cuda.is_available() else -1
    from transformers import pipeline

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

    outputs = gen(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_k=50)
    text = outputs[0].get("generated_text", "")

    return json.dumps({"text": text})


@stub.local_entrypoint()
def main():
    """Local helper: call this locally (with HF_TOKEN set) to download the model into the persisted Modal volume.

    Example:
      export HF_TOKEN="hf_..."
      python modal_qwen_deploy.py

    Or add HF_TOKEN as a Modal secret and call the downloader via modal CLI or UI.
    """
    print("Downloading model to Modal persisted volume. Make sure HF_TOKEN is set in your environment.")
    path = download_model.call(DEFAULT_MODEL)
    print("Downloaded to:", path)
    print("After this deploy with: modal deploy modal_qwen_deploy.py")


# NOTES FOR USERS (instructions to add as README or follow manually):
# 1) Create a Modal secret named HF_TOKEN with your Hugging Face token or set HF_TOKEN locally.
#    - To create a Modal secret via CLI: `modal secret set HF_TOKEN` and follow prompts.
#    - Or in the Modal web dashboard, add a secret named HF_TOKEN with your token value.
# 2) If the model is gated on Hugging Face, ensure the token has access (accept license on HF first).
# 3) If you need a specific GPU SKU, replace gpu="any" with gpu="A100" or the SKU your Modal account supports.
# 4) If quantized load fails due to incompatibility, check versions of bitsandbytes, CUDA, and Transformers; consider building a custom image.

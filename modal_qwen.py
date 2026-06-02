"""
modal_qwen.py

Deployment helper to run Qwen-14B on Modal.com with a persisted volume.

IMPORTANT NOTES (read before using):
- Qwen-14B is a very large model. You MUST ensure you have the legal right / license to download and run it.
  Some Qwen checkpoints on Hugging Face are gated and require accepting a license before download.
- Qwen-14B requires GPUs with large memory (A100 40/80GB or similar). For cheaper runs you must use a quantized variant
  (4-bit) or distilled/quantized checkpoints. Expect memory / cost tradeoffs.
- This file shows two approaches:
  1) Automated download from Hugging Face into the persisted Modal volume (requires HF token with access). Uses snapshot_download.
  2) If the model is gated or too large, download locally and upload the model directory into the persisted volume manually.
- The serve function requests a GPU. Edit `gpu` to match your Modal GPU SKU or set to "any" if you want Modal to pick.
- You will need a CUDA-enabled container image (PyTorch + CUDA). Replace `image` below with a CUDA-ready image matching your CUDA/PyTorch versions.

Usage outline:
1) Put your HF token in environment variable HF_TOKEN (locally) and run `python modal_qwen.py` once to download into the persisted volume.
2) Deploy: `modal deploy modal_qwen.py`
3) Call the deployed endpoint (POST JSON {"prompt": "..."})

"""

import os
import json
import modal

stub = modal.Stub("qwen-14b-server")
volume = modal.SharedVolume().persisted("my-bot-hf-volume")

# NOTE: This image must include CUDA-enabled PyTorch if you want to use GPU workers.
# Replace with a CUDA PyTorch image that matches the CUDA version of the Modal GPU instance.
# Example (replace with one compatible with your GPU and Modal):
#   "pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime"
# If you can't use a Docker image string here, use modal.Image.debian_slim().pip_install(...) for CPU tests.
image = modal.Image.from_dockerhub("pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime").pip_install(
    "transformers>=4.30.0",
    "accelerate",
    "huggingface_hub",
    "sentencepiece",
    "bitsandbytes",
    "xformers"
)

# Model repo id on HF. Example: "Qwen/Qwen-14B-Chat" or local directory name if you uploaded to the volume.
DEFAULT_MODEL = "Qwen/Qwen-14B"
MODEL_MOUNT_PATH = "/vol/models"


@stub.function(
    image=image,
    shared_volumes={MODEL_MOUNT_PATH: volume},
    timeout=60 * 60,
)
def download_qwen(model_name: str = DEFAULT_MODEL):
    """Download model snapshot from Hugging Face into the persisted volume.
    This requires HF_TOKEN in the environment with access to the model if it's gated.
    """
    from huggingface_hub import snapshot_download
    import pathlib

    hf_token = os.environ.get("HF_TOKEN")
    model_dirname = model_name.replace("/", "__")
    model_path = os.path.join(MODEL_MOUNT_PATH, model_dirname)
    path = pathlib.Path(model_path)
    if path.exists() and any(path.iterdir()):
        print(f"Model already exists at {model_path}")
        return model_path

    path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_name} into {model_path} ... this may be large and take a long time")

    # snapshot_download will download the full repo into model_path
    snapshot_download(repo_id=model_name, local_dir=model_path, token=hf_token, local_dir_use_symlinks=False)

    print("Download complete")
    return model_path


@stub.function(
    image=image,
    shared_volumes={MODEL_MOUNT_PATH: volume},
    is_web_endpoint=True,
    # Request a GPU. Set to the SKU you want or "any". Modal may offer different ways to request GPUs.
    # If your account supports it, you can put gpu="A100" or gpu="any".
    gpu="any",
    keep_warm=1,
    timeout=60 * 5,
)
def serve_qwen(request):
    """Serve endpoint for Qwen-like models.

    POST JSON: { "model": "<repo-or-dir>", "prompt": "...", "max_new_tokens": 128 }

    Notes:
    - Loads the model from the persisted volume. If weights aren't present, returns 404 with guidance.
    - Attempts to use 4-bit quantization (bitsandbytes) if possible to reduce GPU memory usage.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import os

    body = request.json()
    model_name = body.get("model", DEFAULT_MODEL)
    prompt = body.get("prompt", "Hello")
    max_new_tokens = int(body.get("max_new_tokens", 128))

    model_dirname = model_name.replace("/", "__")
    model_path = os.path.join(MODEL_MOUNT_PATH, model_dirname)

    if not os.path.exists(model_path):
        return (json.dumps({"error": "model-not-found", "message": f"Model not found at {model_path}. Run the downloader or upload the model files to the persisted volume."}), 404)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Try to load model using bitsandbytes 4-bit quantization to reduce memory.
    model = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        # Transformers >=4.31 offers BitsAndBytesConfig in transformers.tools. Use dynamic import to be safe.
        from transformers import AutoConfig
        try:
            # Try 4-bit load (this will require bitsandbytes and compatible CUDA)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                load_in_4bit=True,
                trust_remote_code=True,
            )
        except Exception as e:
            # Fall back to normal loading if quantized load fails
            print("4-bit load failed, falling back to full load:", e)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
    except Exception as e:
        # Catch-all fallback
        print("Model load error, attempting CPU load:", e)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # Create generation pipeline using the loaded model and tokenizer
    from transformers import pipeline

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    outputs = gen(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_k=50)
    text = outputs[0].get("generated_text", "")

    return json.dumps({"text": text})


@stub.local_entrypoint()
def main():
    print("This helper will attempt to download Qwen into the persisted Modal volume.")
    print("Make sure HF_TOKEN is set in your env (and you have accepted any model license on HF if required).")
    path = download_qwen.call(DEFAULT_MODEL)
    print("Downloaded to:", path)
    print("Now run: modal deploy modal_qwen.py")

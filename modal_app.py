import modal
import os
import json

# Modal app: persistent volume + web endpoint to serve a small Hugging Face model
# This app downloads the model once into a persisted SharedVolume and then serves
# generation requests from the saved model so it does not re-download on each cold start.

stub = modal.Stub("my-bot-hf-server")

# Persisted volume name inside your Modal account. Change if you want a different name.
volume = modal.SharedVolume().persisted("my-bot-hf-volume")

# Minimal image with Transformers and PyTorch. Add other deps if you need them.
image = modal.Image.debian_slim().pip_install(
    "transformers>=4.0.0", "torch", "sentencepiece", "accelerate", "aiohttp"
)

# Choose a small model for demos so it works on CPU and is cheap to store/run.
# You can replace this with any HF model id (e.g. "gpt2", a local repo in /vol, etc.).
DEFAULT_MODEL = "sshleifer/tiny-gpt2"
MODEL_MOUNT_PATH = "/vol/models"


@stub.function(
    image=image,
    shared_volumes={MODEL_MOUNT_PATH: volume},
    timeout=600,
)
def download_and_save_model(model_name: str = DEFAULT_MODEL):
    """Download model and tokenizer once and save under the shared volume.
    This function is intended to be called before deploying the web endpoint (or
    during the first deploy) so the model files are available on disk inside the volume.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import pathlib

    model_path = os.path.join(MODEL_MOUNT_PATH, model_name.replace("/", "__"))
    path = pathlib.Path(model_path)
    if path.exists():
        print(f"Model already present at {model_path}")
        return model_path

    path.mkdir(parents=True, exist_ok=True)
    print(f"Downloading model {model_name} into {model_path} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    print("Download complete.")
    return model_path


@stub.function(
    image=image,
    shared_volumes={MODEL_MOUNT_PATH: volume},
    is_web_endpoint=True,
    keep_warm=1,  # try to keep a worker warm to reduce cold starts
    timeout=60,
)
def serve(request):
    """HTTP POST endpoint.
    JSON body: { "model": "<model-id-optional>", "prompt": "...", "max_new_tokens": 50 }
    Returns JSON { "text": "generated text..." }
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import pathlib

    data = request.json()
    model_name = data.get("model", DEFAULT_MODEL)
    prompt = data.get("prompt", "Hello")
    max_new_tokens = int(data.get("max_new_tokens", 50))

    model_dirname = model_name.replace("/", "__")
    model_path = os.path.join(MODEL_MOUNT_PATH, model_dirname)

    if not os.path.exists(model_path):
        return (json.dumps({"error": "model-not-found", "message": f"Model not found at {model_path}. Call the downloader first."}), 404)

    # Load tokenizer and model from the persisted volume
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

    outputs = gen(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_k=50)
    text = outputs[0]["generated_text"]

    return json.dumps({"text": text})


@stub.local_entrypoint()
def main():
    """Run this locally to download the model into the Modal shared volume.
    It will start a Modal function that writes into the persisted volume. Run once.
    """
    print("Downloading model to the persisted Modal volume. This may take a few minutes...")
    path = download_and_save_model.call(DEFAULT_MODEL)
    print("Model saved to persisted volume at:", path)
    print("Next: run `modal deploy modal_app.py` to deploy the web endpoint on Modal.com")

import os

MODEL_PATH = os.getenv("MODEL_PATH", "")
MODEL_LABELS = os.getenv("MODEL_LABELS", "W0,W1,W2,W3,W4").split(",")
DEFAULT_CLASS = os.getenv("DEFAULT_CLASS", "W1")

# Hugging Face access token (do NOT hardcode; set env var)
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN", "")

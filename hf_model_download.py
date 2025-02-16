import argparse
import os
from typing import Optional

from dotenv import load_dotenv
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub import snapshot_download

load_dotenv(override=True)


class ModelNotFoundError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def model_download(repo_id: str, cache_dir: str, token: Optional[str] = None):
    try:
        snapshot_download(
            repo_id,
            cache_dir=cache_dir,
            token=token,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.py",
                "tokenizer.model",
                "*.tiktoken",
                "*.txt",
            ],
        )
    except:
        raise ModelNotFoundError(
            f"Model not found for path or HF repo: {repo_id}.\n"
            "Please make sure you specified the local path or Hugging Face"
            " repo id correctly.\nIf you are trying to access a private or"
            " gated Hugging Face repo, make sure you are authenticated:\n"
            "https://huggingface.co/docs/huggingface_hub/en/guides/cli#huggingface-cli-login"
        ) from None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face Model from the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        help="Path or Hugging Face Model Repository ID",
        required=True
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Hugging Face API Token",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Cache directory for the model",
        default=HF_HUB_CACHE
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    repo_id: str = args.repo_id
    token: str = args.token
    cache_dir: str = args.cache_dir

    if token is None:
        token = os.getenv("HF_TOKEN", None)

    model_download(repo_id, cache_dir, token)

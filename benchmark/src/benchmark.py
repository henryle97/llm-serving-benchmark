import argparse
import functools
import logging
import os
import sys

# Add the src directory to the Python path
from pathlib import Path

sys.path.append(str(Path(__file__).parent / ".."))

from transformers import AutoTokenizer

from src.userdef import UserDef as BaseUserDef

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    max_tokens = int(os.environ.get("MAX_TOKENS"))
except (TypeError, ValueError):
    max_tokens = 512

logger.info(f"max_tokens set to {max_tokens}")

# Use the Hugging Face API token from the environment variable
api_token = os.environ.get("HUGGINGFACE_API_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    use_auth_token=api_token,
)

default_system_prompt = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe. Your answers should not include "
    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why "
    "instead of answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)

if os.environ.get("SYSTEM_PROMPT") == "1":
    system_prompt = default_system_prompt
    system_prompt_file = os.environ.get("SYSTEM_PROMPT_FILE")
    if system_prompt_file is not None:
        with Path(system_prompt_file).open() as f:
            system_prompt = f.read().strip()
else:
    system_prompt = ""

base_url = os.environ.get("BASE_URL", "http://localhost:3000")


@functools.lru_cache(maxsize=8)
def get_prompt_set(
    min_input_length: int = 0, max_input_length: int = 500
) -> list[str]:
    """
    return a list of prompts with length between min_input_length and max_input_length
    """
    import json

    import requests

    # check if the dataset is cached
    if Path("databricks-dolly-15k.jsonl").exists():
        logger.info("Loading cached dataset")
        with Path("databricks-dolly-15k.jsonl").open() as f:
            dataset = [json.loads(line) for line in f]
    else:
        logger.info("Downloading dataset")
        raw_dataset = requests.get(
            "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
            timeout=10,
        )
        content = raw_dataset.content
        with Path("databricks-dolly-15k.jsonl").open("wb") as f:
            f.write(content)
        dataset = [json.loads(line) for line in content.decode().split("\n")]
        logger.info("Dataset downloaded")

    for d in dataset:
        d["question"] = d["context"] + d["instruction"]
        d["input_tokens"] = len(tokenizer(d["question"])["input_ids"])
        d["output_tokens"] = len(tokenizer(d["response"]))
    return [
        d["question"]
        for d in dataset
        if min_input_length <= d["input_tokens"] <= max_input_length
    ]


prompts = get_prompt_set(30, 150)


class UserDef(BaseUserDef):
    BASE_URL = base_url
    PROMPTS = prompts

    @classmethod
    def make_request(cls) -> tuple[str, dict, str]:
        import json
        import random

        prompt = random.choice(cls.PROMPTS)
        headers = {"Content-Type": "application/json"}
        url = f"{cls.BASE_URL}/generate"
        data = {
            "prompt": prompt,
            # this is important because there's a default system prompt
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
        }
        return url, headers, json.dumps(data)

    @staticmethod
    def parse_response(chunk: bytes) -> list[int]:
        text = chunk.decode("utf-8").strip()
        return tokenizer.encode(text, add_special_tokens=False)


if __name__ == "__main__":
    import asyncio

    from .metric_collector import start_benchmark_session

    # arg parsing
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--max_users", type=int, required=True)
    parser.add_argument("--session_time", type=float, default=None)
    parser.add_argument("--ping_correction", action="store_true")
    args = parser.parse_args()

    asyncio.run(start_benchmark_session(args, UserDef))

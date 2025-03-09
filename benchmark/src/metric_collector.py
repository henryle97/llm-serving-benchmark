import argparse
import asyncio
import collections
import contextlib
import functools
import logging
import math
import time

import aiohttp
import numpy as np

from src.user import UserDef
from src.user_spawer import UserSpawner

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCollector:
    def __init__(
        self,
        user_def: dict,
        session_time: float | None = None,
        ping_latency: float = 0.0,
    ) -> None:
        self.start_time = math.floor(time.time())
        self.response_word_bucket = collections.defaultdict(int)
        self.response_head_latency_bucket = collections.defaultdict(list)
        self.response_latency_bucket = collections.defaultdict(list)
        self.on_going_requests = 0
        self.response_bucket = collections.defaultdict(int)
        self.total_requests = 0
        self.on_going_users = 0
        self.status_bucket = collections.defaultdict(int)
        self.user_def = user_def
        self.session_time = session_time
        self.ping_latency = ping_latency

    def collect_response_chunk(self, chunk: list) -> None:
        self.response_word_bucket[math.floor(time.time())] += len(chunk)

    def collect_response_status(self, status: str) -> None:
        self.status_bucket[status] += 1

    def collect_response_head_latency(self, latency: float) -> None:
        self.response_head_latency_bucket[math.floor(time.time())] += [
            latency - self.ping_latency
        ]

    @contextlib.contextmanager
    def collect_http_request(self) -> collections.abc.Generator:
        start_time = time.time()
        self.on_going_requests += 1
        yield
        self.on_going_requests -= 1
        self.response_bucket[math.floor(time.time())] += 1
        self.response_latency_bucket[math.floor(time.time())] += [
            time.time() - start_time - self.ping_latency
        ]

    @contextlib.contextmanager
    def collect_user(self) -> collections.abc.Generator:
        self.on_going_users += 1
        yield
        self.on_going_users -= 1

    async def report_loop(self, time_window: int = 5) -> None:
        """
        Each bucket is in 1s. This function will report the avg metrics in the past time_window seconds.
        """
        while True:
            await asyncio.sleep(time_window)
            now = math.floor(time.time())
            logger.info(f"Time: {now - self.start_time}")
            logger.info(f"Active Users: {self.on_going_users}")
            logger.info(
                f"Request/s: {sum(self.response_bucket[i] for i in range(now - time_window, now)) / time_window}"
            )
            logger.info(f"Total Requests: {self.total_requests}")
            logger.info(f"Active Requests: {self.on_going_requests}")
            latency_bucket = [
                j
                for i in range(now - time_window, now)
                for j in self.response_head_latency_bucket[i]
            ]
            if latency_bucket:
                logger.info(
                    f"Response Head Latency: {np.mean(latency_bucket)}"
                )
            latency_bucket = [
                j
                for i in range(now - time_window, now)
                for j in self.response_latency_bucket[i]
            ]
            if latency_bucket:
                logger.info(f"Response Latency: {np.mean(latency_bucket)}")
            logger.info(
                f"Response Tokens/s: {sum(self.response_word_bucket[i] for i in range(now - time_window, now)) / time_window}"
            )
            logger.info(f"Status: {self.status_bucket}")
            logger.info("")

            if (
                self.session_time
                and now - self.start_time >= self.session_time
            ):
                self.report_final()
                break

    def report_final(self) -> None:
        logger.info("=================== Final Report ====================")
        logger.info(f"Total Requests: {self.total_requests}")
        logger.info(
            f"Average Request/s: {self.total_requests / (time.time() - self.start_time)}"
        )

        head_latency_size = sum(
            len(i) for i in self.response_head_latency_bucket.values()
        )
        if head_latency_size:
            head_latencies = [
                j
                for i in self.response_head_latency_bucket.values()
                for j in i
            ]

            logger.info(
                f"Average Response Head Latency: {sum(head_latencies) / head_latency_size}"
            )
            logger.info(
                f"Median Response Head Latency: {np.percentile(head_latencies, 50)}"
            )
            logger.info(
                f"95% Response Head Latency: {np.percentile(head_latencies, 95)}"
            )
            logger.info(
                f"99% Response Head Latency: {np.percentile(head_latencies, 99)}"
            )

        latency_size = sum(
            len(i) for i in self.response_latency_bucket.values()
        )
        if latency_size:
            latencies = [
                j for i in self.response_latency_bucket.values() for j in i
            ]
            logger.info(
                f"Average Response Latency: {sum(latencies) / latency_size}"
            )
            logger.info(
                f"Median Response Latency: {np.percentile(latencies, 50)}"
            )
            logger.info(
                f"95% Response Latency: {np.percentile(latencies, 95)}"
            )
            logger.info(
                f"99% Response Latency: {np.percentile(latencies, 99)}"
            )

        logger.info(
            f"Average Response Tokens/s: {sum(self.response_word_bucket.values()) / (time.time() - self.start_time)}"
        )


async def start_benchmark_session(
    args: argparse.Namespace, user_def: UserDef
) -> int:
    # ping server
    response_times = []
    async with aiohttp.ClientSession() as session:
        async with session.get(user_def.ping_url()) as response:
            assert response.status == 200
        await asyncio.sleep(0.3)

        for _ in range(5):
            time_start = time.time()
            async with session.get(user_def.ping_url()) as response:
                assert response.status == 200
            response_times.append(time.time() - time_start)
            await asyncio.sleep(0.3)
    ping_latency = sum(response_times) / len(response_times)
    logger.info(
        f"Ping latency: {ping_latency}. ping correction: {args.ping_correction}"
    )

    # init
    collector = MetricsCollector(
        user_def,
        args.session_time,
        ping_latency - 0.005 if args.ping_correction else 0,
    )
    user_spawner = UserSpawner(
        user_def, collector, args.max_users, target_time=time.time() + 20
    )
    spawner_task = asyncio.create_task(user_spawner.spawner_loop())
    await spawner_task
    report_task = asyncio.create_task(collector.report_loop())
    await report_task
    if args.max_users is None:
        aimd_task = asyncio.create_task(user_spawner.aimd_loop())
        await aimd_task

    if args.session_time is not None:
        await asyncio.sleep(args.session_time + 1)
    else:
        await asyncio.wait(user_spawner.user_list)

    await user_spawner.cancel_all_users()
    return 0


@functools.lru_cache(maxsize=1)
def get_tokenizer() -> callable:
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    def _tokenizer(text: str) -> list[int]:
        return tokenizer(text)["input_ids"][1:]

    return _tokenizer


@functools.lru_cache(maxsize=8)
def get_prompt_set(
    min_input_length: int = 0, max_input_length: int = 500
) -> list[str]:
    """
    return a list of prompts with length between min_input_length and max_input_length
    """
    import json
    from pathlib import Path

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

    tokenizer = get_tokenizer()
    for d in dataset:
        d["input_tokens"] = len(tokenizer(d["instruction"]))
        d["output_tokens"] = len(tokenizer(d["response"]))
    return [
        d["instruction"]
        for d in dataset
        if min_input_length <= d["input_tokens"] <= max_input_length
    ]

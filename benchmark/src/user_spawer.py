import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING

import aiohttp

from .utils import MetricsCollector, linear_regression

if TYPE_CHECKING:
    from asyncio.tasks import Task
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserSpawner:
    def __init__(
        self,
        user_def: dict,
        collector: MetricsCollector,
        target_user_count: int | None = None,
        target_time: int | None = None,
    ) -> None:
        self.target_user_count = (
            1 if target_user_count is None else target_user_count
        )
        self.target_time = (
            time.time() + 10 if target_time is None else target_time
        )

        self.data_collector = collector
        self.user_def = user_def

        self.user_list: list[Task] = []

    async def sync(self) -> None:
        while True:
            if self.current_user_count == self.target_user_count:
                return
            await asyncio.sleep(0.1)

    @property
    def current_user_count(self) -> int:
        return len(self.user_list)

    async def user_loop(self) -> None:
        with self.data_collector.collect_user():
            cookie_jar = aiohttp.DummyCookieJar()
            try:
                async with aiohttp.ClientSession(
                    cookie_jar=cookie_jar
                ) as session:
                    while True:
                        url, headers, data = self.user_def.make_request()
                        self.data_collector.total_requests += 1
                        with self.data_collector.collect_http_request():
                            req_start = time.time()
                            async with session.post(
                                url,
                                headers=headers,
                                data=data,
                            ) as response:
                                self.data_collector.collect_response_status(
                                    response.status
                                )
                                try:
                                    if response.status != 200:
                                        continue

                                    first = True
                                    async for (
                                        data,
                                        end_of_http_chunk,
                                    ) in response.content.iter_chunks():
                                        result = self.user_def.parse_response(
                                            data
                                        )
                                        if first:
                                            first = False
                                            self.data_collector.collect_response_head_latency(
                                                time.time() - req_start
                                            )

                                        self.data_collector.collect_response_chunk(
                                            result
                                        )
                                        if not end_of_http_chunk:
                                            break
                                except Exception as e:
                                    self.data_collector.collect_response_status(
                                        str(e)
                                    )
                                    raise
                        await self.user_def.rest()
            except asyncio.CancelledError:
                pass

    def spawn_user(self) -> None:
        self.user_list.append(asyncio.create_task(self.user_loop()))

    async def cancel_all_users(self) -> None:
        try:
            user = self.user_list.pop()
            user.cancel()
        except IndexError:
            pass
        await asyncio.sleep(0)

    async def spawner_loop(self) -> None:
        while True:
            current_users = len(self.user_list)
            if current_users == self.target_user_count:
                await asyncio.sleep(0.1)
            elif current_users < self.target_user_count:
                self.spawn_user()
                sleep_time = max(
                    (self.target_time - time.time())
                    / (self.target_user_count - current_users),
                    0,
                )
                await asyncio.sleep(sleep_time)
            elif current_users > self.target_user_count:
                self.user_list.pop().cancel()
                sleep_time = max(
                    (time.time() - self.target_time)
                    / (current_users - self.target_user_count),
                    0,
                )
                await asyncio.sleep(sleep_time)

    async def aimd_loop(
        self,
        adjust_interval: int = 5,
        sampling_interval: int = 5,
        ss_delta: int = 1,
    ) -> None:
        """
        Detect a suitable number of users to maximize the words/s.
        """
        while True:
            while True:
                # slow start
                now = math.floor(time.time())
                words_per_seconds = [
                    self.data_collector.response_word_bucket[i]
                    for i in range(now - sampling_interval, now)
                ]
                slope = linear_regression(
                    range(len(words_per_seconds)), words_per_seconds
                )[0]
                if slope >= -0.01:
                    # throughput is increasing
                    cwnd = self.current_user_count
                    target_cwnd = max(int(cwnd * (1 + ss_delta)), cwnd + 1)
                    self.target_user_count = target_cwnd
                    self.target_time = time.time() + adjust_interval
                    logger.info(f"SS: {cwnd} -> {target_cwnd}")
                    await asyncio.sleep(adjust_interval)
                else:
                    # throughput is decreasing, stop slow start
                    cwnd = self.current_user_count
                    target_cwnd = math.ceil(cwnd * 0.5)
                    self.target_user_count = target_cwnd
                    self.target_time = time.time() + adjust_interval
                    logger.info(f"SS Ended: {target_cwnd}")
                    break

            await self.sync()
            await asyncio.sleep(min(adjust_interval, sampling_interval, 10))
            return 0

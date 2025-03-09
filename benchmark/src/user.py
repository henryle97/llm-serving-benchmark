import asyncio


class UserDef:
    def __init__(self, base_url: str, prompts: list[str]) -> None:
        self.BASE_URL = base_url
        self.PROMPTS = prompts

    @classmethod
    def ping_url(cls) -> str:
        return f"{cls.BASE_URL}/healthz"

    @staticmethod
    async def rest() -> None:
        await asyncio.sleep(0.01)

    @classmethod
    def make_request(
        cls, system_prompt: str, max_tokens: int
    ) -> tuple[str, dict, str]:
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
    def parse_response(chunk: bytes, tokenizer: object) -> list[int]:
        text = chunk.decode("utf-8").strip()
        return tokenizer.encode(text, add_special_tokens=False)

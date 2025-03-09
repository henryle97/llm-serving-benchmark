import asyncio


class UserDef:
    BASE_URL = ""

    @classmethod
    def ping_url(cls) -> str:
        return f"{cls.BASE_URL}/healthz"

    @staticmethod
    async def rest() -> None:
        await asyncio.sleep(0.01)

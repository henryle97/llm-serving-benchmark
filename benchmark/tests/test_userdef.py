import pytest

from benchmark.src.userdef import UserDef


def test_ping_url() -> None:
    UserDef.BASE_URL = "http://example.com"
    assert UserDef.ping_url() == "http://example.com/healthz"


def test_ping_url_empty_base() -> None:
    UserDef.BASE_URL = ""
    assert UserDef.ping_url() == "/healthz"


@pytest.mark.asyncio
async def test_rest() -> None:
    result = await UserDef.rest()
    assert result is None

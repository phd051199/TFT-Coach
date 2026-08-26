from __future__ import annotations

import asyncio
from typing import Any

import httpx


USER_AGENT = "HexCoach/0.1 (+TFT analytics; contact via local project)"


class PublicHttp:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/html;q=0.9,*/*;q=0.8"},
            timeout=25.0,
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def json(self, url: str, **kwargs: Any) -> Any:
        response = await self.client.get(url, **kwargs)
        response.raise_for_status()
        if self.delay:
            await asyncio.sleep(self.delay)
        return response.json()

    async def text(self, url: str, **kwargs: Any) -> str:
        response = await self.client.get(url, **kwargs)
        response.raise_for_status()
        if self.delay:
            await asyncio.sleep(self.delay)
        return response.text

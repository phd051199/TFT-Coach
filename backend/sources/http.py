from __future__ import annotations

import asyncio
from typing import Any

import httpx


USER_AGENT = "HexCoach/0.1 (+TFT analytics; contact via local project)"


class PublicHttp:
    def __init__(self, delay: float = 0.0, retries: int = 4) -> None:
        self.delay = delay
        self.retries = retries
        self.client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/html;q=0.9,*/*;q=0.8"},
            timeout=25.0,
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        last: httpx.Response | None = None
        for attempt in range(self.retries + 1):
            response = await self.client.get(url, **kwargs)
            last = response
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                if self.delay:
                    await asyncio.sleep(self.delay)
                return response
            if attempt >= self.retries:
                break
            retry_after = response.headers.get("Retry-After")
            try:
                wait = min(12.0, float(retry_after)) if retry_after else min(12.0, 1.25 * (2**attempt))
            except ValueError:
                wait = min(12.0, 1.25 * (2**attempt))
            await asyncio.sleep(max(0.5, wait))
        assert last is not None
        last.raise_for_status()
        return last

    async def json(self, url: str, **kwargs: Any) -> Any:
        response = await self._get(url, **kwargs)
        return response.json()

    async def text(self, url: str, **kwargs: Any) -> str:
        response = await self._get(url, **kwargs)
        return response.text

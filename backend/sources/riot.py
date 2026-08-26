from __future__ import annotations

import asyncio
import os
from typing import Any

from .base import TrainingSample
from .http import USER_AGENT

import httpx


class RiotHighEloSource:
    """Ground-truth high-Elo final boards from Riot TFT Match-V1.

    Requires a server-side RIOT_API_KEY. The collector samples Challenger -> GM -> Master,
    deduplicates matches, and emits one supervised board per ranked participant.
    """

    name = "riot-high-elo"

    def __init__(self) -> None:
        self.key = os.getenv("RIOT_API_KEY")
        self.platform = os.getenv("RIOT_PLATFORM", "vn2")
        self.region = os.getenv("RIOT_REGION", "sea")
        self.max_players = int(os.getenv("RIOT_MAX_PLAYERS", "24"))
        self.matches_per_player = int(os.getenv("RIOT_MATCHES_PER_PLAYER", "12"))
        self.delay = float(os.getenv("RIOT_REQUEST_DELAY_MS", "1250")) / 1000.0
        self.weight = float(os.getenv("HEXCOACH_WEIGHT_RIOT", "1.00"))
        self.client: httpx.AsyncClient | None = None

    async def _get(self, host: str, path: str) -> Any:
        assert self.client is not None and self.key
        while True:
            response = await self.client.get(
                f"https://{host}{path}",
                headers={"X-Riot-Token": self.key, "User-Agent": USER_AGENT},
            )
            if response.status_code != 429:
                response.raise_for_status()
                await asyncio.sleep(self.delay)
                return response.json()
            retry = max(1.0, float(response.headers.get("Retry-After", "2")))
            await asyncio.sleep(retry)

    async def collect(self) -> list[TrainingSample]:
        if not self.key or self.weight <= 0:
            return []
        self.client = httpx.AsyncClient(timeout=25.0, follow_redirects=True)
        try:
            platform_host = f"{self.platform}.api.riotgames.com"
            entries: list[dict] = []
            for endpoint in ("challenger", "grandmaster", "master"):
                payload = await self._get(platform_host, f"/tft/league/v1/{endpoint}")
                entries.extend(payload.get("entries") or [])
            entries.sort(key=lambda row: int(row.get("leaguePoints") or 0), reverse=True)
            entries = entries[: self.max_players]

            puuids: list[str] = []
            for entry in entries:
                puuid = entry.get("puuid")
                if not puuid and entry.get("summonerId"):
                    summoner = await self._get(
                        platform_host,
                        f"/tft/summoner/v1/summoners/{entry['summonerId']}",
                    )
                    puuid = summoner.get("puuid")
                if puuid:
                    puuids.append(puuid)

            match_ids: set[str] = set()
            match_host = f"{self.region}.api.riotgames.com"
            for puuid in puuids:
                ids = await self._get(
                    match_host,
                    f"/tft/match/v1/matches/by-puuid/{puuid}/ids?start=0&count={self.matches_per_player}",
                )
                match_ids.update(ids)

            samples: list[TrainingSample] = []
            for match_id in match_ids:
                match = await self._get(match_host, f"/tft/match/v1/matches/{match_id}")
                info = match.get("info") or {}
                if int(info.get("tft_set_number") or 0) != 18:
                    continue
                version = str(info.get("game_version") or "")
                for participant in info.get("participants") or []:
                    placement = int(participant.get("placement") or 0)
                    units = participant.get("units") or []
                    unit_ids = [unit.get("character_id") for unit in units if unit.get("character_id")]
                    if placement < 1 or placement > 8 or len(unit_ids) < 4:
                        continue
                    item_holders: dict[str, list[str]] = {}
                    all_items: list[str] = []
                    for unit in units:
                        unit_id = unit.get("character_id")
                        item_names = [item for item in unit.get("itemNames") or [] if item]
                        if unit_id and item_names:
                            item_holders[unit_id] = item_names
                            all_items.extend(item_names)
                    traits = [
                        trait.get("name")
                        for trait in participant.get("traits") or []
                        if trait.get("name") and int(trait.get("style") or 0) > 0
                    ]
                    samples.append(
                        TrainingSample(
                            source=self.name,
                            patch=version,
                            region=self.platform.upper(),
                            units=unit_ids,
                            items=all_items,
                            item_holders=item_holders,
                            traits=traits,
                            level=int(participant.get("level") or len(unit_ids)),
                            avg_placement=float(placement),
                            top4_rate=1.0 if placement <= 4 else 0.0,
                            win_rate=1.0 if placement == 1 else 0.0,
                            games=1,
                            source_weight=self.weight,
                        )
                    )
            return samples
        finally:
            if self.client:
                await self.client.aclose()

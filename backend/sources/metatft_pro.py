from __future__ import annotations

import os
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from .base import TrainingSample
from .http import PublicHttp


class MetaTFTProSource:
    """Current Set 18 match histories of known competitive/pro players.

    MetaTFT exposes a public pro-player directory and public profile match history. This is
    intentionally collected separately from aggregate comp clusters so training can give
    individual high-level games a different reliability weight and preserve their region.
    """

    name = "metatft-pro-live"
    directory_url = "https://api.metatft.com/public/pro_players"
    cache_dir = Path(__file__).resolve().parents[2] / "backend" / "data" / "lobby-cache"

    def __init__(self) -> None:
        # Individual pro games are highly valuable behavior signals but noisy placement
        # labels. Keep them below aggregate-source weight so a small hot streak cannot bend
        # the global strength scale.
        self.weight = float(os.getenv("HEXCOACH_WEIGHT_METATFT_PRO", "0.78"))
        self.max_profiles = int(os.getenv("HEXCOACH_PRO_MAX_PROFILES", "96"))
        self.max_matches = int(os.getenv("HEXCOACH_PRO_MATCHES_PER_PLAYER", "40"))
        self.max_lobbies = int(os.getenv("HEXCOACH_PRO_MAX_LOBBIES", "320"))
        self.region_counts: dict[str, int] = {}
        self.diagnostics: dict[str, int] = {}

    @staticmethod
    def _usable_aliases(player: dict) -> list[dict]:
        aliases = []
        for alias in player.get("aliases") or []:
            region = str(alias.get("region") or "").lower()
            riot_id = str(alias.get("riot_id") or "")
            if not region or not riot_id or "#" not in riot_id:
                continue
            if region.startswith("loltmnt") or "esportstmnt" in region:
                continue
            aliases.append(alias)
        # Prefer real regional ladders with strong competitive populations, especially KR.
        priority = {"kr": 0, "kr1": 0, "tw2": 1, "vn2": 1, "jp1": 1, "sg2": 1, "euw1": 2, "na1": 2}
        aliases.sort(key=lambda row: priority.get(str(row.get("region") or "").lower(), 3))
        return aliases

    async def _lobby_payload(self, http: PublicHttp, url: str, match_id: str) -> dict:
        cache_path = self.cache_dir / f"{match_id}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text("utf-8"))
            except Exception:
                pass
        payload = await http.json(url)
        info = payload.get("info") or {}
        if int(info.get("tft_set_number") or 0) == 18:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
        return payload

    async def collect(self) -> list[TrainingSample]:
        if self.weight <= 0:
            return []
        http = PublicHttp(delay=0.06)
        lobby_http = PublicHttp(delay=0.45, retries=2)
        try:
            directory = await http.json(self.directory_url)
            players = list(directory.get("data") or [])
            # Sort by most recent/current competitive points if present; still retain KR/APAC
            # priority through alias selection below.
            players.sort(
                key=lambda player: max(
                    [int(point.get("points") or 0) for point in player.get("points") or []] or [0]
                ),
                reverse=True,
            )

            selected: list[tuple[dict, dict]] = []
            seen_riot_ids: set[tuple[str, str]] = set()
            # First pass guarantees regional diversity rather than filling all slots with one
            # circuit. Mainland CN has no normal Riot global ladder endpoint, so TW/KR/APAC are
            # kept distinct and never mislabeled as CN.
            buckets: dict[str, list[tuple[dict, dict]]] = {}
            for player in players:
                aliases = self._usable_aliases(player)
                if not aliases:
                    continue
                alias = aliases[0]
                region = str(alias["region"]).lower()
                buckets.setdefault(region, []).append((player, alias))
            preferred_regions = ["kr", "kr1", "tw2", "vn2", "jp1", "sg2", "euw1", "na1"]
            while len(selected) < self.max_profiles:
                progressed = False
                for region in preferred_regions + sorted(set(buckets) - set(preferred_regions)):
                    bucket = buckets.get(region) or []
                    if not bucket:
                        continue
                    player, alias = bucket.pop(0)
                    key = (region, str(alias.get("riot_id")))
                    if key in seen_riot_ids:
                        continue
                    seen_riot_ids.add(key)
                    selected.append((player, alias))
                    progressed = True
                    if len(selected) >= self.max_profiles:
                        break
                if not progressed:
                    break

            samples: list[TrainingSample] = []
            seen_matches: set[tuple[str, str]] = set()
            expanded_lobbies: set[str] = set()
            lobby_success = 0
            lobby_failures = 0
            lobby_rows = 0
            self.region_counts = {}
            for player, alias in selected:
                region = str(alias["region"]).lower()
                riot_id = str(alias["riot_id"])
                name, tag = riot_id.split("#", 1)
                url = (
                    f"https://api.metatft.com/public/profile/lookup_by_riotid/{quote(region)}/"
                    f"{quote(name, safe='')}/{quote(tag, safe='')}"
                )
                try:
                    profile = await http.json(
                        url,
                        params={"source": "full_profile", "tft_set": "TFTSet18"},
                    )
                except Exception:
                    continue
                matches = list(profile.get("matches") or [])[: self.max_matches]
                for match in matches:
                    if match.get("tft_set") != "TFTSet18" or int(match.get("queue_id") or 0) != 1100:
                        continue
                    patch = str(match.get("patch") or "")
                    if not patch.startswith("18."):
                        continue
                    match_id = str(match.get("riot_match_id") or "")
                    dedupe = (match_id, riot_id)
                    if not match_id or dedupe in seen_matches:
                        continue
                    seen_matches.add(dedupe)
                    placement = int(match.get("placement") or 0)
                    timestamp_ms = int(match.get("match_timestamp") or 0)
                    age_hours = None
                    if timestamp_ms > 0:
                        played = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)
                        age_hours = max(0.0, (datetime.now(UTC) - played).total_seconds() / 3600.0)
                    summary = match.get("summary") or {}
                    units = list(summary.get("units") or [])
                    unit_ids = [str(unit.get("character_id")) for unit in units if unit.get("character_id")]
                    if placement < 1 or placement > 8 or len(unit_ids) < 4:
                        continue
                    item_holders: dict[str, list[str]] = {}
                    all_items: list[str] = []
                    for unit in units:
                        unit_id = str(unit.get("character_id") or "")
                        item_names = [str(item) for item in unit.get("itemNames") or [] if item]
                        if unit_id and item_names:
                            item_holders[unit_id] = item_names
                            all_items.extend(item_names)
                    samples.append(
                        TrainingSample(
                            source=self.name,
                            patch=patch,
                            region=region.upper(),
                            units=unit_ids,
                            items=all_items,
                            item_holders=item_holders,
                            traits=[str(value) for value in summary.get("traits") or [] if value],
                            level=int(summary.get("level") or len(unit_ids)),
                            avg_placement=float(placement),
                            top4_rate=1.0 if placement <= 4 else 0.0,
                            win_rate=1.0 if placement == 1 else 0.0,
                            games=1,
                            source_weight=self.weight,
                            sample_kind="pro_final_board",
                            context_id=match_id,
                            age_hours=age_hours,
                        )
                    )
                    for unit_id, item_names in item_holders.items():
                        if len(item_names) < 2:
                            continue
                        samples.append(
                            TrainingSample(
                                source=self.name,
                                patch=patch,
                                region=region.upper(),
                                units=[unit_id],
                                items=item_names,
                                item_holders={unit_id: item_names},
                                level=int(summary.get("level") or len(unit_ids)),
                                avg_placement=float(placement),
                                top4_rate=1.0 if placement <= 4 else 0.0,
                                win_rate=1.0 if placement == 1 else 0.0,
                                games=1,
                                # An item build inherits the whole-match placement label, so it
                                # is informative but more confounded than the final-board row.
                                source_weight=self.weight * 0.58,
                                sample_kind="pro_item_holder_build",
                                context_id=f"{match_id}:{unit_id}",
                                age_hours=age_hours,
                            )
                        )

                    # Expand each unique pro match to the complete high-Elo lobby. The public
                    # match payload is Riot Match-V1-shaped and contains all 8 participants.
                    # This multiplies useful behavior data without treating the extra players
                    # as "pros" or counting the same lobby twice when two pros share a game.
                    match_data_url = str(match.get("match_data_url") or "")
                    if match_data_url and match_id not in expanded_lobbies and len(expanded_lobbies) < self.max_lobbies:
                        expanded_lobbies.add(match_id)
                        try:
                            lobby = await self._lobby_payload(lobby_http, match_data_url, match_id)
                        except Exception:
                            lobby_failures += 1
                            lobby = {}
                        info = lobby.get("info") or {}
                        if int(info.get("tft_set_number") or 0) == 18 and int(info.get("queue_id") or info.get("queueId") or 0) == 1100:
                            lobby_success += 1
                            profile_puuid = str((profile.get("summoner") or {}).get("puuid") or "")
                            for participant in info.get("participants") or []:
                                participant_puuid = str(participant.get("puuid") or "")
                                if participant_puuid and participant_puuid == profile_puuid:
                                    continue
                                lobby_placement = int(participant.get("placement") or 0)
                                lobby_units = list(participant.get("units") or [])
                                lobby_unit_ids = [
                                    str(unit.get("character_id")) for unit in lobby_units if unit.get("character_id")
                                ]
                                if lobby_placement < 1 or lobby_placement > 8 or len(lobby_unit_ids) < 4:
                                    continue
                                lobby_holders: dict[str, list[str]] = {}
                                lobby_items: list[str] = []
                                for unit in lobby_units:
                                    unit_id = str(unit.get("character_id") or "")
                                    item_names = [str(item) for item in unit.get("itemNames") or [] if item]
                                    if unit_id and item_names:
                                        lobby_holders[unit_id] = item_names
                                        lobby_items.extend(item_names)
                                lobby_context = f"{match_id}:{participant_puuid or lobby_placement}"
                                samples.append(
                                    TrainingSample(
                                        source="metatft-high-elo-lobby",
                                        patch=patch,
                                        region=region.upper(),
                                        units=lobby_unit_ids,
                                        items=lobby_items,
                                        item_holders=lobby_holders,
                                        traits=[
                                            str(trait.get("name"))
                                            for trait in participant.get("traits") or []
                                            if trait.get("name") and int(trait.get("style") or 0) > 0
                                        ],
                                        level=int(participant.get("level") or len(lobby_unit_ids)),
                                        avg_placement=float(lobby_placement),
                                        top4_rate=1.0 if lobby_placement <= 4 else 0.0,
                                        win_rate=1.0 if lobby_placement == 1 else 0.0,
                                        games=1,
                                        source_weight=self.weight * 0.92,
                                        sample_kind="high_elo_final_board",
                                        context_id=lobby_context,
                                        age_hours=age_hours,
                                    )
                                )
                                lobby_rows += 1
                                for unit_id, item_names in lobby_holders.items():
                                    if len(item_names) < 2:
                                        continue
                                    samples.append(
                                        TrainingSample(
                                            source="metatft-high-elo-lobby",
                                            patch=patch,
                                            region=region.upper(),
                                            units=[unit_id],
                                            items=item_names,
                                            item_holders={unit_id: item_names},
                                            level=int(participant.get("level") or len(lobby_unit_ids)),
                                            avg_placement=float(lobby_placement),
                                            top4_rate=1.0 if lobby_placement <= 4 else 0.0,
                                            win_rate=1.0 if lobby_placement == 1 else 0.0,
                                            games=1,
                                            source_weight=self.weight * 0.50,
                                            sample_kind="high_elo_item_holder_build",
                                            context_id=f"{lobby_context}:{unit_id}",
                                            age_hours=age_hours,
                                        )
                                    )
                                    lobby_rows += 1
                    self.region_counts[region.upper()] = self.region_counts.get(region.upper(), 0) + 1
            self.diagnostics = {
                "lobbiesAttempted": len(expanded_lobbies),
                "lobbiesSucceeded": lobby_success,
                "lobbiesFailed": lobby_failures,
                "lobbySamples": lobby_rows,
            }
            return samples
        finally:
            await http.close()
            await lobby_http.close()


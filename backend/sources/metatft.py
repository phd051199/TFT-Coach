from __future__ import annotations

from datetime import UTC, datetime
import os
from typing import Any

from .base import TrainingSample
from .http import PublicHttp


class MetaTFTSource:
    """Collect current/live Set 18 cluster options from MetaTFT public endpoints.

    We intentionally use only endpoints referenced by their public web client. No login,
    CAPTCHA bypass, cookie replay, or private credentials are involved.
    """

    name = "metatft"
    base = "https://api-hc.metatft.com/tft-comps-api"

    def __init__(self) -> None:
        self.weight = float(os.getenv("HEXCOACH_WEIGHT_METATFT", "0.90"))
        self.snapshot: dict[str, Any] = {}
        self.live_patch = "18.1"

    @staticmethod
    def _split(value: str | None, separator: str = "&") -> list[str]:
        return [part.strip() for part in (value or "").split(separator) if part.strip()]

    async def collect(self) -> list[TrainingSample]:
        if self.weight <= 0:
            return []
        http = PublicHttp(delay=0.04)
        try:
            # Verify that Set 18 is present on the live ranked queue and discover the newest
            # live patch. MetaTFT's day feed can contain the tail of the previous patch, so
            # only 18.x rows are considered here.
            try:
                games_payload = await http.json("https://api-hc.metatft.com/tft-stat-api/games?days=1")
                patches: list[tuple[int, int]] = []
                for row in games_payload.get("games") or []:
                    srq = list(row.get("srq") or [])
                    patch = str((row.get("patch") or [""])[0] or "")
                    if len(srq) < 3 or str(srq[-1]) != "1100" or not patch.startswith("18."):
                        continue
                    try:
                        major, minor = patch.split(".", 1)
                        patches.append((int(major), int(minor)))
                    except ValueError:
                        continue
                if patches:
                    latest = max(patches)
                    self.live_patch = f"{latest[0]}.{latest[1]}"
            except Exception:
                # Cluster validation below remains mandatory; patch discovery is enrichment.
                pass

            # IMPORTANT: no queue=PBE here. The user explicitly wants current/live data.
            cluster_payload = await http.json(f"{self.base}/latest_cluster_info")
            cluster_info = cluster_payload.get("cluster_info", {})
            if cluster_info.get("tft_set") != "TFTSet18":
                return []
            cluster_id = int(cluster_info["cluster_id"])
            updated_at = cluster_info.get("updated_at")
            snapshot_age_hours: float | None = None
            if updated_at:
                try:
                    updated = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
                    snapshot_age_hours = max(0.0, (datetime.now(UTC) - updated).total_seconds() / 3600.0)
                except ValueError:
                    pass
            clusters = cluster_info.get("cluster_details", {}).get("clusters", [])
            samples: list[TrainingSample] = []
            snapshot_clusters: list[dict[str, Any]] = []

            for cluster in clusters:
                comp_id = int(cluster["Cluster"])
                detail_payload = await http.json(
                    f"{self.base}/comp_details",
                    params={"comp": comp_id, "cluster_id": cluster_id},
                )
                detail = detail_payload.get("results", {})
                # Options are exact final board variants grouped by level and carry the most
                # useful supervised signal: avg placement + count.
                for level_key, options in (detail.get("options") or {}).items():
                    try:
                        level = int(level_key)
                    except ValueError:
                        continue
                    for option in options or []:
                        count = int(option.get("count") or 0)
                        avg = float(option.get("avg") or 0)
                        units = self._split(option.get("units_list"))
                        if count < 3 or not units or not (1.0 <= avg <= 8.0):
                            continue
                        samples.append(
                            TrainingSample(
                                source=self.name,
                                patch=self.live_patch,
                                region="GLOBAL",
                                units=units,
                                traits=self._split(option.get("traits_list")),
                                level=level,
                                avg_placement=avg,
                                games=count,
                                source_weight=self.weight,
                                sample_kind="exact_comp_option",
                                context_id=str(comp_id),
                                age_hours=snapshot_age_hours,
                            )
                        )

                # MetaTFT exposes stage-level early boards for a cluster. These rows are
                # especially valuable for a coaching tool because the user's first decision
                # is usually "what can I field now?", not "what was the final level-8 board?".
                for level_key, options in (detail.get("early_options") or {}).items():
                    try:
                        level = int(level_key)
                    except ValueError:
                        continue
                    for option in options or []:
                        count = int(option.get("count") or 0)
                        avg = float(option.get("avg") or 0)
                        units = self._split(option.get("unit_list"))
                        if count < 3 or len(units) < 3 or not (1.0 <= avg <= 8.0):
                            continue
                        samples.append(
                            TrainingSample(
                                source=self.name,
                                patch=self.live_patch,
                                region="GLOBAL",
                                units=units,
                                level=max(3, min(10, level)),
                                avg_placement=avg,
                                win_rate=float(option.get("win")) if option.get("win") is not None else None,
                                games=count,
                                source_weight=self.weight * 0.92,
                                sample_kind="early_board",
                                context_id=str(comp_id),
                                age_hours=snapshot_age_hours,
                            )
                        )

                # Build rows teach item-holder quality. They are also useful for the shared
                # encoder even though they don't represent a complete board.
                for build in detail.get("builds") or []:
                    count = int(build.get("count") or 0)
                    avg = float(build.get("avg") or 0)
                    unit = build.get("unit")
                    items = list(build.get("buildName") or [])
                    if count < 5 or not unit or len(items) < 2 or not (1.0 <= avg <= 8.0):
                        continue
                    samples.append(
                        TrainingSample(
                            source=self.name,
                            patch=self.live_patch,
                            region="GLOBAL",
                            units=[unit],
                            items=items,
                            item_holders={unit: items},
                            level=8,
                            avg_placement=avg,
                            games=count,
                            source_weight=self.weight * 0.82,
                            sample_kind="item_holder_build",
                            context_id=str(comp_id),
                            age_hours=snapshot_age_hours,
                        )
                    )

                # Keep a compact runtime snapshot. API inference should never crawl a third
                # party site on every user click; collection is an offline/cacheable job.
                compact_options: dict[str, list[dict[str, Any]]] = {}
                for level_key, options in (detail.get("options") or {}).items():
                    cleaned = sorted(
                        [
                            {
                                "units": self._split(option.get("units_list")),
                                "traits": self._split(option.get("traits_list")),
                                "avg": float(option.get("avg") or 0),
                                "count": int(option.get("count") or 0),
                                "score": float(option.get("score") or 0),
                            }
                            for option in options or []
                            if option.get("units_list")
                        ],
                        key=lambda value: (-value["count"], value["avg"] or 9),
                    )[:32]
                    if cleaned:
                        compact_options[str(level_key)] = cleaned

                compact_early: dict[str, list[dict[str, Any]]] = {}
                for level_key, options in (detail.get("early_options") or {}).items():
                    cleaned = sorted(
                        [
                            {
                                "units": self._split(option.get("unit_list")),
                                "avg": float(option.get("avg") or 0),
                                "count": int(option.get("count") or 0),
                                "win": float(option.get("win") or 0),
                                "level": float(option.get("level") or level_key or 0),
                            }
                            for option in options or []
                            if option.get("unit_list")
                        ],
                        key=lambda value: (-value["count"], value["avg"] or 9),
                    )[:20]
                    if cleaned:
                        compact_early[str(level_key)] = cleaned

                compact_builds = sorted(
                    [
                        {
                            "unit": build.get("unit"),
                            "items": list(build.get("buildName") or []),
                            "avg": float(build.get("avg") or 0),
                            "count": int(build.get("count") or 0),
                            "score": float(build.get("score") or 0),
                            "placeChange": float(build.get("place_change") or 0),
                        }
                        for build in detail.get("builds") or []
                        if build.get("unit") and build.get("buildName")
                    ],
                    key=lambda value: (-value["count"], value["avg"] or 9),
                )[:120]

                compact_items = sorted(
                    [
                        {
                            "item": row.get("itemNames"),
                            "avg": float(row.get("avg") or 0),
                            "count": int(row.get("count") or 0),
                            "pcnt": float(row.get("pcnt") or 0),
                            "units": sorted(
                                [
                                    {
                                        "unit": holder.get("units"),
                                        "count": int(holder.get("count") or 0),
                                        "avg": float(holder.get("avg") or 0),
                                        "placeChange": float(holder.get("place_change") or 0),
                                        "itemPick": float(holder.get("item_pick") or 0),
                                    }
                                    for holder in row.get("units") or []
                                    if holder.get("units")
                                ],
                                key=lambda value: (-value["count"], value["avg"] or 9),
                            )[:12],
                        }
                        for row in detail.get("itemNames") or []
                        if row.get("itemNames")
                    ],
                    key=lambda value: (-value["count"], value["avg"] or 9),
                )[:80]

                compact_units = sorted(
                    [
                        {
                            "unit": row.get("unit"),
                            "avg": float(row.get("avg") or 0),
                            "count": int(row.get("count") or 0),
                            "pcnt": float(row.get("pcnt") or 0),
                            "tiers": list(row.get("tiers") or []),
                            "numItems": list(row.get("num_items") or []),
                        }
                        for row in detail.get("unit_stats") or []
                        if row.get("unit")
                    ],
                    key=lambda value: (-value["count"], value["avg"] or 9),
                )[:32]

                overall = detail.get("overall") or {}
                trends = list(detail.get("trends") or [])
                snapshot_clusters.append(
                    {
                        "id": comp_id,
                        "nameParts": list(cluster.get("name") or []),
                        "nameString": cluster.get("name_string") or "",
                        "centroidUnits": [part.strip() for part in str(cluster.get("units_string") or "").split(",") if part.strip()],
                        "traitsString": cluster.get("traits_string") or "",
                        "overall": {
                            "avg": float(overall.get("avg") or (trends[-1].get("avg") if trends else 0) or 0),
                            "count": int(overall.get("count") or (trends[-1].get("count") if trends else 0) or 0),
                            "pick": float(trends[-1].get("pick") or 0) if trends else 0.0,
                        },
                        "options": compact_options,
                        "earlyOptions": compact_early,
                        "builds": compact_builds,
                        "itemStats": compact_items,
                        "unitStats": compact_units,
                        "traits": list(detail.get("traits") or [])[:40],
                    }
                )

            self.snapshot = {
                "generatedAt": datetime.now(UTC).isoformat(),
                "source": self.name,
                "set": 18,
                "patch": self.live_patch,
                "queue": "LIVE",
                "clusterId": cluster_id,
                "clusterCreatedAt": cluster_info.get("created_at"),
                "clusterUpdatedAt": cluster_info.get("updated_at"),
                "clusters": snapshot_clusters,
            }

            return samples
        finally:
            await http.close()

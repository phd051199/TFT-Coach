from __future__ import annotations

from datetime import UTC, datetime
import os
import re
from typing import Any

from bs4 import BeautifulSoup

from backend.app.catalog import load_catalog

from .base import TrainingSample
from .http import PublicHttp


class OpggLiveSource:
    """Independent live Set 18 aggregate source from OP.GG.

    OP.GG server-renders current patch tables, which makes it useful as a cross-check against
    MetaTFT. We only parse statistics visibly present in the public HTML and require the page
    to identify Set/Season 18 plus the requested live patch.
    """

    name = "opgg-live"
    base = "https://op.gg/tft/meta-trends"

    def __init__(self) -> None:
        self.patch = os.getenv("HEXCOACH_PATCH", "18.1")
        self.weight = float(os.getenv("HEXCOACH_WEIGHT_OPGG", "0.82"))
        self.item_weight = float(os.getenv("HEXCOACH_WEIGHT_OPGG_ITEM", "0.76"))
        self.snapshot: dict[str, Any] = {}
        self.catalog = load_catalog()
        self.name_to_id = {str(row["name"]): str(row["id"]) for row in self.catalog.champions}

    @staticmethod
    def _stats(text: str) -> tuple[float, float | None, float | None, int] | None:
        avg_match = re.search(r"#\s*([1-8](?:\.\d+)?)", text)
        percentages = [float(value) / 100.0 for value in re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)]
        game_matches = re.findall(r"\b([0-9][0-9,]{1,})\b", text)
        if not avg_match or not game_matches:
            return None
        games = int(game_matches[-1].replace(",", ""))
        top4 = percentages[0] if percentages else None
        win = percentages[-1] if len(percentages) >= 2 else None
        return float(avg_match.group(1)), top4, win, games

    @staticmethod
    def _champion_alts(node: Any) -> list[str]:
        output: list[str] = []
        for image in node.find_all("img"):
            if "/tft-champion/tiles/" not in str(image.get("src") or ""):
                continue
            name = str(image.get("alt") or "").strip()
            if name and name not in output:
                output.append(name)
        return output

    def _parse_comps(self, html: str) -> tuple[list[TrainingSample], list[dict[str, Any]], int]:
        soup = BeautifulSoup(html, "html.parser")
        page_text = " ".join(soup.stripped_strings)
        if "Set 18" not in page_text and "Season 18" not in page_text:
            return [], [], 0
        total_match = re.search(r"analysis of\s+([0-9,]+)\s+games", page_text, re.IGNORECASE)
        total_games = int(total_match.group(1).replace(",", "")) if total_match else 0
        rows: list[TrainingSample] = []
        snapshot: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        for li in soup.find_all("li"):
            text = " ".join(li.stripped_strings)
            if "Avg. place" not in text or "Top 4 rate" not in text or "Pick rate" not in text:
                continue
            units = [self.name_to_id[name] for name in self._champion_alts(li) if name in self.name_to_id]
            units = list(dict.fromkeys(units))
            if len(units) < 4:
                continue
            signature = tuple(sorted(units))
            if signature in seen:
                continue
            seen.add(signature)
            avg_match = re.search(r"Avg\.\s*place\s*([0-9.]+)", text, re.IGNORECASE)
            win_match = re.search(r"1st\s*place\s*([0-9.]+)%", text, re.IGNORECASE)
            top4_match = re.search(r"Top\s*4\s*rate\s*([0-9.]+)%", text, re.IGNORECASE)
            pick_match = re.search(r"Pick\s*rate\s*([0-9.]+)%", text, re.IGNORECASE)
            if not avg_match or not top4_match:
                continue
            avg = float(avg_match.group(1))
            top4 = float(top4_match.group(1)) / 100.0
            win = float(win_match.group(1)) / 100.0 if win_match else None
            pick = float(pick_match.group(1)) / 100.0 if pick_match else 0.0
            games = max(10, round(total_games * pick)) if total_games and pick else 10
            strongs = [" ".join(tag.stripped_strings) for tag in li.find_all("strong")]
            name = next((value for value in strongs if value and not re.fullmatch(r"[#0-9.%]+", value)), "")
            context = f"opgg-comp:{'-'.join(signature)}"
            rows.append(
                TrainingSample(
                    source=self.name,
                    patch=self.patch,
                    region="GLOBAL",
                    units=units,
                    level=len(units),
                    avg_placement=avg,
                    top4_rate=top4,
                    win_rate=win,
                    games=games,
                    source_weight=self.weight,
                    sample_kind="opgg_comp",
                    context_id=context,
                    age_hours=0.25,
                )
            )
            snapshot.append({
                "name": name,
                "units": units,
                "avg": avg,
                "top4": top4,
                "win": win,
                "pick": pick,
                "games": games,
            })
        return rows, snapshot, total_games

    def _parse_champions(self, html: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for tr in soup.find_all("tr"):
            images = [image for image in tr.find_all("img") if "/tft-champion/tiles/" in str(image.get("src") or "")]
            if not images:
                continue
            name = str(images[0].get("alt") or "")
            unit_id = self.name_to_id.get(name)
            if not unit_id or unit_id in seen:
                continue
            stats = self._stats(" ".join(tr.stripped_strings))
            if not stats:
                continue
            avg, top4, win, games = stats
            seen.add(unit_id)
            output.append({"unit": unit_id, "avg": avg, "top4": top4, "win": win, "games": games})
        return output

    def _parse_items(self, html: str) -> tuple[list[TrainingSample], list[dict[str, Any]], list[dict[str, Any]]]:
        soup = BeautifulSoup(html, "html.parser")
        rows: list[TrainingSample] = []
        item_stats: list[dict[str, Any]] = []
        holder_stats: list[dict[str, Any]] = []
        current_item: str | None = None
        for tr in soup.find_all("tr"):
            images = tr.find_all("img")
            item_image = next((image for image in images if "/tft-item/" in str(image.get("src") or "")), None)
            if item_image is not None:
                match = re.search(r"/tft-item/([^/?]+)\.png", str(item_image.get("src") or ""))
                current_item = match.group(1) if match else None
                item = self.catalog.item_by_id.get(current_item or "")
                if not item or item.get("category") not in {"completed", "artifact", "radiant", "emblem"}:
                    current_item = None
                    continue
                stats = self._stats(" ".join(tr.stripped_strings))
                if stats:
                    avg, top4, win, games = stats
                    item_stats.append({"item": current_item, "avg": avg, "top4": top4, "win": win, "games": games})
                continue
            if not current_item:
                continue
            champion_image = next((image for image in images if "/tft-champion/tiles/" in str(image.get("src") or "")), None)
            if champion_image is None:
                continue
            holder_name = str(champion_image.get("alt") or "")
            holder_id = self.name_to_id.get(holder_name)
            stats = self._stats(" ".join(tr.stripped_strings))
            if not holder_id or not stats:
                continue
            avg, top4, win, games = stats
            context = f"opgg-item:{current_item}:{holder_id}"
            rows.append(
                TrainingSample(
                    source=self.name,
                    patch=self.patch,
                    region="GLOBAL",
                    units=[holder_id],
                    items=[current_item],
                    item_holders={holder_id: [current_item]},
                    level=8,
                    avg_placement=avg,
                    top4_rate=top4,
                    win_rate=win,
                    games=games,
                    source_weight=self.item_weight,
                    sample_kind="opgg_item_holder",
                    context_id=context,
                    age_hours=0.25,
                )
            )
            holder_stats.append({
                "item": current_item,
                "unit": holder_id,
                "avg": avg,
                "top4": top4,
                "win": win,
                "games": games,
            })
        return rows, item_stats, holder_stats

    async def collect(self) -> list[TrainingSample]:
        if self.weight <= 0 and self.item_weight <= 0:
            return []
        http = PublicHttp(delay=0.05)
        try:
            query = f"version={self.patch}"
            comps_html = await http.text(f"{self.base}/comps?{query}")
            champions_html = await http.text(f"{self.base}/champion?{query}")
            items_html = await http.text(f"{self.base}/item?{query}")
            # Reject stale/wrong-set pages rather than silently polluting the model.
            if f"Set 18" not in comps_html and f"Season 18" not in comps_html:
                return []
            comp_rows, comps, total_games = self._parse_comps(comps_html)
            item_rows, items, holders = self._parse_items(items_html)
            units = self._parse_champions(champions_html)
            self.snapshot = {
                "generatedAt": datetime.now(UTC).isoformat(),
                "source": self.name,
                "set": 18,
                "patch": self.patch,
                "queue": "LIVE",
                "games24h": total_games,
                "comps": comps,
                "unitStats": units,
                "itemStats": items,
                "itemHolders": holders,
            }
            return comp_rows + item_rows
        finally:
            await http.close()

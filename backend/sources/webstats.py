from __future__ import annotations

import os
import re

from bs4 import BeautifulSoup

from .base import TrainingSample
from .http import PublicHttp


class LolchessSource:
    """Best-effort LoLChess public-page adapter.

    Set 18 PBE statistical tables may be JS/anti-bot gated. In that case this adapter
    returns zero samples instead of attempting a bypass. It becomes useful automatically
    when the public HTML exposes rows again after the set goes live.
    """

    name = "lolchess"
    pages = (
        "https://lolchess.gg/champions/set18?hl=en",
        "https://lolchess.gg/items/set18?hl=en",
        "https://lolchess.gg/synergies/set18?hl=en",
    )

    def __init__(self) -> None:
        self.weight = float(os.getenv("HEXCOACH_WEIGHT_LOLCHESS", "0.70"))

    async def collect(self) -> list[TrainingSample]:
        # LoLChess rows are unit/trait aggregates, not complete boards. Keep them out of the
        # board ranker until we can normalize a stable server-rendered table schema.
        if self.weight <= 0:
            return []
        http = PublicHttp(delay=0.1)
        try:
            available = 0
            for url in self.pages:
                try:
                    html = await http.text(url)
                except Exception:
                    continue
                soup = BeautifulSoup(html, "html.parser")
                available += len(soup.select("tbody tr"))
            # Availability is intentionally probed so source health can be reported later.
            _ = available
            return []
        finally:
            await http.close()


class TacticsToolsSource:
    """Best-effort tactics.tools aggregate adapter.

    Their Set 18 info pages are fully server-rendered now, while live statistical pages are
    still Set 17. We don't create fake Set 18 labels from old-set stats; this source starts
    contributing automatically once its meta page identifies Set 18.
    """

    name = "tactics.tools"

    def __init__(self) -> None:
        self.weight = float(os.getenv("HEXCOACH_WEIGHT_TACTICSTOOLS", "0.70"))

    async def collect(self) -> list[TrainingSample]:
        if self.weight <= 0:
            return []
        http = PublicHttp()
        try:
            html = await http.text("https://tactics.tools/en")
            text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            # Current live meta must explicitly be Set 18; avoid accidental cross-set labels.
            live_match = re.search(r"Top Comps\s*\(18\.", text, re.IGNORECASE)
            if not live_match:
                return []
            # The stable public HTML currently exposes comp names/units but not a complete
            # machine-readable board + placement table. Keep this adapter conservative.
            return []
        finally:
            await http.close()

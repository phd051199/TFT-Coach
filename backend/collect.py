from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path

from .sources import LolchessSource, MetaTFTSource, RiotHighEloSource, TacticsToolsSource
from .sources.base import TrainingSample, write_jsonl


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "backend" / "data"


async def main() -> None:
    sources = [
        RiotHighEloSource(),
        MetaTFTSource(),
        LolchessSource(),
        TacticsToolsSource(),
    ]
    all_rows: list[TrainingSample] = []
    health: dict[str, dict] = {}
    for source in sources:
        try:
            rows = await source.collect()
            health[source.name] = {"ok": True, "samples": len(rows)}
            all_rows.extend(rows)
            snapshot = getattr(source, "snapshot", None)
            if snapshot:
                DATA.mkdir(parents=True, exist_ok=True)
                (DATA / f"{source.name}.snapshot.json").write_text(
                    json.dumps(snapshot, ensure_ascii=False, indent=2),
                    "utf-8",
                )
            print(f"{source.name}: {len(rows):,} samples")
        except Exception as exc:
            health[source.name] = {"ok": False, "samples": 0, "error": str(exc)}
            print(f"{source.name}: FAILED: {exc}")

    # Deduplicate exact aggregate rows from repeated source pages while preserving source
    # diversity. Riot individual matches naturally remain distinct because their labels vary.
    deduped: list[TrainingSample] = []
    seen: set[tuple] = set()
    for row in all_rows:
        key = (
            row.source,
            row.patch,
            row.region,
            tuple(sorted(row.units)),
            tuple(sorted(row.items)),
            row.level,
            round(row.avg_placement, 4),
            row.games,
            row.sample_kind,
            row.context_id,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    write_jsonl(DATA / "training.jsonl", deduped)
    counts = Counter(row.source for row in deduped)
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "source-health.json").write_text(
        json.dumps({"sources": health, "samples": dict(counts), "total": len(deduped)}, indent=2),
        "utf-8",
    )
    print(f"total: {len(deduped):,} -> backend/data/training.jsonl")


if __name__ == "__main__":
    asyncio.run(main())

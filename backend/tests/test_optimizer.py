from backend.app.catalog import load_catalog
from backend.app.optimizer import HybridCoach


def test_fallback_returns_legal_board() -> None:
    coach = HybridCoach(load_catalog())
    board = coach._fallback_early_board(4, set())
    assert len(board) == 4
    assert len(board) == len(set(board))
    assert all(unit in coach.catalog.champion_by_id for unit in board)


def test_recommendation_schema() -> None:
    coach = HybridCoach(load_catalog())
    result = coach.recommend(4, [], [])
    assert len(result["earlyBoardIds"]) == 4
    assert "itemPlan" in result
    assert "model" in result
    assert "data" in result


def test_item_plan_names_stage_and_holder() -> None:
    coach = HybridCoach(load_catalog())
    components = [item["id"] for item in coach.catalog.items if item.get("category") == "component"][:4]
    result = coach.recommend(4, [], components)
    for row in result["itemPlan"]:
        assert row["stage"] in {"opener", "mid", "late", "bis"}
        assert row.get("holderId") in coach.catalog.champion_by_id


def test_live_data_never_reports_pbe() -> None:
    coach = HybridCoach(load_catalog())
    status = coach.data_status()
    assert status.get("patch") == "18.1"
    assert str(status.get("queue") or "").upper() != "PBE"


def test_comp_confidence_is_bounded() -> None:
    coach = HybridCoach(load_catalog())
    result = coach.recommend(4, [], [])
    for comp in result["comps"]:
        assert 0.0 <= comp["confidence"] <= 100.0
        assert 0.0 <= comp["uncertainty"] <= 100.0
        assert len(comp["boardIds"]) == len(set(comp["boardIds"]))


def test_each_stage_respects_component_inventory() -> None:
    from collections import Counter

    coach = HybridCoach(load_catalog())
    component_ids = [
        item["id"] for item in coach.catalog.items
        if item.get("category") == "component" and "Spatula" not in str(item.get("nameEn"))
    ][:6]
    inventory = Counter(component_ids)
    result = coach.recommend(4, [], component_ids)
    for stage in ("opener", "mid", "late"):
        used: Counter[str] = Counter()
        for row in result["itemPlan"]:
            if row.get("stage") != stage:
                continue
            item = coach.catalog.item_by_id[row["itemId"]]
            used.update(item.get("composition") or [])
        assert all(used[key] <= inventory[key] for key in used)


def test_ad_item_is_not_assigned_to_ap_holder_without_evidence() -> None:
    coach = HybridCoach(load_catalog())
    last_whisper = next(item for item in coach.catalog.items if item.get("nameEn") == "Last Whisper")
    leblanc = next(champion for champion in coach.catalog.champions if champion.get("name") == "LeBlanc")
    xayah = next(champion for champion in coach.catalog.champions if champion.get("name") == "Xayah")
    assert coach._item_role_score(last_whisper, xayah["id"]) > coach._item_role_score(last_whisper, leblanc["id"])


def test_transition_path_is_monotonic_and_legal() -> None:
    coach = HybridCoach(load_catalog())
    result = coach.recommend(4, [], [])
    for comp in result["comps"]:
        path = comp.get("transitionPath") or []
        levels = [row["level"] for row in path]
        assert levels == sorted(levels)
        for row in path:
            assert len(row["boardIds"]) == len(set(row["boardIds"]))
            assert all(unit in coach.catalog.champion_by_id for unit in row["boardIds"])

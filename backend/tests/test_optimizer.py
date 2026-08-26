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

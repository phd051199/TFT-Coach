from backend.app.catalog import load_catalog
from backend.app.optimizer import HybridCoach
from backend.ml.position import display_grid_position
from backend.ml.train import load_item_affinity_rows, load_item_pair_affinity_rows, load_lobby_star_rows


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


def test_lobby_cache_supplies_real_star_and_reroll_supervision() -> None:
    star_rows, reroll_rows = load_lobby_star_rows()
    assert len(star_rows) >= 300
    assert len(reroll_rows) >= 100
    assert {int(row["stars"]) for row in star_rows} == {1, 2, 3}
    assert any(float(row["target_strength"]) >= 0.5 for row in reroll_rows)
    assert any(float(row["target_strength"]) < 0.5 for row in reroll_rows)


def test_recommendation_exposes_star_targets_and_reroll_strategy() -> None:
    coach = HybridCoach(load_catalog())
    status = coach.model_status()
    assert "starAvailable" in status
    assert "rerollAvailable" in status
    result = coach.recommend(4, [], [])
    if status["starAvailable"] and status["rerollAvailable"]:
        assert result["comps"]
        for comp in result["comps"]:
            targets = comp.get("starTargets") or []
            assert len(targets) == len(comp["boardIds"])
            assert {row["unitId"] for row in targets} == set(comp["boardIds"])
            assert all(1 <= int(row["stars"]) <= 3 for row in targets)
            assert 0.0 <= float(comp.get("rerollScore") or 0.0) <= 100.0


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


def test_infinity_edge_is_strict_mismatch_for_rakan_but_not_xayah() -> None:
    coach = HybridCoach(load_catalog())
    infinity_edge = next(item for item in coach.catalog.items if item.get("nameEn") == "Infinity Edge")
    rakan = next(champion for champion in coach.catalog.champions if champion.get("name") == "Rakan")
    xayah = next(champion for champion in coach.catalog.champions if champion.get("name") == "Xayah")
    assert coach._strict_role_mismatch(infinity_edge, rakan["id"])
    assert not coach._strict_role_mismatch(infinity_edge, xayah["id"])


def test_emblem_prefers_unit_without_existing_trait() -> None:
    coach = HybridCoach(load_catalog())
    emblem = next(item for item in coach.catalog.items if item.get("nameEn") == "Fae Emblem")
    ids = {
        champion["name"]: champion["id"]
        for champion in coach.catalog.champions
        if champion.get("name") in {"Rakan", "Xayah", "Rammus"}
    }
    cluster = coach.clusters[0]
    ranked = coach._rank_holders(cluster, emblem, [ids["Rakan"], ids["Xayah"], ids["Rammus"]])
    assert ranked
    assert ranked[0][0] == ids["Rammus"]


def test_joint_item_solver_can_stack_three_items_on_one_holder() -> None:
    coach = HybridCoach(load_catalog())
    components = ["a", "b", "c", "d", "e", "f"]
    ranked = [
        {"holderId": "carry", "itemId": "i1", "recipe": ["a", "b"], "score": 88.0},
        {"holderId": "carry", "itemId": "i2", "recipe": ["c", "d"], "score": 86.0},
        {"holderId": "carry", "itemId": "i3", "recipe": ["e", "f"], "score": 84.0},
        {"holderId": "other", "itemId": "i2", "recipe": ["c", "d"], "score": 88.0},
        {"holderId": "other2", "itemId": "i3", "recipe": ["e", "f"], "score": 87.0},
    ]
    selected = coach._select_stage_item_set(ranked, components, limit=3)
    assert len(selected) == 3
    assert sum(row["holderId"] == "carry" for row in selected) == 3


def test_item_solver_preserves_core_item_over_two_mediocre_slams() -> None:
    coach = HybridCoach(load_catalog())
    components = ["sword", "bow", "belt", "cloak"]
    ranked = [
        {
            "holderId": "carry",
            "itemId": "core",
            "recipe": ["sword", "bow"],
            "score": 94.0,
            "corePriority": 0.94,
        },
        {
            "holderId": "tank",
            "itemId": "trash-a",
            "recipe": ["sword", "belt"],
            "score": 75.0,
            "corePriority": 0.08,
        },
        {
            "holderId": "support",
            "itemId": "trash-b",
            "recipe": ["bow", "cloak"],
            "score": 75.0,
            "corePriority": 0.05,
        },
    ]
    selected = coach._select_stage_item_set(ranked, components, limit=3, stage_id="opener")
    assert {row["itemId"] for row in selected} == {"core"}


def test_item_solver_can_hold_low_value_components_in_opener() -> None:
    coach = HybridCoach(load_catalog())
    selected = coach._select_stage_item_set(
        [
            {
                "holderId": "holder",
                "itemId": "low-value",
                "recipe": ["a", "b"],
                "score": 54.0,
                "corePriority": 0.0,
            }
        ],
        ["a", "b"],
        limit=3,
        stage_id="opener",
    )
    assert selected == []


def test_unrelated_component_does_not_hide_existing_craft() -> None:
    coach = HybridCoach(load_catalog())
    ranked = [
        {
            "holderId": "carry",
            "itemId": "edge",
            "recipe": ["sword", "vest"],
            "score": 82.0,
            "corePriority": 0.72,
        },
        {
            "holderId": "carry",
            "itemId": "deathblade",
            "recipe": ["sword", "sword"],
            "score": 79.0,
            "corePriority": 0.48,
        },
        {
            "holderId": "caster",
            "itemId": "shojin",
            "recipe": ["sword", "tear"],
            "score": 82.0,
            "corePriority": 0.42,
        },
    ]
    base = coach._select_stage_item_set(
        ranked,
        ["sword", "sword", "sword", "vest"],
        limit=3,
        stage_id="opener",
    )
    extended = coach._select_stage_item_set(
        ranked,
        ["sword", "sword", "sword", "vest", "tear"],
        limit=3,
        stage_id="opener",
    )
    base_items = {row["itemId"] for row in base}
    extended_items = {row["itemId"] for row in extended}
    assert "deathblade" in base_items
    assert base_items <= extended_items


def test_new_sidegrade_does_not_evict_stable_item_set() -> None:
    coach = HybridCoach(load_catalog())
    ranked = [
        {
            "holderId": "carry",
            "itemId": "edge",
            "recipe": ["sword", "vest"],
            "score": 80.0,
            "corePriority": 0.80,
        },
        {
            "holderId": "carry",
            "itemId": "deathblade",
            "recipe": ["sword", "sword"],
            "score": 79.0,
            "corePriority": 0.48,
        },
        {
            "holderId": "tank",
            "itemId": "vow",
            "recipe": ["vest", "tear"],
            "score": 82.0,
            "corePriority": 0.35,
        },
    ]
    base = coach._select_stage_item_set(
        ranked,
        ["sword", "sword", "sword", "vest"],
        limit=3,
        stage_id="opener",
    )
    extended = coach._select_stage_item_set(
        ranked,
        ["sword", "sword", "sword", "vest", "tear"],
        limit=3,
        stage_id="opener",
    )
    assert {row["itemId"] for row in base} <= {row["itemId"] for row in extended}


def test_previous_backend_plan_is_preserved_when_only_adding_components() -> None:
    coach = HybridCoach(load_catalog())
    target = "422020"
    sword = "DA_Component_BFSword"
    belt = "DA_Component_GiantsBelt"
    vest = "DA_Component_ChainVest"
    base_components = [sword, sword, sword, belt]
    base = coach.recommend(6, [], base_components, target)
    previous_plan = [row for row in base["itemPlan"] if row.get("stage") != "bis"]
    extended = coach.recommend(
        6,
        [],
        base_components + [vest],
        target,
        previous_level=6,
        previous_comp_id=target,
        previous_owned_ids=[],
        previous_components=base_components,
        previous_item_plan=previous_plan,
    )
    base_items = {row["itemId"] for row in base["itemPlan"] if row.get("stage") == "opener"}
    extended_items = {row["itemId"] for row in extended["itemPlan"] if row.get("stage") == "opener"}
    assert base_items
    assert base_items <= extended_items


def test_target_comp_returns_positioning_for_selected_comp() -> None:
    coach = HybridCoach(load_catalog())
    initial = coach.recommend(4, [], [])
    assert len(initial["comps"]) >= 2
    target_id = initial["comps"][1]["id"]
    selected = coach.recommend(4, [], [], target_id)
    target = next(row for row in selected["comps"] if row["id"] == target_id)
    assert target["positioning"]


def test_target_comp_outside_default_top_six_is_still_returned() -> None:
    coach = HybridCoach(load_catalog())
    initial = coach.recommend(4, [], [])
    shown = {row["id"] for row in initial["comps"]}
    target_id = next(
        str(cluster.get("id"))
        for cluster in reversed(coach.clusters)
        if str(cluster.get("id")) not in shown and len(coach._best_final(cluster)[0]) >= 4
    )
    selected = coach.recommend(4, [], [], target_id)
    target = next(row for row in selected["comps"] if row["id"] == target_id)
    assert target["positioning"]
    assert target["rank"] >= 1


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


def test_positioning_is_legal_and_collision_free() -> None:
    coach = HybridCoach(load_catalog())
    result = coach.recommend(4, [], [])
    early = result.get("earlyPositioning") or []
    assert len(early) == len(result["earlyBoardIds"])
    assert len({row["cell"] for row in early}) == len(early)
    for row in early:
        assert row["unitId"] in result["earlyBoardIds"]
        assert 0 <= row["row"] <= 3
        assert 0 <= row["col"] <= 6
        assert row["cell"].startswith("cell_")
        assert 0.0 <= row["confidence"] <= 100.0

    if result["comps"]:
        final = result["comps"][0].get("positioning") or []
        assert len(final) == len(result["comps"][0]["boardIds"])
        assert len({row["cell"] for row in final}) == len(final)


def test_metatft_cell_order_is_mirrored_vertically_for_display() -> None:
    # MetaTFT/Riot cell_1..7 is the player's back row. The UI renders enemy-side/frontline at
    # the top, matching MetaTFT's own board component (cell_22..28 on display row zero).
    assert display_grid_position(0) == (3, 0)   # cell_1
    assert display_grid_position(6) == (3, 6)   # cell_7
    assert display_grid_position(21) == (0, 0)  # cell_22
    assert display_grid_position(27) == (0, 6)  # cell_28

    coach = HybridCoach(load_catalog())
    unit_id = str(coach.catalog.champions[0]["id"])
    cluster = {
        "id": "orientation-test",
        "positioning": {
            "units": {
                unit_id: [{"cell": "cell_1", "count": 500}],
            },
        },
    }
    position = coach._position_board(cluster, [unit_id])[0]
    assert position["cell"] == "cell_1"
    assert (position["row"], position["col"]) == (3, 0)


def test_nonstacking_item_pairs_are_rejected_without_live_pair_evidence() -> None:
    coach = HybridCoach(load_catalog())
    item_id = {str(item.get("nameEn")): str(item["id"]) for item in coach.catalog.items}
    duplicate_effect_pairs = [
        ("Sunfire Cape", "Red Buff", "wound"),
        ("Sunfire Cape", "Morellonomicon", "wound"),
        ("Red Buff", "Morellonomicon", "wound"),
        ("Evenshroud", "Last Whisper", "sunder"),
    ]
    for left_name, right_name, effect in duplicate_effect_pairs:
        left = item_id[left_name]
        right = item_id[right_name]
        assert effect in coach._item_unique_effects(left)
        assert effect in coach._item_unique_effects(right)
        compatibility, support, pair_count, overlap, _ = coach._item_pair_compatibility(None, left, right)
        assert support >= 100
        assert pair_count == 0
        assert effect in overlap
        assert compatibility < 0.12
        assert not coach._items_can_coexist(None, left, right)


def test_item_pair_training_has_global_zero_cooccurrence_targets() -> None:
    rows = load_item_pair_affinity_rows()
    global_rows = {
        frozenset(row["items"]): row
        for row in rows
        if row.get("source") == "metatft-item-pair-global"
    }
    for pair in [
        ("DA_SunfireCape", "DA_RedBuff"),
        ("DA_SunfireCape", "DA_Morellonomicon"),
        ("DA_RedBuff", "DA_Morellonomicon"),
        ("DA_Evenshroud", "DA_LastWhisper"),
    ]:
        row = global_rows[frozenset(pair)]
        assert row["pair_count"] == 0
        assert row["target_strength"] == 0.0
        assert min(row["left_support"], row["right_support"]) >= 100


def test_item_affinity_training_has_one_label_per_holder_item_input() -> None:
    rows = load_item_affinity_rows()
    assert len(rows) >= 1000
    signatures = {
        (str(row["units"][0]), str(row["items"][0]))
        for row in rows
    }
    assert len(signatures) == len(rows)
    assert all(int(row.get("games") or 0) >= 10 for row in rows)


def test_item_pair_training_has_one_label_per_model_input() -> None:
    rows = load_item_pair_affinity_rows()
    signatures = {
        (tuple(sorted(str(value) for value in row.get("units") or [])), tuple(sorted(row["items"])))
        for row in rows
    }
    assert len(signatures) == len(rows)


def test_position_model_reports_runtime_status() -> None:
    coach = HybridCoach(load_catalog())
    status = coach.model_status()
    assert "positionAvailable" in status
    if status["positionAvailable"]:
        assert status["position"].get("samples", 0) >= 100
    assert "itemPairAvailable" in status
    assert 0.0 <= float(status["boardRuntimeValueReliability"]) <= 1.0
    assert 0.0 <= float(status["itemRuntimeValueReliability"]) <= 1.0

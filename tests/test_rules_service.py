from tests.helpers import rules


def test_recommend_returns_k_distinct_items_not_in_basket():
    basket = ["85123A"]
    recs = rules().recommend(basket, 5)
    assert 1 <= len(recs) <= 5
    items = [r["item"] for r in recs]
    assert len(items) == len(set(items)) and not set(items) & set(basket)
    assert all(r["strategy"] in ("rule", "popularity") for r in recs)


def test_empty_basket_falls_back_to_popularity():
    recs = rules().recommend([], 3)
    assert len(recs) == 3 and all(r["strategy"] == "popularity" for r in recs)


def test_unknown_code_does_not_crash():
    assert len(rules().recommend(["00000ZZ"], 3)) == 3
    assert rules().describe("00000ZZ") is None

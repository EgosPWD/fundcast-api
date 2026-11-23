"""POST /predict, GET /predictions and GET /metrics with the real baseline model."""

import math

import pytest

from app.domain.entities import UNDISCLOSED_WARNING
from tests.integration.api.conftest import register_and_login

MINIMAL = {"year": 2025, "founded_year": 2021, "stage": "Seed", "sector": "Fintech",
           "country": "United States"}
CONTRACT_EXAMPLE = {**MINIMAL, "city": "New York", "investor_count": 3,
                    "lead_investor": "Y Combinator", "news_mentions": 2,
                    "news_source": "techcrunch.com", "company_domain": "acme.io", "prev_rounds": 1,
                    "prev_amount_usd": 1500000}
RESPONSE_FIELDS = {"prediction_id", "estimated_amount_usd", "estimated_log10", "range_usd",
                   "model_version", "warning", "warnings"}


def assert_valid_estimate(body: dict) -> None:
    assert set(body) == RESPONSE_FIELDS
    assert isinstance(body["prediction_id"], int)
    assert body["model_version"] == 1
    assert math.isfinite(body["estimated_log10"])
    assert body["estimated_amount_usd"] == pytest.approx(10 ** body["estimated_log10"])
    low, high = body["range_usd"]
    assert low == pytest.approx(body["estimated_amount_usd"] / 1.5)
    assert high == pytest.approx(body["estimated_amount_usd"] * 1.5)


def test_predict_with_the_contract_example(api, user_headers):
    response = api.client.post("/predict", json=CONTRACT_EXAMPLE, headers=user_headers)

    assert response.status_code == 200
    body = response.json()
    assert_valid_estimate(body)
    assert body["warning"] is None
    assert 3.0 <= body["estimated_log10"] <= 10.0


def test_predict_with_minimal_fields(api, user_headers):
    response = api.client.post("/predict", json=MINIMAL, headers=user_headers)

    assert response.status_code == 200
    assert_valid_estimate(response.json())


def test_predict_with_explicit_nulls(api, user_headers):
    payload = {**MINIMAL, "city": None, "investor_count": None, "lead_investor": None,
               "news_mentions": None, "news_source": None, "company_domain": None,
               "prev_rounds": None, "prev_amount_usd": None}

    response = api.client.post("/predict", json=payload, headers=user_headers)

    assert response.status_code == 200
    assert_valid_estimate(response.json())


def test_unknown_categories_do_not_break(api, user_headers):
    payload = {**MINIMAL, "stage": "Series Z", "sector": "Quantum Bakery", "country": "Atlantis",
               "city": "Nowhere", "lead_investor": "Nobody Capital", "news_source": "Blog"}

    response = api.client.post("/predict", json=payload, headers=user_headers)

    assert response.status_code == 200
    assert_valid_estimate(response.json())


def test_undisclosed_stage_returns_the_warning(api, user_headers):
    response = api.client.post("/predict", json={**MINIMAL, "stage": "Undisclosed"},
                               headers=user_headers)

    assert response.status_code == 200
    assert response.json()["warning"] == UNDISCLOSED_WARNING


@pytest.mark.parametrize("missing", ["year", "founded_year", "stage", "sector", "country"])
def test_predict_without_a_required_field_returns_422(api, user_headers, missing):
    payload = {k: v for k, v in MINIMAL.items() if k != missing}

    response = api.client.post("/predict", json=payload, headers=user_headers)

    assert response.status_code == 422
    assert list(response.json()) == ["detail"]
    assert missing in response.json()["detail"]


@pytest.mark.parametrize("override,message", [
    ({"stage": "   "}, "stage must not be empty"),
    ({"prev_amount_usd": 0}, "prev_amount_usd must be greater than 0"),
    ({"prev_rounds": -1}, "prev_rounds must be greater than or equal to 0"),
    ({"year": 2024, "founded_year": 2025},
     "founded_year must not be later than year (company_age would be negative)"),
])
def test_predict_with_a_domain_rule_violation_returns_422(api, user_headers, override, message):
    response = api.client.post("/predict", json={**MINIMAL, **override}, headers=user_headers)

    assert response.status_code == 422
    assert response.json() == {"detail": message}


def test_predict_with_a_wrong_type_returns_422(api, user_headers):
    response = api.client.post("/predict", json={**MINIMAL, "year": "next year"},
                               headers=user_headers)

    assert response.status_code == 422
    assert list(response.json()) == ["detail"]
    assert "year" in response.json()["detail"]


def test_predict_without_an_active_model_returns_503(api, user_headers):
    for version in api.models.versions.values():
        version.is_active = False

    response = api.client.post("/predict", json=MINIMAL, headers=user_headers)

    assert response.status_code == 503
    assert response.json() == {"detail": "No active model version"}


def test_predict_never_stores_news_excerpt(api, user_headers):
    payload = {**MINIMAL, "news_excerpt": "raised $5M in a seed round"}

    response = api.client.post("/predict", json=payload, headers=user_headers)

    assert response.status_code == 200
    stored = api.predictions.items[-1].input
    assert not hasattr(stored, "news_excerpt")


def test_history_is_filtered_per_user_and_admin_sees_all(api, user_headers, admin_headers):
    other_headers = register_and_login(api.client, "other@example.com")
    api.client.post("/predict", json=MINIMAL, headers=user_headers)
    api.client.post("/predict", json={**MINIMAL, "stage": "Series A"}, headers=user_headers)
    api.client.post("/predict", json=CONTRACT_EXAMPLE, headers=other_headers)

    mine = api.client.get("/predictions", headers=user_headers)
    theirs = api.client.get("/predictions", headers=other_headers)
    everything = api.client.get("/predictions", headers=admin_headers)

    assert mine.status_code == theirs.status_code == everything.status_code == 200
    assert mine.json()["total"] == 2
    assert [item["input"]["stage"] for item in mine.json()["items"]] == ["Series A", "Seed"]
    assert theirs.json()["total"] == 1
    assert theirs.json()["items"][0]["input"] == CONTRACT_EXAMPLE
    assert everything.json()["total"] == 3
    assert len({item["user_id"] for item in everything.json()["items"]}) == 2


def test_history_item_has_the_sent_data_and_the_result(api, user_headers):
    predicted = api.client.post("/predict", json=MINIMAL, headers=user_headers).json()

    item = api.client.get("/predictions", headers=user_headers).json()["items"][0]

    assert item["id"] == predicted["prediction_id"]
    assert item["estimated_amount_usd"] == pytest.approx(predicted["estimated_amount_usd"])
    assert item["estimated_log10"] == pytest.approx(predicted["estimated_log10"])
    assert item["model_version_id"] == 1
    assert item["input"]["stage"] == "Seed"
    assert item["input"]["city"] is None
    assert item["created_at"]


def test_history_is_paginated(api, user_headers):
    for _ in range(3):
        api.client.post("/predict", json=MINIMAL, headers=user_headers)

    response = api.client.get("/predictions", params={"page": 2, "size": 2},
                              headers=user_headers)

    assert response.status_code == 200
    body = response.json()
    assert (body["total"], body["page"], body["size"], len(body["items"])) == (3, 2, 2, 1)


@pytest.mark.parametrize("params,message", [({"page": 0}, "page must be >= 1"),
                                            ({"size": 101}, "size must be between 1 and 100")])
def test_history_with_invalid_paging_returns_422(api, user_headers, params, message):
    response = api.client.get("/predictions", params=params, headers=user_headers)

    assert response.status_code == 422
    assert response.json() == {"detail": message}


def test_options_come_from_the_active_model_with_cities_by_country(api, user_headers):
    response = api.client.get("/options", headers=user_headers)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"stages", "sectors", "countries", "cities_by_country",
                         "lead_investors", "news_sources", "model_version"}
    assert body["model_version"] == 1
    for key in ("stages", "sectors", "countries", "lead_investors", "news_sources"):
        assert body[key] == sorted(body[key]) and "Unknown" not in body[key]
    assert {"Seed", "Series A"} <= set(body["stages"])
    assert "Fintech" in body["sectors"] and "United States" in body["countries"]
    assert "Y Combinator" in body["lead_investors"]
    assert "techcrunch.com" in body["news_sources"] and "TechCrunch" not in body["news_sources"]
    # Only the (city, country) pairs stored in funding_rounds that the model knows.
    assert body["cities_by_country"]["Chile"] == ["Santiago"]
    assert body["cities_by_country"]["United States"] == ["New York"]
    assert "Atlantis City" not in str(body["cities_by_country"])  # country is null
    assert "Gotham" not in str(body["cities_by_country"])  # the model does not know it


def test_predict_warns_about_each_unknown_category(api, user_headers):
    payload = {**MINIMAL, "news_source": "TechCrunch", "city": "Nowhereville"}

    response = api.client.post("/predict", json=payload, headers=user_headers)

    assert response.status_code == 200
    assert response.json()["warnings"] == [
        {"code": "unknown_category", "field": "city", "value": "Nowhereville"},
        {"code": "unknown_category", "field": "news_source", "value": "TechCrunch"}]


def test_predict_without_unknown_categories_has_empty_warnings(api, user_headers):
    response = api.client.post("/predict", json=CONTRACT_EXAMPLE, headers=user_headers)

    assert response.json()["warnings"] == []


def test_predict_normalizes_company_domain_before_storing(api, user_headers):
    payload = {**MINIMAL, "company_domain": "https://WWW.Acme.io/"}

    api.client.post("/predict", json=payload, headers=user_headers)

    assert api.predictions.items[-1].input.company_domain == "acme.io"


def test_options_requires_a_token(api):
    response = api.client.get("/options")

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}


def test_metrics_returns_the_active_version_and_history(api, user_headers):
    response = api.client.get("/metrics", headers=user_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["active"]["version"] == 1
    assert body["active"]["is_active"] is True
    assert body["active"]["metrics"] == {"mae_log": 0.35, "r2": 0.615, "rmse_log": 0.495,
                                         "mdape": 0.529, "coverage": 0.473}
    assert (body["active"]["n_train"], body["active"]["n_test"]) == (12758, 3190)
    assert [v["version"] for v in body["versions"]] == [1]
    assert "artifact_path" not in body["active"]

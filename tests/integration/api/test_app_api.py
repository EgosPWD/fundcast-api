"""Health, CORS, startup hook and the uniform {"detail": "..."} error shape."""

from fastapi.testclient import TestClient

from app.adapters.inbound.api.app_factory import ApiUseCases, create_app
from app.domain.errors import NotFoundError
from tests.integration.api.conftest import ALLOWED_ORIGIN


def test_health_is_public(api):
    response = api.client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_startup_hook_runs_once(api):
    assert api.startup_calls == ["started"]


def test_cors_allows_a_configured_origin(api):
    response = api.client.options("/predict", headers={
        "Origin": ALLOWED_ORIGIN, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type"})

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


def test_cors_header_is_present_on_a_simple_request(api):
    response = api.client.get("/health", headers={"Origin": ALLOWED_ORIGIN})

    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN


def test_cors_rejects_an_unknown_origin(api):
    response = api.client.get("/health", headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in response.headers


def test_unknown_route_returns_404_with_detail(api):
    response = api.client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}


def test_wrong_method_returns_405_with_detail(api):
    response = api.client.get("/retrain")

    assert response.status_code == 405
    assert response.json() == {"detail": "Method Not Allowed"}


def test_malformed_json_returns_422_with_a_detail_string(api):
    response = api.client.post("/auth/login", content=b"{not json",
                               headers={"Content-Type": "application/json"})

    assert response.status_code == 422
    assert list(response.json()) == ["detail"]
    assert isinstance(response.json()["detail"], str)


def test_domain_not_found_maps_to_404(api, admin_headers):
    api.retrain.error = NotFoundError("training file 9 does not exist")

    response = api.client.post("/retrain", headers=admin_headers)

    assert response.status_code == 404
    assert response.json() == {"detail": "training file 9 does not exist"}


def test_unexpected_error_returns_500_without_leaking_internals(api, admin_headers):
    api.retrain.error = RuntimeError("database password is hunter2")
    client = TestClient(api.client.app, raise_server_exceptions=False)

    response = client.post("/retrain", headers=admin_headers)

    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_openapi_documents_every_endpoint(api):
    paths = api.client.get("/openapi.json").json()["paths"]

    assert {"/auth/register", "/auth/login", "/predict", "/predictions", "/metrics",
            "/training-files", "/retrain", "/health"} <= set(paths)
    assert "bearer" in str(api.client.get("/openapi.json").json()["components"]).lower()


def test_create_app_works_without_a_startup_hook(api):
    use_cases: ApiUseCases = api.client.app.state.use_cases
    app = create_app(use_cases, cors_origins=[])

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

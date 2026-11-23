"""POST /auth/register, POST /auth/login and bearer-token handling."""

import pytest

from app.adapters.outbound.security.jwt_token_service import JwtTokenService
from app.application.ports.token_service import TokenClaims
from app.domain.entities import Role
from tests.integration.api.conftest import ADMIN_EMAIL, login_headers


def test_register_creates_a_user_with_role_user(api):
    response = api.client.post("/auth/register",
                               json={"email": " New@Example.com ", "password": "secret-pass"})

    assert response.status_code == 201
    body = response.json()
    assert body["email"] == "new@example.com"
    assert body["role"] == "user"
    assert isinstance(body["id"], int)
    assert "password" not in body and "password_hash" not in body
    assert api.users.get_by_email("new@example.com").password_hash.startswith("$2b$")


def test_register_ignores_a_requested_admin_role(api):
    response = api.client.post("/auth/register", json={"email": "sneaky@example.com",
                                                       "password": "secret-pass",
                                                       "role": "admin"})

    assert response.status_code == 201
    assert response.json()["role"] == "user"


def test_register_twice_returns_409(api):
    payload = {"email": "dup@example.com", "password": "secret-pass"}
    api.client.post("/auth/register", json=payload)

    response = api.client.post("/auth/register", json=payload)

    assert response.status_code == 409
    assert response.json() == {"detail": "email is already registered"}


def test_register_with_short_password_returns_422(api):
    response = api.client.post("/auth/register", json={"email": "a@b.com", "password": "short"})

    assert response.status_code == 422
    assert response.json() == {"detail": "password must have at least 8 characters"}


def test_register_without_fields_returns_422_with_a_detail_string(api):
    response = api.client.post("/auth/register", json={})

    assert response.status_code == 422
    assert list(response.json()) == ["detail"]
    assert "email" in response.json()["detail"]
    assert "password" in response.json()["detail"]


def test_login_returns_a_bearer_token(api):
    api.client.post("/auth/register", json={"email": "me@example.com", "password": "secret-pass"})

    response = api.client.post("/auth/login",
                               json={"email": "me@example.com", "password": "secret-pass"})

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"access_token", "token_type"}
    assert body["token_type"] == "bearer"
    assert body["access_token"].count(".") == 2


@pytest.mark.parametrize("email,password", [("me@example.com", "wrong-pass"),
                                            ("nobody@example.com", "secret-pass")])
def test_login_with_bad_credentials_returns_401(api, email, password):
    api.client.post("/auth/register", json={"email": "me@example.com", "password": "secret-pass"})

    response = api.client.post("/auth/login", json={"email": email, "password": password})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid email or password"}


def test_admin_logs_in_with_its_configured_credentials(api, admin_headers):
    response = api.client.get("/predictions", headers=admin_headers)

    assert response.status_code == 200


@pytest.mark.parametrize("path,method", [("/predict", "post"), ("/predictions", "get"),
                                         ("/metrics", "get"), ("/retrain", "post"),
                                         ("/training-files", "post")])
def test_protected_routes_without_token_return_401(api, path, method):
    response = getattr(api.client, method)(path)

    assert response.status_code == 401
    assert response.json() == {"detail": "Not authenticated"}
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize("header", ["Bearer not-a-jwt", "Bearer a.b.c", "Basic dXNlcjpwYXNz",
                                    "Bearer"])
def test_bad_token_returns_401(api, header):
    response = api.client.get("/metrics", headers={"Authorization": header})

    assert response.status_code == 401
    assert list(response.json()) == ["detail"]
    assert response.headers["www-authenticate"] == "Bearer"


def test_token_signed_with_another_secret_returns_401(api):
    forged = JwtTokenService(secret="attacker-secret-long-enough-0123456789").issue(
        TokenClaims(user_id=1, role=Role.ADMIN))

    response = api.client.post("/retrain", headers={"Authorization": f"Bearer {forged}"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid token"}
    assert api.retrain.calls == 0


def test_expired_token_returns_401(api):
    from datetime import UTC, datetime, timedelta

    from tests.integration.api.conftest import JWT_SECRET

    old = JwtTokenService(secret=JWT_SECRET, expire_minutes=1,
                          clock=lambda: datetime.now(UTC) - timedelta(hours=1))
    token = old.issue(TokenClaims(user_id=1, role=Role.ADMIN))

    response = api.client.get("/metrics", headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Token has expired"}


def test_admin_email_cannot_be_registered_again(api):
    response = api.client.post("/auth/register",
                               json={"email": ADMIN_EMAIL, "password": "takeover-pass"})

    assert response.status_code == 409
    assert login_headers(api.client, ADMIN_EMAIL, "admin-password")

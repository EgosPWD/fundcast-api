"""Bcrypt hasher and JWT token service adapters."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from app.adapters.outbound.security.bcrypt_hasher import BcryptPasswordHasher
from app.adapters.outbound.security.jwt_token_service import JwtTokenService
from app.application.ports.token_service import TokenClaims
from app.domain.entities import Role
from app.domain.errors import AuthenticationError, InvalidInputError

SECRET = "test-secret-that-is-long-enough-for-hs256-0123456789"


@pytest.fixture
def hasher() -> BcryptPasswordHasher:
    return BcryptPasswordHasher(rounds=4)


def test_bcrypt_hash_verifies_the_right_password_only(hasher):
    password_hash = hasher.hash("correct horse battery")

    assert password_hash != "correct horse battery"
    assert password_hash.startswith("$2b$04$")
    assert hasher.verify("correct horse battery", password_hash) is True
    assert hasher.verify("wrong password", password_hash) is False


def test_bcrypt_salts_every_hash(hasher):
    assert hasher.hash("same password") != hasher.hash("same password")


def test_bcrypt_verify_returns_false_for_a_malformed_hash(hasher):
    assert hasher.verify("anything", "not-a-bcrypt-hash") is False


def test_bcrypt_rejects_passwords_longer_than_72_bytes(hasher):
    with pytest.raises(InvalidInputError, match="72 bytes"):
        hasher.hash("x" * 73)


def test_bcrypt_verify_is_false_for_passwords_longer_than_72_bytes(hasher):
    password_hash = hasher.hash("x" * 72)

    assert hasher.verify("x" * 73, password_hash) is False


def test_jwt_round_trips_the_claims():
    tokens = JwtTokenService(secret=SECRET, expire_minutes=5)

    token = tokens.issue(TokenClaims(user_id=7, role=Role.ADMIN))

    assert tokens.decode(token) == TokenClaims(user_id=7, role=Role.ADMIN)


def test_jwt_carries_an_expiry():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    tokens = JwtTokenService(secret=SECRET, expire_minutes=30, clock=lambda: now)

    payload = jwt.decode(tokens.issue(TokenClaims(user_id=1, role=Role.USER)), SECRET,
                         algorithms=["HS256"], options={"verify_exp": False})

    assert payload["sub"] == "1"
    assert payload["role"] == "user"
    assert payload["exp"] == int((now + timedelta(minutes=30)).timestamp())


def test_expired_token_raises_authentication_error():
    past = datetime.now(UTC) - timedelta(hours=2)
    issuer = JwtTokenService(secret=SECRET, expire_minutes=60, clock=lambda: past)
    token = issuer.issue(TokenClaims(user_id=1, role=Role.USER))

    with pytest.raises(AuthenticationError, match="expired"):
        JwtTokenService(secret=SECRET).decode(token)


def test_token_signed_with_another_secret_is_rejected():
    token = JwtTokenService(secret="another-secret-also-long-enough-0123456789").issue(
        TokenClaims(user_id=1, role=Role.ADMIN))

    with pytest.raises(AuthenticationError):
        JwtTokenService(secret=SECRET).decode(token)


@pytest.mark.parametrize("token", ["", "garbage", "a.b.c"])
def test_malformed_token_is_rejected(token):
    with pytest.raises(AuthenticationError):
        JwtTokenService(secret=SECRET).decode(token)


@pytest.mark.parametrize("payload", [
    {"sub": "1", "role": "superuser"},
    {"sub": "abc", "role": "user"},
    {"role": "user"},
    {"sub": "1"},
])
def test_token_with_invalid_claims_is_rejected(payload):
    exp = datetime.now(UTC) + timedelta(minutes=5)
    token = jwt.encode({**payload, "exp": exp}, SECRET, algorithm="HS256")

    with pytest.raises(AuthenticationError):
        JwtTokenService(secret=SECRET).decode(token)


def test_token_without_expiry_is_rejected():
    token = jwt.encode({"sub": "1", "role": "user"}, SECRET, algorithm="HS256")

    with pytest.raises(AuthenticationError):
        JwtTokenService(secret=SECRET).decode(token)


def test_unsigned_none_algorithm_token_is_rejected():
    exp = datetime.now(UTC) + timedelta(minutes=5)
    token = jwt.encode({"sub": "1", "role": "admin", "exp": exp}, key=None, algorithm="none")

    with pytest.raises(AuthenticationError):
        JwtTokenService(secret=SECRET).decode(token)

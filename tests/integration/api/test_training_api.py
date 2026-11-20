"""POST /training-files and POST /retrain (admin only)."""

import pytest

from app.domain.entities import MAX_TRAINING_FILE_BYTES
from app.domain.errors import (
    InsufficientTrainingDataError,
    InvalidTrainingFileError,
    RetrainInProgressError,
)

CSV = b"announced_date,year\n2025-01-01,2025\n"


def upload(api, headers, filename="rounds.csv", content=CSV):
    return api.client.post("/training-files", headers=headers,
                           files={"file": (filename, content, "text/csv")})


def test_user_cannot_upload_training_files(api, user_headers):
    response = upload(api, user_headers)

    assert response.status_code == 403
    assert response.json() == {"detail": "Admin role required"}
    assert api.upload.calls == []


def test_user_cannot_retrain(api, user_headers):
    response = api.client.post("/retrain", headers=user_headers)

    assert response.status_code == 403
    assert response.json() == {"detail": "Admin role required"}
    assert api.retrain.calls == 0


def test_admin_uploads_a_csv(api, admin_headers):
    response = upload(api, admin_headers)

    assert response.status_code == 201
    assert response.json() == {"id": 1, "filename": "rounds.csv", "rows": 10,
                               "rows_inserted": 8, "rows_rejected": 2, "status": "loaded"}
    assert api.upload.calls == [(1, "rounds.csv", CSV)]


def test_invalid_csv_returns_422(api, admin_headers):
    api.upload.error = InvalidTrainingFileError("missing required columns: amount_usd")

    response = upload(api, admin_headers)

    assert response.status_code == 422
    assert response.json() == {"detail": "missing required columns: amount_usd"}


@pytest.mark.parametrize("filename", ["rounds.txt", "rounds.csv.exe", "rounds"])
def test_non_csv_upload_returns_422_before_the_use_case(api, admin_headers, filename):
    response = upload(api, admin_headers, filename=filename)

    assert response.status_code == 422
    assert response.json() == {"detail": "file must have a .csv extension"}
    assert api.upload.calls == []


def test_oversize_upload_returns_422_before_the_use_case(api, admin_headers):
    response = upload(api, admin_headers, content=b"x" * (MAX_TRAINING_FILE_BYTES + 1))

    assert response.status_code == 422
    assert response.json() == {"detail": "file exceeds the 20 MB limit"}
    assert api.upload.calls == []


def test_upload_exactly_at_the_limit_reaches_the_use_case(api, admin_headers):
    response = upload(api, admin_headers, content=b"x" * MAX_TRAINING_FILE_BYTES)

    assert response.status_code == 201
    assert len(api.upload.calls[0][2]) == MAX_TRAINING_FILE_BYTES


def test_upload_without_file_returns_422(api, admin_headers):
    response = api.client.post("/training-files", headers=admin_headers)

    assert response.status_code == 422
    assert list(response.json()) == ["detail"]
    assert "file" in response.json()["detail"]


def test_admin_retrains_and_the_new_version_is_activated(api, admin_headers):
    response = api.client.post("/retrain", headers=admin_headers)

    assert response.status_code == 201
    assert response.json() == {
        "version": 2, "activated": True,
        "metrics": {"mae_log": 0.35, "r2": 0.615, "rmse_log": 0.495, "mdape": 0.529,
                    "coverage": 0.473},
        "n_train": 12758, "n_test": 3190, "message": None,
    }


def test_retrain_not_activated_explains_why(api, admin_headers):
    api.retrain.activated = False

    response = api.client.post("/retrain", headers=admin_headers)

    assert response.status_code == 201
    body = response.json()
    assert body["activated"] is False
    assert "inactive" in body["message"]
    assert "MAE_log" in body["message"]


def test_retrain_in_progress_returns_409(api, admin_headers):
    api.retrain.error = RetrainInProgressError("A retraining is already in progress")

    response = api.client.post("/retrain", headers=admin_headers)

    assert response.status_code == 409
    assert response.json() == {"detail": "A retraining is already in progress"}


def test_retrain_with_insufficient_data_returns_422(api, admin_headers):
    api.retrain.error = InsufficientTrainingDataError("a stratum has fewer than 2 usable rows")

    response = api.client.post("/retrain", headers=admin_headers)

    assert response.status_code == 422
    assert response.json() == {"detail": "a stratum has fewer than 2 usable rows"}

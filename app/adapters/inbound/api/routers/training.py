from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.adapters.inbound.api import deps
from app.adapters.inbound.api.routers.metrics import metrics_schema
from app.adapters.inbound.api.schemas import ErrorResponse, RetrainResponse, TrainingFileResponse
from app.adapters.inbound.api.security import AdminUser
from app.adapters.inbound.api.use_case_ports import RetrainModelUseCase, UploadTrainingFileUseCase
from app.domain.entities import MAX_TRAINING_FILE_BYTES
from app.domain.errors import InvalidTrainingFileError

router = APIRouter(tags=["training"])
ADMIN_ERRORS = {401: {"model": ErrorResponse}, 403: {"model": ErrorResponse},
                422: {"model": ErrorResponse}}
NOT_ACTIVATED_MESSAGE = (
    "The new version was stored inactive: its MAE_log is worse than the active version's "
    "MAE_log + 0.01. The API keeps predicting with the active version."
)


@router.post("/training-files", status_code=201, response_model=TrainingFileResponse,
             responses=ADMIN_ERRORS)
def upload_training_file(
    claims: AdminUser,
    upload_use_case: Annotated[UploadTrainingFileUseCase, Depends(deps.upload_training_file)],
    file: Annotated[UploadFile, File(description="CSV with the columns of AGENTS.md section 13")],
):
    """Validate a CSV and insert its rows into funding_rounds."""
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise InvalidTrainingFileError("file must have a .csv extension")
    # Read at most limit + 1 bytes: enough to detect an oversize file without buffering it all.
    content = file.file.read(MAX_TRAINING_FILE_BYTES + 1)
    if len(content) > MAX_TRAINING_FILE_BYTES:
        raise InvalidTrainingFileError("file exceeds the 20 MB limit")
    loaded = upload_use_case.execute(claims.user_id, filename, content)
    return TrainingFileResponse(id=loaded.id, filename=loaded.filename, rows=loaded.rows,
                                rows_inserted=loaded.rows_inserted,
                                rows_rejected=loaded.rows_rejected, status=str(loaded.status))


@router.post("/retrain", status_code=201, response_model=RetrainResponse,
             responses={**ADMIN_ERRORS, 409: {"model": ErrorResponse}})
def retrain(_: AdminUser,
            retrain_use_case: Annotated[RetrainModelUseCase, Depends(deps.retrain_model)]):
    """Retrain on every row of funding_rounds with a new stratified 80/20 split."""
    outcome = retrain_use_case.execute()
    version = outcome.model_version
    return RetrainResponse(
        version=version.version, activated=outcome.activated,
        metrics=metrics_schema(version.metrics), n_train=version.n_train,
        n_test=version.n_test, message=None if outcome.activated else NOT_ACTIVATED_MESSAGE,
    )

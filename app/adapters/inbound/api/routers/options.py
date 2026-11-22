from typing import Annotated

from fastapi import APIRouter, Depends

from app.adapters.inbound.api import deps
from app.adapters.inbound.api.schemas import ErrorResponse, OptionsResponse
from app.adapters.inbound.api.security import AnyUser
from app.adapters.inbound.api.use_case_ports import GetOptionsUseCase

router = APIRouter(tags=["predictions"])


@router.get("/options", response_model=OptionsResponse,
            responses={401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
def get_options(_: AnyUser,
                options_use_case: Annotated[GetOptionsUseCase, Depends(deps.get_options)]):
    """Sectors, stages and countries present in the data, for the prediction form.

    The model only knows these exact strings: any other text is treated as an unknown
    category and gets an average estimate, so offer them as a closed list."""
    options = options_use_case.execute()
    return OptionsResponse(sectors=options.sectors, stages=options.stages,
                           countries=options.countries)

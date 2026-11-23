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
    """Categories of the active model for the prediction form, plus the cities of each country.

    Offer them as closed lists: the model matches categories by exact text, and any other
    text is treated as unknown. After a retrain activates a new version the lists change."""
    options = options_use_case.execute()
    return OptionsResponse(
        stages=options.stages, sectors=options.sectors, countries=options.countries,
        cities_by_country=options.cities_by_country, lead_investors=options.lead_investors,
        news_sources=options.news_sources, model_version=options.model_version)

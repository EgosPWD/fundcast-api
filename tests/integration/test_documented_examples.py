"""Every documented /predict example must use categories the base model knows (S7)."""

import json
import re
from pathlib import Path

import pytest

from app.adapters.inbound.api.schemas import PredictRequest
from app.adapters.outbound.ml.sklearn_predictor import SklearnPredictor
from tests.integration.api.test_predict_api import CONTRACT_EXAMPLE

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
BASELINE = ARTIFACTS_DIR / "modelo_rondas.joblib"
FIELDS = {"stage": "stages", "sector": "sectors", "country": "countries", "city": "cities",
          "lead_investor": "lead_investors", "news_source": "news_sources"}


def unknown_values(example: dict) -> list[str]:
    vocabulary = SklearnPredictor(str(ARTIFACTS_DIR)).vocabulary(str(BASELINE))
    return [f"{field}={example[field]!r}" for field, attribute in FIELDS.items()
            if example.get(field) is not None
            and example[field] not in getattr(vocabulary, attribute)]


def json_examples(markdown: Path) -> list[dict]:
    blocks = re.findall(r"```json\n(.*?)```", markdown.read_text(encoding="utf-8"), re.DOTALL)
    parsed = [json.loads(block) for block in blocks]
    return [b for b in parsed if isinstance(b, dict) and "founded_year" in b]


def test_the_contract_example_used_by_the_tests_has_known_categories():
    assert unknown_values(CONTRACT_EXAMPLE) == []


def test_the_openapi_schema_examples_have_known_categories():
    properties = PredictRequest.model_json_schema()["properties"]
    example = {name: spec["examples"][0] for name, spec in properties.items()
               if "examples" in spec}

    assert unknown_values(example) == []


@pytest.mark.parametrize("name", ["AGENTS.md", "README.md"])
def test_the_markdown_predict_examples_have_known_categories(name):
    examples = json_examples(ROOT / name)

    assert examples, f"{name} has no /predict example"
    for example in examples:
        assert unknown_values(example) == []

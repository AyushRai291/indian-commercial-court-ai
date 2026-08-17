from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from qdrant_client.models import Distance

from legal_rag.models import Case, Paragraph
from legal_rag.vector import QdrantParagraphIndex


class FakeQdrantClient:
    def __init__(self, *, exists: bool, size: int = 3, distance=Distance.COSINE):
        self.exists = exists
        self.deleted = False
        self.created = False
        self.collection = SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=size, distance=distance)
                )
            )
        )

    def collection_exists(self, _name: str) -> bool:
        return self.exists

    def get_collection(self, _name: str):
        return self.collection

    def create_collection(self, **_kwargs) -> None:
        self.exists = True
        self.created = True

    def delete_collection(self, _name: str) -> None:
        self.exists = False
        self.deleted = True


def _index(client: FakeQdrantClient) -> QdrantParagraphIndex:
    return QdrantParagraphIndex(
        url="http://unused.test",
        collection_name="paragraphs",
        client=client,
    )


def _load_index_script() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "index_vectors.py"
    spec = importlib.util.spec_from_file_location("index_vectors_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_existing_qdrant_collection_requires_cosine_distance() -> None:
    index = _index(FakeQdrantClient(exists=True, distance=Distance.DOT))

    with pytest.raises(ValueError, match="requires cosine"):
        index.ensure_collection(3)


def test_search_validation_does_not_create_a_missing_collection() -> None:
    client = FakeQdrantClient(exists=False)

    with pytest.raises(ValueError, match="index_vectors.py"):
        _index(client).validate_collection(3)

    assert client.created is False


def test_explicit_recreate_replaces_the_collection() -> None:
    client = FakeQdrantClient(exists=True)

    _index(client).ensure_collection(3, recreate=True)

    assert client.deleted is True
    assert client.created is True


def test_paragraph_payload_has_exact_required_keys() -> None:
    script = _load_index_script()
    case = Case(
        id=11,
        title="Acme Ltd. v. Zenith Ltd.",
        case_number="CS(COMM) 1/2025",
        court="Delhi High Court",
        judgment_date=date(2025, 2, 3),
        source="test",
        raw_text="The claim is decreed.",
        document_hash="a" * 64,
    )
    paragraph = Paragraph(
        id=19,
        case_id=11,
        paragraph_number=7,
        page_number=2,
        text="The claim is decreed.",
        text_hash="b" * 64,
    )

    payload = script.paragraph_payload(paragraph, case)

    assert payload == {
        "case_id": 11,
        "title": "Acme Ltd. v. Zenith Ltd.",
        "court": "Delhi High Court",
        "year": 2025,
        "paragraph_number": 7,
        "text": "The claim is decreed.",
    }

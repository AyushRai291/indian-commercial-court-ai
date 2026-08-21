from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from sqlalchemy import select

from legal_rag.corpus import generate_paragraph_uid, normalize_record
from legal_rag.database import get_engine, get_session_factory, init_db
from legal_rag.models import Case, Paragraph
from legal_rag.services.ingestion import insert_case
from legal_rag.vector import ParagraphVectorRecord, QdrantParagraphIndex


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
    paragraph_uid = generate_paragraph_uid("a" * 64, 7, "b" * 64)
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
        paragraph_uid=paragraph_uid,
        case_id=11,
        paragraph_number=7,
        page_number=2,
        text="The claim is decreed.",
        text_hash="b" * 64,
    )

    payload = script.paragraph_payload(paragraph, case)

    assert payload == {
        "paragraph_uid": paragraph_uid,
        "case_id": 11,
        "title": "Acme Ltd. v. Zenith Ltd.",
        "court": "Delhi High Court",
        "year": 2025,
        "paragraph_number": 7,
        "text": "The claim is decreed.",
    }


def test_qdrant_point_struct_persists_uuid_id_and_payload() -> None:
    paragraph_uid = generate_paragraph_uid("a" * 64, 7, "b" * 64)
    client = QdrantClient(":memory:")
    paragraph_index = QdrantParagraphIndex(
        url="http://unused.test",
        collection_name="paragraphs",
        client=client,
    )
    paragraph_index.ensure_collection(3)

    assert paragraph_index.upsert(
        [
            ParagraphVectorRecord(
                point_id=paragraph_uid,
                vector=[1.0, 0.0, 0.0],
                payload={"paragraph_uid": paragraph_uid},
            )
        ]
    ) == 1

    stored = client.retrieve(
        collection_name="paragraphs",
        ids=[paragraph_uid],
        with_payload=True,
    )
    assert len(stored) == 1
    assert stored[0].id == paragraph_uid
    assert stored[0].payload == {"paragraph_uid": paragraph_uid}


def test_qdrant_point_ids_are_scrolled_without_embedding_downloads() -> None:
    identifiers = [
        "00000000-0000-5000-8000-000000000001",
        "00000000-0000-5000-8000-000000000002",
        "00000000-0000-5000-8000-000000000003",
    ]

    class FakeScrollingQdrant:
        def scroll(self, *, offset=None, **_kwargs):
            if offset is None:
                return (
                    [SimpleNamespace(id=value) for value in identifiers[:2]],
                    identifiers[1],
                )
            return ([SimpleNamespace(id=identifiers[2])], None)

    paragraph_index = QdrantParagraphIndex(
        url="http://unused.test",
        collection_name="paragraphs",
        client=FakeScrollingQdrant(),
    )

    assert paragraph_index.list_point_ids(batch_size=2) == identifiers


def test_index_vectors_uses_paragraph_uid_for_qdrant_point_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _load_index_script()
    database_path = tmp_path / "vectors.sqlite3"
    database_url = f"sqlite+pysqlite:///{database_path.as_posix()}"
    engine = get_engine(database_url)
    init_db(engine)
    session_factory = get_session_factory(engine)
    canonical = normalize_record(
        {
            "title": "Acme Ltd. v. Zenith Ltd.",
            "court": "Delhi High Court",
            "judgment_date": "2025-02-03",
            "raw_text": "1. The claim is decreed.",
        }
    )

    with session_factory.begin() as session:
        insert_case(session, canonical)
    with session_factory() as session:
        stored_paragraph = session.scalars(select(Paragraph)).one()
        numeric_id = stored_paragraph.id
        paragraph_uid = stored_paragraph.paragraph_uid

    captured_records = []

    class FakeEmbeddingProvider:
        dimension = 3

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def embed_documents(self, texts, *, batch_size):
            assert texts == ["The claim is decreed."]
            assert batch_size == 10
            return [[1.0, 0.0, 0.0]]

    class CapturingParagraphIndex:
        def __init__(self, **_kwargs) -> None:
            pass

        def ensure_collection(self, vector_size, *, recreate=False) -> None:
            assert vector_size == 3
            assert recreate is False

        def upsert(self, records) -> int:
            captured_records.extend(records)
            return len(records)

    settings = SimpleNamespace(
        database_url=database_url,
        embedding_model="test-model",
        embedding_dimension=3,
        qdrant_url="http://unused.test",
        qdrant_api_key=None,
        qdrant_collection="paragraphs",
    )
    monkeypatch.setattr(script, "get_settings", lambda: settings)
    monkeypatch.setattr(
        script,
        "SentenceTransformerEmbeddingProvider",
        FakeEmbeddingProvider,
    )
    monkeypatch.setattr(script, "QdrantParagraphIndex", CapturingParagraphIndex)

    try:
        assert script.index_vectors(
            batch_size=10,
            collection=None,
            model=None,
        ) == 1
    finally:
        engine.dispose()

    assert len(captured_records) == 1
    record = captured_records[0]
    assert record.point_id == paragraph_uid
    assert record.point_id != numeric_id
    assert record.payload["paragraph_uid"] == paragraph_uid

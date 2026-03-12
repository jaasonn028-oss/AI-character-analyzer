import json
import pytest
import app as app_module
from app import app, analyze_with_fallback


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Index route
# ---------------------------------------------------------------------------

def test_index_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI Character Analyzer" in response.data


# ---------------------------------------------------------------------------
# /analyze – validation
# ---------------------------------------------------------------------------

def test_analyze_missing_body(client):
    response = client.post("/analyze", content_type="application/json", data="")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_analyze_empty_text(client):
    response = client.post(
        "/analyze",
        json={"text": ""},
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_analyze_text_too_short(client):
    response = client.post("/analyze", json={"text": "short"})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_analyze_text_too_long(client):
    response = client.post("/analyze", json={"text": "x" * 5001})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


# ---------------------------------------------------------------------------
# /analyze – fallback (demo) mode
# ---------------------------------------------------------------------------

def test_analyze_returns_demo_result(client, monkeypatch):
    # Ensure no OpenAI client is created so fallback is used
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(app_module, "_openai_client", None)

    text = "Hermione Granger is a brilliant and kind witch who is loyal to her friends."
    response = client.post("/analyze", json={"text": text})
    assert response.status_code == 200
    data = response.get_json()
    assert data.get("demo_mode") is True
    assert "summary" in data
    assert "traits" in data
    assert "strengths" in data
    assert "weaknesses" in data
    assert "archetype" in data
    assert "motivation" in data
    assert "mbti" in data


# ---------------------------------------------------------------------------
# analyze_with_fallback unit tests
# ---------------------------------------------------------------------------

def test_fallback_detects_positive_traits():
    result = analyze_with_fallback(
        "She is brave, kind, and wise — a true hero among her peers."
    )
    trait_names = [t.lower() for t in result["traits"]]
    assert any("courageous" in t or "kind" in t or "wise" in t for t in trait_names)


def test_fallback_detects_villain_archetype():
    result = analyze_with_fallback(
        "He is a dark and evil villain who is corrupt to the core and full of malice."
    )
    assert result["archetype"] == "Shadow"


def test_fallback_detects_mentor_archetype():
    result = analyze_with_fallback(
        "The old wise mentor guided the young hero with patience and wisdom."
    )
    assert result["archetype"] == "Mentor"


def test_fallback_returns_defaults_for_unknown_text():
    result = analyze_with_fallback(
        "A mysterious figure walked slowly through the misty forest at midnight."
    )
    assert len(result["traits"]) > 0
    assert len(result["strengths"]) > 0
    assert len(result["weaknesses"]) > 0
    assert result["archetype"] in {"Hero", "Mentor", "Shadow", "Trickster", "Lover"}

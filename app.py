import json
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key")

_openai_client = None


def get_openai_client():
    """Return a cached OpenAI client, or None if API key is not configured."""
    global _openai_client
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "your_openai_api_key_here":
        return None
    if _openai_client is None:
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def analyze_with_openai(character_text: str) -> dict:
    """Send character text to OpenAI and return a structured analysis."""
    client = get_openai_client()
    prompt = (
        "You are an expert character analyst. Analyze the following character description "
        "or passage and provide a structured personality analysis. Return a JSON object with "
        "these exact keys:\n"
        "- name: character name or 'Unknown' if not mentioned\n"
        "- summary: 2-3 sentence overall character summary\n"
        "- traits: list of 5 key personality traits (strings)\n"
        "- strengths: list of 3 character strengths (strings)\n"
        "- weaknesses: list of 3 character weaknesses (strings)\n"
        "- archetype: the classic character archetype (e.g., Hero, Mentor, Trickster)\n"
        "- motivation: the character's core motivation (1 sentence)\n"
        "- mbti: the most likely MBTI type with a brief explanation\n\n"
        f"Character text:\n{character_text}\n\n"
        "Respond with valid JSON only."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    return json.loads(response.choices[0].message.content)


def analyze_with_fallback(character_text: str) -> dict:
    """
    Provide a basic rule-based analysis when OpenAI is not configured.
    This is used for demo purposes or when no API key is available.
    """
    text_lower = character_text.lower()
    word_count = len(character_text.split())

    traits = []
    strengths = []
    weaknesses = []

    positive_keywords = {
        "brave": "Courageous",
        "courageous": "Courageous",
        "kind": "Kind-hearted",
        "gentle": "Gentle",
        "wise": "Wise",
        "smart": "Intelligent",
        "intelligent": "Intelligent",
        "loyal": "Loyal",
        "honest": "Honest",
        "compassionate": "Compassionate",
        "determined": "Determined",
        "creative": "Creative",
        "patient": "Patient",
        "generous": "Generous",
        "humble": "Humble",
    }

    negative_keywords = {
        "angry": "Quick-tempered",
        "arrogant": "Arrogant",
        "stubborn": "Stubborn",
        "impulsive": "Impulsive",
        "reckless": "Reckless",
        "jealous": "Jealous",
        "selfish": "Selfish",
        "cowardly": "Cowardly",
        "deceitful": "Deceitful",
        "lazy": "Lazy",
    }

    found_traits = set()
    for keyword, trait in positive_keywords.items():
        if keyword in text_lower and trait not in found_traits:
            traits.append(trait)
            strengths.append(trait)
            found_traits.add(trait)

    for keyword, trait in negative_keywords.items():
        if keyword in text_lower and trait not in found_traits:
            traits.append(trait)
            weaknesses.append(trait)
            found_traits.add(trait)

    if not traits:
        traits = ["Complex", "Multi-dimensional", "Nuanced"]
    if not strengths:
        strengths = ["Resilient", "Adaptable", "Resourceful"]
    if not weaknesses:
        weaknesses = ["Conflicted", "Unpredictable", "Guarded"]

    archetype = "Hero"
    if any(w in text_lower for w in ["mentor", "teacher", "guide", "wise"]):
        archetype = "Mentor"
    elif any(w in text_lower for w in ["villain", "evil", "dark", "corrupt"]):
        archetype = "Shadow"
    elif any(w in text_lower for w in ["trickster", "jester", "jokester", "funny"]):
        archetype = "Trickster"
    elif any(w in text_lower for w in ["love", "romance", "heart", "care"]):
        archetype = "Lover"

    lines = [s.strip() for s in character_text.split(".") if s.strip()]
    summary = ". ".join(lines[:2]) + "." if lines else character_text[:200]
    if word_count > 10:
        summary = (
            f"This character presents a {archetype.lower()} archetype with "
            f"{len(traits)} identifiable personality traits. "
            f"They appear to be a {traits[0].lower()} individual with notable depth."
        )

    return {
        "name": "Unknown",
        "summary": summary,
        "traits": traits[:5],
        "strengths": strengths[:3],
        "weaknesses": weaknesses[:3],
        "archetype": archetype,
        "motivation": "To fulfill their role in the story and overcome personal challenges.",
        "mbti": "INFJ – The Advocate (requires AI analysis for accurate typing)",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True)
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "Please provide character text to analyze."}), 400

    text = data["text"].strip()
    if len(text) < 20:
        return jsonify({"error": "Please provide at least 20 characters of text."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text must be 5000 characters or fewer."}), 400

    client = get_openai_client()
    if client is not None:
        try:
            result = analyze_with_openai(text)
        except Exception as exc:
            return jsonify({"error": f"AI analysis failed: {exc}"}), 500
    else:
        result = analyze_with_fallback(text)
        result["demo_mode"] = True

    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

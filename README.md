# AI Character Analyzer

A cloud-based AI character analyzer that reveals the personality profile of any fictional (or real) character from a text description or narrative passage.

## Features

- **AI-powered analysis** via OpenAI GPT-4o-mini
- **Demo mode** (rule-based fallback) when no API key is provided
- **Personality traits, strengths & weaknesses** breakdown
- **Character archetype** classification (Hero, Mentor, Shadow, Trickster, Lover…)
- **MBTI type** suggestion
- **Core motivation** summary
- Clean, modern web UI

## Quick Start

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys) *(optional — the app runs in demo mode without one)*

### Local setup

```bash
# 1. Clone the repo
git clone https://github.com/jaasonn028-oss/AI-character-analyzer.git
cd AI-character-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run the app
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Usage

1. Paste a character description or narrative passage (20 – 5000 characters) into the text box.
2. Click **Analyze Character**.
3. View the personality profile including traits, archetype, MBTI type, motivation, strengths, and weaknesses.

**Example input:**
> Hermione Granger is a brilliant and hardworking witch who values knowledge above all else. She is fiercely loyal to her friends, but can sometimes be overly rigid and rule-bound when facing moral grey areas.

## Deployment

The app is a standard Flask application compatible with any cloud that supports Python:

| Platform | Command |
|----------|---------|
| Heroku / Railway / Render | Set `OPENAI_API_KEY` as an env var; deploy with `gunicorn app:app` |
| Google Cloud Run / AWS ECS | Containerise with the Dockerfile of your choice; expose port `$PORT` |

## Running tests

```bash
pip install pytest
pytest test_app.py -v
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | No | OpenAI API key — enables full AI analysis |
| `FLASK_SECRET_KEY` | Recommended | Random secret key for Flask sessions |
| `PORT` | No | Port to listen on (default: `5000`) |


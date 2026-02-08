"""
Ad Generator: Takes game events from Gemini video analysis and generates
real-time advertisement ideas using Groq (Llama 3.3 70B) for speed.

Pipeline: Game Event → Significance Filter → Ad Generation → Output
"""
import json
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parent / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You are an elite real-time sports marketing AI for small and medium businesses.

When given a game event from a Super Bowl / NFL match, you instantly create
a short, punchy advertisement or promotion idea that a business can post
on social media or display in-store within seconds.

Rules:
- Keep it SHORT: 1-2 sentences max for the ad copy
- Make it TIMELY: directly reference what just happened in the game
- Make it CLEVER: use wordplay, puns, or emotional hooks tied to the moment
- Include a concrete PROMO suggestion (discount, deal, freebie, etc.)
- Output valid JSON with these fields:
  {
    "is_significant": true/false,
    "event_type": "touchdown|fumble|interception|big_play|tackle|penalty|injury|halftime|commercial_break|other",
    "ad_copy": "The catchy ad text",
    "promo_suggestion": "Specific promotion idea",
    "social_hashtags": ["#hashtag1", "#hashtag2"],
    "urgency": "high|medium|low"
  }
- Be GENEROUS with is_significant. ANY play that involves action should be true:
  tackles, completed passes, runs, sacks, big hits — all count.
  Only set is_significant to false if literally nothing happens
  (pre-snap lineup, camera panning, scoreboard graphic, no play at all).
- ONLY output the JSON object, nothing else.
"""


def _build_event_prompt(event: dict, business_name: str = "", business_type: str = "") -> str:
    """Build the user prompt for a single game event."""
    biz_context = ""
    if business_name or business_type:
        biz_context = f"\nBusiness: {business_name or 'Local business'} ({business_type or 'general'})\n"

    return (
        f"Game event at {event.get('window', 'unknown time')}:\n"
        f"{event.get('analysis', 'No description')}\n"
        f"{biz_context}\n"
        "Generate an ad for this moment. Respond with JSON only."
    )


def generate_ad(
    event: dict,
    business_name: str = "",
    business_type: str = "",
    client: Optional[Groq] = None,
) -> dict:
    """Generate an ad for a single game event. Returns parsed JSON dict."""
    if client is None:
        client = Groq()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_event_prompt(event, business_name, business_type)},
        ],
        temperature=0.7,
        max_tokens=300,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"is_significant": False, "raw_response": raw}


def generate_ad_stream(
    event: dict,
    business_name: str = "",
    business_type: str = "",
    client: Optional[Groq] = None,
):
    """Stream ad generation for a single event (yields text chunks)."""
    if client is None:
        client = Groq()

    stream = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_event_prompt(event, business_name, business_type)},
        ],
        temperature=0.7,
        max_tokens=300,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


def process_all_events(
    events: List[dict],
    business_name: str = "",
    business_type: str = "",
    output_json: str = "ad_results.json",
) -> List[dict]:
    """Process all game events, generate ads for significant ones, save to JSON."""
    client = Groq()
    results = []

    for i, event in enumerate(events):
        window = event.get("window", f"segment {i}")
        print(f"[{i+1}/{len(events)}] Processing {window} ...")

        ad = generate_ad(event, business_name, business_type, client)
        ad["source_event"] = event

        if ad.get("is_significant"):
            print(f"  → AD: {ad.get('ad_copy', '')}")
            print(f"    PROMO: {ad.get('promo_suggestion', '')}")
        else:
            print(f"  → (routine play, skipped)")

        results.append(ad)

        # Incremental save
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)

    significant = sum(1 for r in results if r.get("is_significant"))
    print(f"\nDone — {significant}/{len(results)} events generated ads → {output_json}")
    return results


if __name__ == "__main__":
    # Load events from Gemini analysis
    events_path = Path(__file__).resolve().parent / "results.json"
    if not events_path.exists():
        print("No results.json found. Run understanding.py first.")
        raise SystemExit(1)

    events = json.loads(events_path.read_text())
    process_all_events(
        events,
        business_name="MVP Pizza",
        business_type="pizza restaurant",
        output_json="ad_results.json",
    )

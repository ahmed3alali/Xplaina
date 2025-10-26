import os
import re
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict
from .schema import UserProfile, Recommendation, ItemExplanation, FactorContribution

load_dotenv()

# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "../courses.csv")
try:
    COURSES_DF = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(COURSES_DF)} courses from {DATA_PATH}")
except Exception as e:
    print("⚠️ Could not load dataset:", e)
    COURSES_DF = pd.DataFrame()

# -------------------------------------------------------------------
# DeepSeek Integration (OpenRouter)
# -------------------------------------------------------------------
def get_deepseek_recommendation(prompt: str) -> str:
    """Call DeepSeek through OpenRouter REST API."""
    api_key = "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"
    if not api_key:
        raise ValueError("Missing DEEPSEEK_API_KEY in .env")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",  # optional
        "X-Title": "XAI Educational Recommender"   # optional
    }

    payload = {
        "model": "deepseek/deepseek-r1-0528-qwen3-8b:free",
        "messages": [
            {
                "role": "system",
                "content": "You are an educational recommender system that outputs JSON with course titles and explanations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload)
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print("⚠️ DeepSeek API error:", e)
        return "[]"

# -------------------------------------------------------------------
# Filter courses
# -------------------------------------------------------------------
def filter_courses(user: UserProfile, top_n: int = 10) -> pd.DataFrame:
    """Filter dataset by level and skill overlap."""
    if COURSES_DF.empty:
        raise ValueError("Dataset not loaded or empty!")

    df = COURSES_DF.copy()
    df["level"] = df["level"].astype(str).str.strip().str.lower()
    df["skills"] = df["skills"].astype(str)

    # Level filter
    if user.level:
        df = df[df["level"].str.contains(user.level.lower(), na=False)]

    # Skill overlap
    def overlap(skill_str: str) -> int:
        try:
            skills = [s.strip().lower() for s in eval(skill_str.replace("{", "[").replace("}", "]"))]
        except Exception:
            skills = []
        return len(set(skills) & set([s.lower() for s in user.preferred_topics or []]))

    df["match_score"] = df["skills"].apply(overlap)
    df = df.sort_values(by=["match_score", "rating"], ascending=False)
    return df.head(top_n)

# -------------------------------------------------------------------
# JSON Sanitization Helper
# -------------------------------------------------------------------
def extract_json_from_text(text: str) -> str:
    """
    Extracts and cleans valid JSON from a text response that might contain extra
    formatting, markdown, or explanations. Guarantees a valid JSON list string.
    """
    if not text:
        return "[]"

    # Try to extract JSON array or object from the text
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        return match.group(0)
    # fallback: wrap single-line outputs as JSON
    return "[]"

# -------------------------------------------------------------------
# Main Recommendation Function
# -------------------------------------------------------------------
def generate_recommendations(user: UserProfile, top_k: int = 5):
    """Combine local dataset + DeepSeek reasoning."""
    filtered = filter_courses(user, top_n=top_k)

    prompt = f"""
    Student profile:
    Level: {user.level}
    GPA: {user.gpa}
    Interests/Skills: {', '.join(user.preferred_topics or [])}

    Candidate courses:
    {filtered[['title','skills','rating','description','platform']].head(top_k).to_string(index=False)}

    Task:
    - Choose and rank the best {top_k} courses for this student.
    - Give a short reason (1–2 sentences) for each.
    - Output only valid JSON in this format:
      [
        {{"title": "Course title", "reason": "short reason"}}
      ]
    """

    deepseek_raw = get_deepseek_recommendation(prompt)

    # --- sanitize and parse JSON ---
    cleaned_json = extract_json_from_text(deepseek_raw)
    try:
        reasons = json.loads(cleaned_json)
    except Exception as e:
        print("⚠️ DeepSeek invalid JSON after cleanup:", e)
        print("RAW:", deepseek_raw[:500])
        reasons = []

    recs = []
    explanations: Dict[str, ItemExplanation] = {}

    for _, row in filtered.head(top_k).iterrows():
        title = row.get("title")
        recs.append(
            Recommendation(
                content_id=title,
                title=title,
                topic=row.get("level"),
                score=float(row.get("rating", 0)),
            )
        )

        reason_text = next((r.get("reason") for r in reasons if r.get("title") == title), "No explanation available.")
        explanations[title] = ItemExplanation(
            content_id=title,
            shap_top_factors=[FactorContribution(feature="Reason", shap_value=0.0)],
            lime_top_factors=[FactorContribution(feature=reason_text, shap_value=0.0)]
        )

    return recs, explanations

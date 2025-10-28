import os
import re
import json
import requests
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict, Tuple
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
# Explanation Engine
# -------------------------------------------------------------------

def analyze_course_match(user: UserProfile, course_row) -> Tuple[float, List[str], Dict[str, float]]:
    """
    Analyze why a course matches the user's profile.
    Returns: (match_score, reasons, feature_contributions)
    """
    reasons = []
    feature_contributions = {}
    
    # Extract course data
    title = course_row.get('title', '')
    course_level = str(course_row.get('level', '')).lower()
    rating = float(course_row.get('rating', 0))
    skills_str = course_row.get('skills', '[]')
    
    # Parse skills
    try:
        if skills_str.startswith('[') and skills_str.endswith(']'):
            course_skills = [s.strip().lower() for s in eval(skills_str)]
        else:
            course_skills = [s.strip().lower() for s in skills_str.split(',')]
    except:
        course_skills = []
    
    user_level = user.level.lower() if user.level else ''
    user_interests = [interest.lower() for interest in (user.preferred_topics or [])]
    
    # 1. Level Matching Analysis
    level_score = 0.0
    if user_level and course_level:
        if user_level in course_level or course_level in user_level:
            level_score = 1.0
            reasons.append(f"Perfect match for your {user_level} level")
            feature_contributions["level_match"] = 0.3
        elif any(term in course_level for term in [user_level, 'all levels', 'beginner to advanced']):
            level_score = 0.8
            reasons.append(f"Suitable for {user_level} level learners")
            feature_contributions["level_suitable"] = 0.2
    
    # 2. Interest/Skill Matching Analysis
    interest_score = 0.0
    matched_interests = set(user_interests) & set(course_skills)
    if matched_interests:
        interest_score = min(len(matched_interests) / 3, 1.0)  # Normalize
        interest_list = list(matched_interests)[:3]  # Top 3 matches
        reasons.append(f"Matches your interests in: {', '.join(interest_list)}")
        feature_contributions["interest_match"] = interest_score * 0.4
    
    # 3. Rating Quality Analysis
    rating_score = 0.0
    if rating >= 4.5:
        rating_score = 1.0
        reasons.append("Excellent rating (4.5+ stars)")
        feature_contributions["high_rating"] = 0.2
    elif rating >= 4.0:
        rating_score = 0.7
        reasons.append("High quality rating (4.0+ stars)")
        feature_contributions["good_rating"] = 0.15
    elif rating >= 3.5:
        rating_score = 0.5
        reasons.append("Solid user ratings")
        feature_contributions["decent_rating"] = 0.1
    
    # 4. Content Relevance Analysis
    content_score = 0.0
    title_lower = title.lower()
    
    # Check if course title contains user interests
    title_matches = [interest for interest in user_interests if interest in title_lower]
    if title_matches:
        content_score = 0.8
        reasons.append(f"Course focuses on: {', '.join(title_matches[:2])}")
        feature_contributions["content_relevance"] = 0.2
    
    # Calculate overall match score
    total_score = (
        level_score * 0.3 +
        interest_score * 0.4 +
        rating_score * 0.2 +
        content_score * 0.1
    )
    
    # Add comprehensive summary reason
    if total_score >= 0.8:
        reasons.insert(0, "Excellent match for your profile")
    elif total_score >= 0.6:
        reasons.insert(0, "Strong match for your learning goals")
    else:
        reasons.insert(0, "Good learning opportunity")
    
    return total_score, reasons, feature_contributions

def generate_detailed_explanations(user: UserProfile, courses_df: pd.DataFrame) -> List[dict]:
    """Generate detailed explanations for why courses were selected"""
    explanations = []
    
    for _, course in courses_df.iterrows():
        match_score, reasons, features = analyze_course_match(user, course)
        
        explanation = {
            "title": course.get('title', ''),
            "match_score": match_score,
            "reasons": reasons,
            "feature_contributions": features,
            "details": {
                "user_level": user.level,
                "user_interests": user.preferred_topics,
                "course_level": course.get('level', ''),
                "course_rating": course.get('rating', 0),
                "matched_skills": list(set([i.lower() for i in (user.preferred_topics or [])]) & 
                                    set([s.lower() for s in eval(course.get('skills', '[]'))]))
            }
        }
        explanations.append(explanation)
    
    return explanations

# -------------------------------------------------------------------
# Filter courses with scoring
# -------------------------------------------------------------------
def filter_and_score_courses(user: UserProfile, top_n: int = 10) -> pd.DataFrame:
    """Filter courses and calculate match scores"""
    if COURSES_DF.empty:
        raise ValueError("Dataset not loaded or empty!")

    df = COURSES_DF.copy()
    df["level"] = df["level"].astype(str).str.strip().str.lower()
    df["skills"] = df["skills"].astype(str)

    # Level filter
    if user.level:
        user_level = user.level.lower().strip()
        df = df[df["level"].str.contains(user_level, na=False)]

    # Calculate match scores for each course
    match_scores = []
    detailed_reasons = []
    
    for _, row in df.iterrows():
        match_score, reasons, _ = analyze_course_match(user, row)
        match_scores.append(match_score)
        detailed_reasons.append(" | ".join(reasons[:2]))  # Top 2 reasons
    
    df["match_score"] = match_scores
    df["match_reason"] = detailed_reasons
    
    # Sort by match score and rating
    df = df.sort_values(by=["match_score", "rating"], ascending=[False, False])
    
    print(f"📊 Top course matches:")
    for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
        print(f"   {i}. {row['title'][:50]}... (Score: {row['match_score']:.2f})")
        print(f"      Reason: {row['match_reason']}")
    
    return df.head(top_n)

# -------------------------------------------------------------------
# Main Recommendation Function with Explanations
# -------------------------------------------------------------------
def generate_recommendations(user: UserProfile, top_k: int = 5):
    """Generate recommendations with detailed explanations"""
    print(f"🎯 Generating recommendations for: {user.student_id}")
    print(f"📚 Profile - Level: {user.level}, Interests: {user.preferred_topics}")
    
    # Get filtered and scored courses
    filtered_df = filter_and_score_courses(user, top_n=top_k)
    
    if filtered_df.empty:
        print("❌ No suitable courses found")
        return [], {}

    # Generate detailed explanations
    detailed_explanations = generate_detailed_explanations(user, filtered_df.head(top_k))
    
    recs = []
    explanations = {}

    for i, (_, row) in enumerate(filtered_df.head(top_k).iterrows()):
        title = row.get("title", "Unknown")
        
        # Create recommendation
        recs.append(
            Recommendation(
                content_id=title,
                title=title,
                topic=row.get("level", "general"),
                score=float(row.get("rating", 0)),  # Show original rating
            )
        )

        # Find detailed explanation for this course
        course_explanation = next(
            (exp for exp in detailed_explanations if exp["title"] == title), 
            None
        )
        
        if course_explanation:
            # Convert feature contributions to SHAP factors
            shap_factors = []
            for feature, value in course_explanation["feature_contributions"].items():
                shap_factors.append(FactorContribution(feature=feature, shap_value=value))
            
            # Create comprehensive reason text
            reason_text = " | ".join(course_explanation["reasons"])
            
            explanations[title] = ItemExplanation(
                content_id=title,
                shap_top_factors=shap_factors[:3],  # Top 3 factors
                lime_top_factors=[FactorContribution(feature=reason_text, shap_value=course_explanation["match_score"])]
            )
            
            print(f"✅ Course {i+1}: {title[:40]}...")
            print(f"   Match Score: {course_explanation['match_score']:.2f}")
            print(f"   Reasons: {reason_text}")
        else:
            # Fallback explanation
            explanations[title] = ItemExplanation(
                content_id=title,
                shap_top_factors=[FactorContribution(feature="Course Quality", shap_value=0.5)],
                lime_top_factors=[FactorContribution(feature="Recommended based on your learning profile", shap_value=0.5)]
            )

    print(f"🎉 Generated {len(recs)} recommendations with detailed explanations")
    return recs, explanations
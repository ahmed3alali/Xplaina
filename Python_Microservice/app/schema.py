from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class UserProfile(BaseModel):
    student_id: str
    level: Optional[str] = Field(None, description="e.g., beginner | intermediate | advanced")
    gpa: Optional[float] = None
    preferred_topics: List[str] = []

class ContentItem(BaseModel):
    content_id: str
    title: str
    description: str
    skills: List[str]=[]
    topic: str
    difficulty: Optional[str] = None
    duration_minutes: Optional[int] = None

class RecommendRequest(BaseModel):
    user: UserProfile
    catalog: List[ContentItem]
    top_k: int = 5

class FactorContribution(BaseModel):
    feature: str
    shap_value: float

class ItemExplanation(BaseModel):
    content_id: str
    shap_top_factors: List[FactorContribution]
    lime_top_factors: List[FactorContribution]

class Recommendation(BaseModel):
    content_id: str
    score: float
    title: str
    topic: str

class RecommendResponse(BaseModel):
    user_id: str
    recommendations: List[Recommendation]
    explanations: Dict[str, ItemExplanation]